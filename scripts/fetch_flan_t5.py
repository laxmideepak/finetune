#!/usr/bin/env python3
"""Download a Flan-T5 T5X checkpoint from the public `t5-data` GCS bucket and convert it
to a Hugging Face `T5ForConditionalGeneration` folder (no Hugging Face Hub needed).

Only the model parameters (`target.*` zarr arrays) are fetched, not optimizer state.
Mapping follows the T5 v1.1 / Flan-T5 layout (gated-GELU MLP, untied LM head, relative
position bias in layer 0 of each stack).

    python scripts/fetch_flan_t5.py --size small --out ../models/flan-t5-small
    python scripts/fetch_flan_t5.py --size base  --out ../models/flan-t5-base

Sanity check after conversion: the script asks the model to translate a short English
sentence to German and prints the answer.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "https://storage.googleapis.com"
LISTING = "https://storage.googleapis.com/storage/v1/b/t5-data/o"
CHECKPOINTS = {
    "small": ("pretrained_models/t5x/flan_t5_small/checkpoint_1198000", dict(d_model=512, d_ff=1024, num_layers=8, num_decoder_layers=8, num_heads=6, d_kv=64)),
    "base": ("pretrained_models/t5x/flan_t5_base/checkpoint_1184000", dict(d_model=768, d_ff=2048, num_layers=12, num_decoder_layers=12, num_heads=12, d_kv=64)),
    "large": ("pretrained_models/t5x/flan_t5_large/checkpoint_1164000", dict(d_model=1024, d_ff=2816, num_layers=24, num_decoder_layers=24, num_heads=16, d_kv=64)),
}
VOCAB = "t5-data/vocabs/cc_all.32000/sentencepiece.model"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def list_objects(prefix: str) -> list[str]:
    names, token = [], None
    while True:
        query = {"prefix": prefix, "fields": "items(name),nextPageToken", "maxResults": "1000"}
        if token:
            query["pageToken"] = token
        page = json.loads(_get(LISTING + "?" + urllib.parse.urlencode(query)))
        names += [item["name"] for item in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            return names


def download(prefix: str, cache: Path) -> Path:
    names = [n for n in list_objects(prefix + "/target.") if "/target." in n]
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> None:
        dest = cache / name[len(prefix) + 1 :]
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_get(f"{BUCKET}/t5-data/{name}"))

    with cf.ThreadPoolExecutor(8) as pool:
        list(pool.map(fetch, names))
    index = cache / "checkpoint"
    if not index.exists():  # small parameters (norms, position biases) may live inline here
        index.write_bytes(_get(f"{BUCKET}/t5-data/{prefix}/checkpoint"))
    return cache


def inline_params(cache: Path) -> dict:
    """Arrays stored inline in the T5X msgpack index (some checkpoints keep norms/relpos there)."""
    import msgpack
    import numpy as np

    index = cache / "checkpoint"
    if not index.exists():
        return {}

    def hook(code, data):
        if code == 1:
            shape, dtype, buf = msgpack.unpackb(data, raw=False)
            return np.frombuffer(buf, dtype=dtype).reshape(shape)
        return msgpack.ExtType(code, data)

    tree = msgpack.unpackb(index.read_bytes(), ext_hook=hook, raw=False, strict_map_key=False)
    target = tree.get("target") or tree.get("optimizer", {}).get("target", {})
    out = {}

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, np.ndarray):
            out[path] = node

    walk(target, "")
    return out


class LazyParams(dict):
    """Maps parameter name -> zarr array, read only when indexed (keeps peak memory low)."""

    def __init__(self, cache: Path):
        super().__init__()
        self._dirs = {p.name[len("target.") :]: p for p in cache.iterdir() if p.is_dir() and p.name.startswith("target.")}
        self._inline = inline_params(cache)

    def __getitem__(self, key):
        import zarr

        if key in self._dirs:
            return zarr.open(str(self._dirs[key]), mode="r")[...]
        return self._inline[key]

    def __contains__(self, key):
        return key in self._dirs or key in self._inline

    def shape(self, key):
        import zarr

        if key in self._dirs:
            return zarr.open(str(self._dirs[key]), mode="r").shape
        return self._inline[key].shape


def read_params(cache: Path) -> LazyParams:
    return LazyParams(cache)


def name_map(n_enc: int, n_dec: int) -> list[tuple[str, str, bool]]:
    """(hf_name, t5x_name, transpose) for every Flan-T5 parameter."""
    m = [("shared.weight", "token_embedder.embedding", False)]
    for stack, n in (("encoder", n_enc), ("decoder", n_dec)):
        for i in range(n):
            p, b = f"{stack}.layers_{i}.", f"{stack}.block.{i}.layer."
            m.append((b + "0.layer_norm.weight", p + ("pre_attention_layer_norm.scale" if stack == "encoder" else "pre_self_attention_layer_norm.scale"), False))
            att = "attention" if stack == "encoder" else "self_attention"
            for src, dst in (("query", "q"), ("key", "k"), ("value", "v"), ("out", "o")):
                m.append((b + f"0.SelfAttention.{dst}.weight", p + f"{att}.{src}.kernel", True))
            mlp = "1" if stack == "encoder" else "2"
            if stack == "decoder":
                m.append((b + "1.layer_norm.weight", p + "pre_cross_attention_layer_norm.scale", False))
                for src, dst in (("query", "q"), ("key", "k"), ("value", "v"), ("out", "o")):
                    m.append((b + f"1.EncDecAttention.{dst}.weight", p + f"encoder_decoder_attention.{src}.kernel", True))
            m.append((b + f"{mlp}.layer_norm.weight", p + "pre_mlp_layer_norm.scale", False))
            for w in ("wi_0", "wi_1", "wo"):
                m.append((b + f"{mlp}.DenseReluDense.{w}.weight", p + f"mlp.{w}.kernel", True))
        m.append((f"{stack}.final_layer_norm.weight", f"{stack}.{stack}_norm.scale", False))
        m.append((f"{stack}.block.0.layer.0.SelfAttention.relative_attention_bias.weight", f"{stack}.relpos_bias.rel_embedding", True))
    m.append(("lm_head.weight", "decoder.logits_dense.kernel", True))
    return m


def load_streaming(model, params, n_enc: int, n_dec: int) -> list[str]:
    """Copy one array at a time into the model's parameters; returns HF names not filled."""
    import numpy as np
    import torch

    state = model.state_dict()
    filled = set()
    with torch.no_grad():
        for hf_name, t5x_name, transpose in name_map(n_enc, n_dec):
            array = params[t5x_name]
            array = array.T if transpose else array
            state[hf_name].copy_(torch.from_numpy(np.ascontiguousarray(array)))
            filled.add(hf_name)
            del array
    return [k for k in state if k not in filled and not k.endswith("embed_tokens.weight")]


def find_checkpoint(model_prefix: str) -> str:
    """Latest checkpoint_* folder under a t5-data T5X model prefix."""
    query = {"prefix": model_prefix.rstrip("/") + "/", "delimiter": "/", "fields": "prefixes"}
    prefixes = json.loads(_get(LISTING + "?" + urllib.parse.urlencode(query))).get("prefixes", [])
    ckpts = sorted((p for p in prefixes if "/checkpoint_" in p), key=lambda p: int(p.rstrip("/").rsplit("_", 1)[1]))
    if not ckpts:
        raise SystemExit(f"no checkpoint_* under {model_prefix}")
    return ckpts[-1].rstrip("/")


def infer_dims(cache: Path) -> dict:
    """Read T5 v1.1 dimensions from the array shapes (zarr folders or inline index arrays)."""
    shape = LazyParams(cache).shape
    n_enc = len([p for p in cache.iterdir() if p.name.startswith("target.encoder.layers_") and p.name.endswith(".attention.query.kernel")])
    n_dec = len([p for p in cache.iterdir() if p.name.startswith("target.decoder.layers_") and p.name.endswith(".self_attention.query.kernel")])
    d_model, inner = shape("encoder.layers_0.attention.query.kernel")
    heads = shape("encoder.relpos_bias.rel_embedding")[0]
    return dict(d_model=d_model, d_ff=shape("encoder.layers_0.mlp.wi_0.kernel")[1], num_layers=n_enc,
                num_decoder_layers=n_dec, num_heads=heads, d_kv=inner // heads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=sorted(CHECKPOINTS), default="small")
    parser.add_argument("--model", default=None, help="any T5 v1.1-layout model under gs://t5-data/pretrained_models/t5x/ (e.g. t5_1_1_lm100k_base, byt5_small); overrides --size")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args(argv)

    import torch
    from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

    if args.model:
        prefix, dims = find_checkpoint(f"pretrained_models/t5x/{args.model}"), None
    else:
        prefix, dims = CHECKPOINTS[args.size]
    out = Path(args.out)
    cache = Path(args.cache) if args.cache else out.with_name(out.name + "-t5x")
    params = read_params(download(prefix, cache))
    dims = dims or infer_dims(cache)
    params = read_params(cache)
    print("dims", dims)
    import zarr

    vocab = params.shape("token_embedder.embedding")[0]
    config = T5Config(vocab_size=vocab, feed_forward_proj="gated-gelu", tie_word_embeddings=False,
                      dropout_rate=0.1, relative_attention_num_buckets=32, relative_attention_max_distance=128,
                      decoder_start_token_id=0, pad_token_id=0, eos_token_id=1, **dims)
    model = T5ForConditionalGeneration(config)
    missing = load_streaming(model, params, dims["num_layers"], dims["num_decoder_layers"])
    print(f"missing={missing}")
    out.mkdir(parents=True, exist_ok=True)
    if args.model and args.model.startswith("byt5"):
        from transformers import ByT5Tokenizer

        tokenizer = ByT5Tokenizer()
    else:
        (out / "spiece.model").write_bytes(_get(f"{BUCKET}/{VOCAB}"))
        tokenizer = T5Tokenizer(str(out / "spiece.model"), extra_ids=100, legacy=False)
    tokenizer.save_pretrained(str(out))
    model.save_pretrained(str(out))
    model.eval()
    with torch.no_grad():
        ids = tokenizer("Translate English to German: The patient stopped smoking last year.", return_tensors="pt").input_ids
        print("sanity:", tokenizer.decode(model.generate(ids, max_new_tokens=30)[0], skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
