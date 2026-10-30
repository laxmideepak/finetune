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
    return cache


def read_params(cache: Path) -> dict:
    import zarr

    params = {}
    for array_dir in sorted(p for p in cache.iterdir() if p.is_dir() and p.name.startswith("target.")):
        params[array_dir.name[len("target.") :]] = zarr.open(str(array_dir), mode="r")[...]
    return params


def to_hf_state(params: dict, n_enc: int, n_dec: int) -> dict:
    import numpy as np
    import torch

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    sd = {"shared.weight": t(params["token_embedder.embedding"])}
    for stack, n in (("encoder", n_enc), ("decoder", n_dec)):
        for i in range(n):
            p = f"{stack}.layers_{i}."
            b = f"{stack}.block.{i}.layer."
            sd[b + "0.layer_norm.weight"] = t(params[p + "pre_attention_layer_norm.scale" if stack == "encoder" else p + "pre_self_attention_layer_norm.scale"])
            att = "attention" if stack == "encoder" else "self_attention"
            for src, dst in (("query", "q"), ("key", "k"), ("value", "v"), ("out", "o")):
                sd[b + f"0.SelfAttention.{dst}.weight"] = t(params[p + f"{att}.{src}.kernel"].T)
            mlp = "1" if stack == "encoder" else "2"
            if stack == "decoder":
                sd[b + "1.layer_norm.weight"] = t(params[p + "pre_cross_attention_layer_norm.scale"])
                for src, dst in (("query", "q"), ("key", "k"), ("value", "v"), ("out", "o")):
                    sd[b + f"1.EncDecAttention.{dst}.weight"] = t(params[p + f"encoder_decoder_attention.{src}.kernel"].T)
            sd[b + f"{mlp}.layer_norm.weight"] = t(params[p + "pre_mlp_layer_norm.scale"])
            sd[b + f"{mlp}.DenseReluDense.wi_0.weight"] = t(params[p + "mlp.wi_0.kernel"].T)
            sd[b + f"{mlp}.DenseReluDense.wi_1.weight"] = t(params[p + "mlp.wi_1.kernel"].T)
            sd[b + f"{mlp}.DenseReluDense.wo.weight"] = t(params[p + "mlp.wo.kernel"].T)
        sd[f"{stack}.final_layer_norm.weight"] = t(params[f"{stack}.{stack}_norm.scale"])
        sd[f"{stack}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = t(params[f"{stack}.relpos_bias.rel_embedding"].T)
    sd["lm_head.weight"] = t(params["decoder.logits_dense.kernel"].T)
    return sd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=sorted(CHECKPOINTS), default="small")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args(argv)

    import torch
    from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

    prefix, dims = CHECKPOINTS[args.size]
    out = Path(args.out)
    cache = Path(args.cache) if args.cache else out.with_name(out.name + "-t5x")
    params = read_params(download(prefix, cache))
    config = T5Config(vocab_size=params["token_embedder.embedding"].shape[0], feed_forward_proj="gated-gelu", tie_word_embeddings=False,
                      dropout_rate=0.1, relative_attention_num_buckets=32, relative_attention_max_distance=128,
                      decoder_start_token_id=0, pad_token_id=0, eos_token_id=1, **dims)
    model = T5ForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(to_hf_state(params, dims["num_layers"], dims["num_decoder_layers"]), strict=False)
    missing = [m for m in missing if not m.endswith("embed_tokens.weight")]  # tied to shared.weight
    print(f"missing={missing} unexpected={unexpected}")
    out.mkdir(parents=True, exist_ok=True)
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
