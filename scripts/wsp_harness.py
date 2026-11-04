#!/usr/bin/env python3
"""H19 Weight-Space Programming harness (protocol: experiments/h19-iterative-wsp/protocol.md).

Library used by the editor's turn scripts. Gives:
  * ``weights()``            — the base model's tensors (read-only copies) for analysis
  * ``save_edit(name, t)``   — store an edited tensor (refuses forbidden tensors / pure rescaling)
  * ``forward(...)``         — the ONLY way to run the model; counts against the 50-pass budget
  * ``note(text)``           — the plain-English explanation of a step, appended to the log

The model is never run anywhere else. No gradients: every model call is under torch.no_grad()
and parameters have requires_grad=False.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_h7_ood import add_evidence, load_sdoh  # noqa: E402
from src.flan_status import ANSWERS, PROMPT, load_tokenizer, score_options  # noqa: E402
from src.metrics import metrics_payload  # noqa: E402

MODEL_DIR = "/home/user/ext/models/flan-t5-large"
SDOH_DIR = "/home/user/ext/datasets/SDOH-NLI"
RUN = REPO_ROOT / "experiments/h19-iterative-wsp"
EDIT_DIR = RUN / "edit"
LOG = RUN / "log"
BUDGET = 50
FORBIDDEN = ("shared.weight", "lm_head.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight",
             "encoder.final_layer_norm.weight", "decoder.final_layer_norm.weight")

_model = _tok = _base = None


def _state() -> dict:
    p = LOG / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"passes_used": 0, "edits_saved": 0}


def _write_state(s: dict) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "state.json").write_text(json.dumps(s, indent=1))


def _append(kind: str, rec: dict) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **rec}
    with open(LOG / "calls.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def note(text: str) -> None:
    _append("explanation", {"text": text})


def dev_data():
    """Development premises only (H9 folds 1-4). Evaluation data is never loaded here."""
    from sklearn.model_selection import StratifiedKFold

    s = add_evidence(load_sdoh(Path(SDOH_DIR), "train")).reset_index(drop=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(s, s["label_native"].to_numpy()))
    dev_idx = np.concatenate([folds[k][1] for k in (1, 2, 3, 4)])
    return s.iloc[np.sort(dev_idx)].reset_index(drop=True)


def _load():
    global _model, _tok, _base
    if _model is None:
        import torch
        from transformers import T5ForConditionalGeneration

        torch.set_num_threads(4)
        _tok = load_tokenizer(MODEL_DIR)
        _model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).eval()
        for p in _model.parameters():
            p.requires_grad_(False)
        _base = {k: v.clone() for k, v in _model.state_dict().items()}
    return _tok, _model


def weights() -> dict:
    """Read-only copies of the base tensors (numpy float32)."""
    _load()
    return {k: v.numpy().copy() for k, v in _base.items()}


def current_edit() -> dict:
    out = {}
    if (EDIT_DIR / "manifest.json").exists():
        for name in json.loads((EDIT_DIR / "manifest.json").read_text()):
            out[name] = np.load(EDIT_DIR / f"{name}.npy")
    return out


def save_edit(name: str, tensor: np.ndarray, why: str) -> None:
    _load()
    if name in FORBIDDEN or "layer_norm" in name and "final" in name:
        raise PermissionError(f"edits to {name} are not allowed")
    base = _base[name].numpy()
    if tensor.shape != base.shape:
        raise ValueError("shape mismatch")
    ratio = np.divide(tensor, base, out=np.full_like(base, np.nan), where=base != 0)
    if np.nanstd(ratio) < 1e-6:
        raise PermissionError("edits that only multiply a whole tensor by a constant are refused")
    EDIT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EDIT_DIR / f"{name}.npy", tensor.astype(np.float32))
    man = json.loads((EDIT_DIR / "manifest.json").read_text()) if (EDIT_DIR / "manifest.json").exists() else []
    if name not in man:
        man.append(name)
    (EDIT_DIR / "manifest.json").write_text(json.dumps(man, indent=1))
    changed = int((np.abs(tensor - base) > 0).sum())
    s = _state()
    s["edits_saved"] += 1
    _write_state(s)
    _append("save_edit", {"name": name, "changed_entries": changed, "why": why})


def drop_edit(name: str, why: str) -> None:
    man = json.loads((EDIT_DIR / "manifest.json").read_text()) if (EDIT_DIR / "manifest.json").exists() else []
    if name in man:
        man.remove(name)
        (EDIT_DIR / f"{name}.npy").unlink(missing_ok=True)
        (EDIT_DIR / "manifest.json").write_text(json.dumps(man, indent=1))
    _append("drop_edit", {"name": name, "why": why})


def _apply(edit: dict | None):
    import torch

    tok, model = _load()
    sd = model.state_dict()
    for k, v in _base.items():
        if edit and k in edit:
            sd[k].copy_(torch.from_numpy(edit[k]))
        elif not torch.equal(sd[k], v):
            sd[k].copy_(v)
    return tok, model


def forward(texts, *, which: str = "edited", mode: str = "predict", hooks=None, why: str = "", calibrate: bool = True):
    """The only way to run the model. One call = one pass (``which='both'`` = original and edited on
    the same texts, still one pass). mode='predict' returns labels and per-option scores;
    mode='activations' runs the encoder (+ the decoder for the first answer token) with ``hooks``,
    a dict name -> callable(module_output) collecting what the editor asks for."""
    import torch

    texts = list(texts)
    if len(texts) > 512:
        raise ValueError("at most 512 texts per pass")
    s = _state()
    if s["passes_used"] >= BUDGET:
        raise RuntimeError("forward-pass budget used up")
    s["passes_used"] += 1
    _write_state(s)
    out = {}
    versions = ("original", "edited") if which == "both" else (which,)
    for v in versions:
        tok, model = _apply(current_edit() if v == "edited" else None)
        if mode == "predict":
            with torch.no_grad():
                out[v] = list(score_options(tok, model, texts, calibrate=calibrate))
        else:
            collected = {k: [] for k in (hooks or {})}
            handles = []
            for name, fn in (hooks or {}).items():
                module = dict(model.named_modules())[name]
                handles.append(module.register_forward_hook(lambda m, i, o, n=name, f=fn: collected[n].append(f(i, o))))
            try:
                with torch.no_grad():
                    for t in texts:
                        enc = tok(PROMPT.format(text=t), return_tensors="pt", truncation=True, max_length=256)
                        dec = tok("unknown", return_tensors="pt").input_ids[:, :1]
                        model(**enc, decoder_input_ids=torch.zeros_like(dec))
            finally:
                for h in handles:
                    h.remove()
            out[v] = collected
    _append("forward", {"pass": s["passes_used"], "which": which, "mode": mode, "n": len(texts), "why": why})
    return out


def report(y, pred) -> dict:
    m = metrics_payload(np.asarray(y, dtype=object), np.asarray(pred, dtype=object))
    return {"macro_f1": round(m["macro_f1"], 4), "per_class_f1": {k: round(v["f1"], 3) for k, v in m["per_class"].items()},
            "confusion_rows_true_cols_pred": m["confusion_matrix_rows_true_cols_pred"], "columns": m["prediction_columns"]}
