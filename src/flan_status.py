"""Fine-tuned Flan-T5 status classifier (the recommended model, H9/H12) and a TF-IDF -> Flan
cascade (H13), usable inside ``src.pipeline.SmokingStatusPipeline``.

The model answers Flan's own multiple-choice prompt; prediction is rank classification over the
four answer strings, so the output is always a valid label. Training/scoring code moved here
verbatim from ``scripts/run_h9_flan_cv.py`` (experiments import it from here).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

def load_tokenizer(model_dir: str):
    """SentencePiece T5 tokenizer when the folder has one (Flan/T5 v1.1; unchanged from H9),
    otherwise the folder's own tokenizer (e.g. ByT5's byte tokenizer)."""
    if (Path(model_dir) / "spiece.model").exists():
        from transformers import T5Tokenizer

        return T5Tokenizer.from_pretrained(model_dir, legacy=False)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


ANSWERS = {"CURRENT": "current smoker", "FORMER": "former smoker", "NEVER": "never smoked", "UNKNOWN": "unknown"}
# Flan's own multiple-choice format (fixed on 8 hand-written policy cases, not on any eval data).
PROMPT = (
    "{text}\n\nBased on the text above, what is the patient's own cigarette smoking status?\n"
    "OPTIONS:\n- current smoker\n- former smoker\n- never smoked\n- unknown"
)


def score_options(tok, model, texts: list[str], batch: int = 16, calibrate: bool = False) -> np.ndarray:
    """Rank classification: sum of target-token log-probs for each answer string.

    With ``calibrate`` the score of each answer given a content-free input ("N/A") is
    subtracted (contextual calibration, Zhao et al. 2021) to remove answer-prior bias.
    """
    import torch

    labels = list(ANSWERS)
    targets = tok([ANSWERS[label] for label in labels], return_tensors="pt", padding=True)
    preds = np.empty(len(texts), dtype=object)
    prior = None
    if calibrate:
        calibrate = False
        prior = _raw_scores(tok, model, ["N/A"], targets, labels)[0]
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            enc = tok([PROMPT.format(text=t) for t in chunk], return_tensors="pt", padding=True, truncation=True, max_length=256)
            encoder_out = model.get_encoder()(**enc)
            scores = torch.zeros(len(chunk), len(labels))
            for j in range(len(labels)):
                tgt = targets.input_ids[j : j + 1].repeat(len(chunk), 1)
                mask = targets.attention_mask[j : j + 1].repeat(len(chunk), 1)
                tgt_labels = tgt.masked_fill(mask == 0, -100)
                out = model(encoder_outputs=encoder_out, attention_mask=enc.attention_mask, labels=tgt_labels)
                logp = torch.log_softmax(out.logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                scores[:, j] = (logp * mask).sum(-1)
            if prior is not None:
                scores = scores - prior
            for k, row in enumerate(scores.argmax(-1).tolist()):
                preds[start + k] = labels[row]
    return preds


def _raw_scores(tok, model, texts, targets, labels):
    import torch

    with torch.no_grad():
        enc = tok([PROMPT.format(text=t) for t in texts], return_tensors="pt", padding=True)
        encoder_out = model.get_encoder()(**enc)
        scores = torch.zeros(len(texts), len(labels))
        for j in range(len(labels)):
            tgt = targets.input_ids[j : j + 1].repeat(len(texts), 1)
            mask = targets.attention_mask[j : j + 1].repeat(len(texts), 1)
            out = model(encoder_outputs=encoder_out, attention_mask=enc.attention_mask, labels=tgt.masked_fill(mask == 0, -100))
            logp = torch.log_softmax(out.logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            scores[:, j] = (logp * mask).sum(-1)
    return scores


def finetune(model_dir: str, texts: list[str], labels: list[str], *, epochs: int, seed: int, lr: float, batch: int = 16, optimizer: str = "adamw"):
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    tok = load_tokenizer(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.train()
    # Oversample minority classes to the size of the largest one (labels are strings here).
    df = pd.DataFrame({"t": texts, "y": labels})
    top = df["y"].value_counts().max()
    df = pd.concat([g.sample(top, replace=True, random_state=seed) for _, g in df.groupby("y")], ignore_index=True)
    order = np.random.default_rng(seed).permutation(len(df))
    if optimizer == "adafactor":
        from transformers.optimization import Adafactor

        opt = Adafactor(model.parameters(), lr=lr, relative_step=False, scale_parameter=False, warmup_init=False)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(df) / batch))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(df))
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            enc = tok([PROMPT.format(text=df.t.iloc[i]) for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=256)
            tgt = tok([ANSWERS[df.y.iloc[i]] for i in idx], return_tensors="pt", padding=True).input_ids
            tgt = tgt.masked_fill(tgt == tok.pad_token_id, -100)
            loss = model(**enc, labels=tgt).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    model.eval()
    return tok, model


class FlanStatusClassifier:
    """Pluggable classifier: ``predict(list[str]) -> list[str]``. Pickles as a path, not weights."""

    def __init__(self, model_dir: str | Path, *, threads: int | None = None):
        self.model_dir = str(model_dir)
        self.threads = threads
        self._tok = self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            if self.threads:
                torch.set_num_threads(self.threads)
            self._tok = load_tokenizer(self.model_dir)
            self._model = T5ForConditionalGeneration.from_pretrained(self.model_dir).eval()
        return self._tok, self._model

    def predict(self, texts) -> list[str]:
        texts = list(texts)
        if not texts:
            return []
        tok, model = self._load()
        return [str(p) for p in score_options(tok, model, texts)]

    def __getstate__(self):
        return {"model_dir": self.model_dir, "threads": self.threads}

    def __setstate__(self, state):
        self.__init__(state["model_dir"], threads=state.get("threads"))

    @classmethod
    def train(cls, base_model_dir: str | Path, texts, labels, out_dir: str | Path, *, epochs: int = 3, seed: int = 42, lr: float = 3e-4):
        tok, model = finetune(str(base_model_dir), list(texts), list(labels), epochs=epochs, seed=seed, lr=lr)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        return cls(out_dir)


class CascadeClassifier:
    """TF-IDF SVM first; sentences whose top-2 decision margin is below ``threshold`` go to Flan.

    ``fit_threshold`` sets the threshold so that about ``share`` of the *training* sentences would
    be routed, using out-of-fold margins (H13 explored share 0.4; exploratory, not confirmed).
    """

    def __init__(self, tfidf, flan: FlanStatusClassifier, threshold: float):
        self.tfidf, self.flan, self.threshold = tfidf, flan, float(threshold)
        self.last_routed_share: float | None = None

    @staticmethod
    def _margins(model, texts) -> np.ndarray:
        scores = model.decision_function(list(texts))
        top2 = np.sort(scores, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    @staticmethod
    def fit_threshold(make_tfidf, texts, labels, share: float = 0.4, seed: int = 0) -> float:
        from sklearn.model_selection import StratifiedKFold

        texts, labels = np.asarray(list(texts), dtype=object), np.asarray(list(labels), dtype=object)
        margins = np.empty(len(texts))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(texts, labels):
            margins[te] = CascadeClassifier._margins(make_tfidf().fit(texts[tr], labels[tr]), texts[te])
        return float(np.quantile(margins, share))

    def predict(self, texts) -> list[str]:
        texts = list(texts)
        if not texts:
            return []
        pred = np.asarray(self.tfidf.predict(texts), dtype=object)
        route = np.where(self._margins(self.tfidf, texts) < self.threshold)[0]
        self.last_routed_share = len(route) / len(texts)
        if len(route):
            pred[route] = self.flan.predict([texts[i] for i in route])
        return [str(p) for p in pred]
