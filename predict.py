"""Score a Dutch ruling with the two JustID BERTje models.

Loads local `models/rg` and `models/bk` when those folders have weights.
Otherwise it downloads `newnus/justid-rechtsgebieden` and
`newnus/justid-bijzondere-kenmerken` from the Hugging Face Hub.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

TaskScores = dict[str, float]


class JustIdModels:
    """Both classifiers, shared tokenizer settings, one `predict` call."""

    def __init__(
        self,
        rg_dir: Path | str | None = None,
        bk_dir: Path | str | None = None,
        threshold: float | None = None,
        max_len: int | None = None,
        device: str | None = None,
    ) -> None:
        self.rg_source = _resolve(rg_dir, config.RG_MODEL_DIR, config.RG_HUB_ID)
        self.bk_source = _resolve(bk_dir, config.BK_MODEL_DIR, config.BK_HUB_ID)
        self.threshold = config.LABEL_THRESHOLD if threshold is None else threshold
        self.max_len = config.MAX_LEN if max_len is None else max_len
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.rg_labels = _load_labels(self.rg_source)
        self.bk_labels = _load_labels(self.bk_source)
        self.rg_tokenizer = AutoTokenizer.from_pretrained(self.rg_source)
        self.bk_tokenizer = AutoTokenizer.from_pretrained(self.bk_source)
        self.rg_model = _load_model(self.rg_source, len(self.rg_labels), self.device)
        self.bk_model = _load_model(self.bk_source, len(self.bk_labels), self.device)

    def predict(self, text: str, threshold: float | None = None) -> dict[str, TaskScores]:
        """Return kept labels with scores for both tasks."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        cut = self.threshold if threshold is None else threshold
        return {
            "rechtsgebieden": self._score(text, self.rg_tokenizer, self.rg_model, self.rg_labels, cut),
            "bijzondere_kenmerken": self._score(
                text, self.bk_tokenizer, self.bk_model, self.bk_labels, cut
            ),
        }

    def _score(
        self,
        text: str,
        tokenizer,
        model,
        labels: list[str],
        threshold: float,
    ) -> TaskScores:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        if isinstance(probs, float):
            probs = [probs]
        kept = {label: round(float(score), 4) for label, score in zip(labels, probs) if score >= threshold}
        return dict(sorted(kept.items(), key=lambda item: item[1], reverse=True))


def _resolve(override: Path | str | None, local: Path, hub_id: str) -> str:
    if override is not None:
        return str(override)
    if (local / "model.safetensors").exists() or (local / "pytorch_model.bin").exists():
        return str(local)
    return hub_id


def _load_labels(source: str) -> list[str]:
    path = Path(source) / "labels.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    downloaded = hf_hub_download(source, "labels.json")
    return json.loads(Path(downloaded).read_text(encoding="utf-8"))


def _load_model(source: str, n_labels: int, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=n_labels,
        problem_type="multi_label_classification",
    )
    model.to(device)
    model.eval()
    return model


def main() -> None:
    print("Loading models")
    bundle = JustIdModels()
    print("device:", bundle.device)
    print("Paste a ruling, then a blank line:\n")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines:
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    out = bundle.predict(text)
    print("\nrechtsgebieden:", out["rechtsgebieden"])
    print("bijzondere_kenmerken:", out["bijzondere_kenmerken"])


if __name__ == "__main__":
    main()
