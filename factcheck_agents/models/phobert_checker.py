"""PhoBERT ViFactCheck text classifier wrapper.

Loads the fine-tuned checkpoint produced by
``notebooks/pipeline/03.9_vifactcheck_training.ipynb``:

    <run>/best_model.pth            (state dict under "model_state_dict")
    <run>/tokenizer/               (saved AutoTokenizer)
    <run>/checkpoint_manifest.json (optional metadata)

The model scores (statement, evidence) -> {Supported, Refuted, NEI}. The
number of classes is inferred from the checkpoint so binary runs also load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..config import settings
from ..state import ModelResult

# label id -> human label for the three_class strategy used in stage 3.9
_LABELS_3 = {0: "SUPPORTED", 1: "REFUTED", 2: "NEI"}
_LABELS_2 = {0: "SUPPORTED", 1: "REFUTED"}


def _read_manifest_metric(run_dir: Path) -> float:
    """Extract the best validation metric from a run's checkpoint_manifest.json.

    Returns -1.0 if the manifest or metric is missing so that runs without
    manifests sort last but don't crash.
    """
    man = run_dir / "checkpoint_manifest.json"
    if not man.exists():
        return -1.0
    try:
        data = json.loads(man.read_text())
        metrics = data.get("best_metrics", {})
        # PhoBERT manifest uses val_macro_f1; COOLANT uses val_accuracy
        for key in (
            "best_val_macro_f1",
            "val_macro_f1",
            "val_accuracy",
            "best_val_accuracy",
        ):
            val = metrics.get(key)
            if val is not None:
                return float(val)
        # Fallback: check top-level selection_metric
        return -1.0
    except Exception:
        return -1.0


def _resolve_run_dir() -> Optional[Path]:
    """Explicit override wins; else pick the run with the best validation metric.

    Reads ``checkpoint_manifest.json`` from each run directory and selects the
    one with the highest ``best_val_macro_f1`` (or ``val_accuracy``). Falls back
    to newest-by-mtime if no manifests are found.
    """
    if settings.phobert_ckpt_dir:
        p = Path(settings.phobert_ckpt_dir)
        return p if (p / "best_model.pth").exists() else None

    root = settings.phobert_search_root()
    if not root.exists():
        return None
    runs = [d for d in root.iterdir() if d.is_dir() and (d / "best_model.pth").exists()]
    if not runs:
        return None

    # Try to pick by best metric from manifest
    scored = [(_read_manifest_metric(d), d) for d in runs]
    best_metric = max(s for s, _ in scored)
    if best_metric > 0:
        best_dir = max(scored, key=lambda pair: pair[0])[1]
        return best_dir

    # Fallback: newest by mtime
    return max(runs, key=lambda d: d.stat().st_mtime)


def _pick_device() -> str:
    import torch

    if settings.device != "auto":
        return settings.device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PhoBERTChecker:
    """Lazy singleton-style wrapper around the fine-tuned PhoBERT classifier."""

    def __init__(self) -> None:
        self._loaded = False
        self._model = None
        self._tokenizer = None
        self._labels = _LABELS_3
        self._max_length = 256
        self._device = "cpu"
        self._load_error: Optional[str] = None

    # ── loading ──────────────────────────────────────────────────────────
    def _build_model(self, backbone: str, num_classes: int, dropout: float):
        import torch.nn as nn
        from transformers import AutoModel

        class PhoBERTClassifier(nn.Module):
            def __init__(self, backbone_name, num_classes, dropout):
                super().__init__()
                self.backbone = AutoModel.from_pretrained(backbone_name)
                hidden = self.backbone.config.hidden_size
                self.classifier = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(hidden, num_classes)
                )

            def forward(self, input_ids, attention_mask):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]
                return self.classifier(cls)

        return PhoBERTClassifier(backbone, num_classes, dropout)

    def load(self) -> bool:
        if self._loaded:
            return True
        if self._load_error is not None:
            return False
        try:
            import torch
            from transformers import AutoTokenizer

            run_dir = _resolve_run_dir()
            if run_dir is None:
                self._load_error = (
                    "No PhoBERT checkpoint found. Set VIFACTCHECK_CKPT_DIR or place a run "
                    f"under {settings.phobert_search_root()}."
                )
                return False

            manifest = {}
            man_path = run_dir / "checkpoint_manifest.json"
            if man_path.exists():
                manifest = json.loads(man_path.read_text())

            backbone = (
                manifest.get("backbone")
                or manifest.get("model", {}).get("backbone")
                or "vinai/phobert-base-v2"
            )
            self._max_length = int(
                manifest.get("max_length")
                or manifest.get("data", {}).get("max_length")
                or 256
            )

            ckpt = torch.load(run_dir / "best_model.pth", map_location="cpu")
            state = (
                ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            )

            # infer num_classes from the classifier head weight
            num_classes = 3
            for key in ("classifier.1.weight", "classifier.weight"):
                if key in state:
                    num_classes = state[key].shape[0]
                    break
            self._labels = _LABELS_2 if num_classes == 2 else _LABELS_3

            dropout = float(
                manifest.get("dropout")
                or manifest.get("model", {}).get("dropout")
                or 0.3
            )

            self._device = _pick_device()
            model = self._build_model(backbone, num_classes, dropout)
            model.load_state_dict(state, strict=False)
            model.to(self._device).eval()
            self._model = model

            tok_dir = run_dir / "tokenizer"
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(tok_dir) if tok_dir.exists() else backbone
            )
            self._loaded = True
            return True
        except Exception as exc:  # pragma: no cover - defensive
            self._load_error = f"PhoBERT load failed: {exc}"
            return False

    # ── inference ─────────────────────────────────────────────────────────
    def predict(self, statement: str, evidence_text: str = "") -> ModelResult:
        if not self.load():
            return ModelResult(
                model="phobert_vifactcheck",
                available=False,
                note=self._load_error or "unavailable",
            )
        try:
            import torch
            import torch.nn.functional as F

            enc = self._tokenizer(
                statement,
                evidence_text or None,
                max_length=self._max_length,
                padding="max_length",
                truncation="only_second" if evidence_text else True,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(enc["input_ids"], enc["attention_mask"])
                probs = F.softmax(logits, dim=-1)[0].cpu().tolist()

            label_id = int(max(range(len(probs)), key=lambda i: probs[i]))
            prob_map = {
                self._labels.get(i, str(i)): round(p, 4) for i, p in enumerate(probs)
            }
            return ModelResult(
                model="phobert_vifactcheck",
                available=True,
                label=self._labels.get(label_id, str(label_id)),
                label_id=label_id,
                probabilities=prob_map,
                confidence=round(probs[label_id], 4),
                note="statement scored against retrieved evidence",
            )
        except Exception as exc:  # pragma: no cover - defensive
            return ModelResult(
                model="phobert_vifactcheck",
                available=False,
                note=f"inference error: {exc}",
            )


def build_evidence_text(evidence: List[dict], max_chars: int = 2000) -> str:
    """Concatenate evidence snippets into a single evidence passage."""
    parts, total = [], 0
    for e in evidence:
        snippet = (e.get("snippet") or "").strip()
        if not snippet:
            continue
        parts.append(snippet)
        total += len(snippet)
        if total >= max_chars:
            break
    return " ".join(parts)[:max_chars]
