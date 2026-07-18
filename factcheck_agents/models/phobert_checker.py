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

    Handles two manifest formats:
    - Enhanced (03.9_vifactcheck_training.ipynb): ``best_metrics.val_macro_f1``
    - Original (03.9_vifactcheck_original_training.ipynb): top-level ``best_dev_macro_f1``
    """
    man = run_dir / "checkpoint_manifest.json"
    if not man.exists():
        return -1.0
    try:
        data = json.loads(man.read_text())
        # Enhanced notebook: nested best_metrics dict
        metrics = data.get("best_metrics", {})
        for key in (
            "best_val_macro_f1",
            "val_macro_f1",
            "val_accuracy",
            "best_val_accuracy",
        ):
            val = metrics.get(key)
            if val is not None:
                return float(val)
        # Original notebook: top-level metric key
        for key in ("best_dev_macro_f1", "best_val_macro_f1"):
            val = data.get(key)
            if val is not None:
                return float(val)
        return -1.0
    except Exception:
        return -1.0


def _resolve_run_dir() -> Optional[Path]:
    """Explicit override wins; else pick the best run across both checkpoint roots.

    Searches both ``checkpoints_vifactcheck/`` (enhanced, 03.9 notebook) and
    ``checkpoints_vifactcheck_original/`` (original paper, 03.9_original notebook),
    picks the run with the highest validation macro-F1. Falls back to newest-by-mtime
    if no manifests are found.
    """
    # Explicit overrides (enhanced then original)
    for env_path in (settings.phobert_ckpt_dir, settings.phobert_original_ckpt_dir):
        if env_path:
            p = Path(env_path)
            return p if (p / "best_model.pth").exists() else None

    # Auto-discover from both checkpoint roots
    runs: list[Path] = []
    for root in (
        settings.phobert_search_root(),
        settings.phobert_original_search_root(),
    ):
        if root.exists():
            runs += [
                d
                for d in root.iterdir()
                if d.is_dir() and (d / "best_model.pth").exists()
            ]
    if not runs:
        return None

    scored = [(_read_manifest_metric(d), d) for d in runs]
    best_metric = max(s for s, _ in scored)
    if best_metric > 0:
        return max(scored, key=lambda pair: pair[0])[1]

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


def _detect_arch(manifest: dict, state: dict) -> str:
    """Return ``'cls'`` or ``'pooler'`` from manifest metadata or state dict keys.

    - ``'cls'``: 03.9_vifactcheck_training.ipynb — ``backbone.`` prefix, ``classifier.1.weight``
    - ``'pooler'``: 03.9_vifactcheck_original_training.ipynb — ``phobert.`` prefix, ``linear.weight``
    """
    arch_str = manifest.get("architecture", "")
    if "pooler" in arch_str.lower():
        return "pooler"
    if any(k.startswith("phobert.") for k in state):
        return "pooler"
    if "linear.weight" in state and not any(k.startswith("backbone.") for k in state):
        return "pooler"
    return "cls"


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
    def _build_cls_model(self, backbone: str, num_classes: int, dropout: float):
        """CLS-token classifier (03.9_vifactcheck_training.ipynb).

        State dict keys: ``backbone.*``, ``classifier.1.weight/bias``.
        """
        import torch.nn as nn
        from transformers import AutoModel

        class _CLSClassifier(nn.Module):
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

        return _CLSClassifier(backbone, num_classes, dropout)

    def _build_pooler_model(self, backbone: str, num_classes: int, dropout: float):
        """Pooler-output classifier (03.9_vifactcheck_original_training.ipynb).

        Mirrors the original ``PhoBERTClassifier`` from ``plm_training.py``.
        State dict keys: ``phobert.*``, ``linear.weight/bias``.
        """
        import torch.nn as nn
        from transformers import AutoModel

        class _PoolerClassifier(nn.Module):
            def __init__(self, backbone_name, num_classes, dropout):
                super().__init__()
                self.phobert = AutoModel.from_pretrained(backbone_name)
                self.dropout = nn.Dropout(dropout)
                self.linear = nn.Linear(self.phobert.config.hidden_size, num_classes)

            def forward(self, input_ids, attention_mask):
                _, pooled = self.phobert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
                return self.linear(self.dropout(pooled))

        return _PoolerClassifier(backbone, num_classes, dropout)

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
                    "No PhoBERT checkpoint found. Set VIFACTCHECK_CKPT_DIR / "
                    "VIFACTCHECK_ORIGINAL_CKPT_DIR or place a run under "
                    f"{settings.phobert_search_root()} or "
                    f"{settings.phobert_original_search_root()}."
                )
                return False

            manifest = {}
            man_path = run_dir / "checkpoint_manifest.json"
            if man_path.exists():
                manifest = json.loads(man_path.read_text())

            ckpt = torch.load(run_dir / "best_model.pth", map_location="cpu")
            state = (
                ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            )

            # backbone: prefer saved value in checkpoint, then manifest, then default
            backbone = (
                (ckpt.get("backbone") if isinstance(ckpt, dict) else None)
                or manifest.get("backbone")
                or manifest.get("model", {}).get("backbone")
                or "vinai/phobert-base-v2"
            )

            # max_length: check all known manifest locations
            self._max_length = int(
                manifest.get("max_length")
                or manifest.get("data", {}).get("max_length")
                or manifest.get("training_setup", {}).get("max_length")
                or 256
            )

            # num_classes: prefer saved value in checkpoint, then infer from head weight
            num_classes = (
                int(ckpt.get("num_classes", 0)) if isinstance(ckpt, dict) else 0
            )
            if num_classes == 0:
                for key in (
                    "classifier.1.weight",
                    "classifier.weight",
                    "linear.weight",
                ):
                    if key in state:
                        num_classes = state[key].shape[0]
                        break
                else:
                    num_classes = 3
            self._labels = _LABELS_2 if num_classes == 2 else _LABELS_3

            dropout = float(
                manifest.get("dropout")
                or manifest.get("model", {}).get("dropout")
                or 0.3
            )

            # Detect architecture and build the matching model class
            arch = _detect_arch(manifest, state)
            self._device = _pick_device()
            if arch == "pooler":
                model = self._build_pooler_model(backbone, num_classes, dropout)
            else:
                model = self._build_cls_model(backbone, num_classes, dropout)
            model.load_state_dict(state, strict=True)
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
