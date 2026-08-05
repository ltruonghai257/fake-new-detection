"""COOLANT multimodal detector wrapper (optional, image required).

COOLANT is a text+image model, so it only runs when the statement is
accompanied by an image. It reuses the project's model + preprocessing code
under ``src/`` and follows the frozen-checkpoint loading pattern from
``notebooks/pipeline/04_mm_vifactcheck_integration.ipynb``.

Everything is best-effort: if there is no image, no checkpoint, or a dimension
mismatch (expected while the model is still being validated), it returns an
``unavailable`` ModelResult instead of raising.
"""

from __future__ import annotations

import json as _json
import sys
from pathlib import Path
from typing import Optional

import warnings

from ..config import settings
from ..state import ModelResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_LABELS = {0: "REAL", 1: "FAKE"}


def _read_manifest_metric(run_dir: Path) -> float:
    """Extract the best validation metric from a run's checkpoint_manifest.json."""
    man = run_dir / "checkpoint_manifest.json"
    if not man.exists():
        return -1.0
    try:
        data = _json.loads(man.read_text())
        metrics = data.get("best_metrics", {})
        for key in (
            "val_accuracy",
            "best_val_accuracy",
            "best_val_macro_f1",
            "val_macro_f1",
        ):
            val = metrics.get(key)
            if val is not None:
                return float(val)
        return -1.0
    except Exception:
        return -1.0


def _resolve_ckpt() -> Optional[Path]:
    """Explicit override wins; else pick the run with the best validation metric.

    Reads ``checkpoint_manifest.json`` from each run directory and selects the
    one with the highest ``val_accuracy``. Falls back to newest-by-mtime if no
    manifests are found.
    """
    if settings.coolant_ckpt_path:
        p = Path(settings.coolant_ckpt_path)
        return p if p.exists() else None
    root = settings.coolant_search_root()
    if not root.exists():
        return None
    runs = [d for d in root.iterdir() if d.is_dir() and (d / "best_model.pth").exists()]
    if not runs:
        return None

    scored = [(_read_manifest_metric(d), d) for d in runs]
    best_metric = max(s for s, _ in scored)
    if best_metric > 0:
        best_dir = max(scored, key=lambda pair: pair[0])[1]
        return best_dir / "best_model.pth"

    newest = max(runs, key=lambda d: d.stat().st_mtime)
    return newest / "best_model.pth"


class CoolantChecker:
    def __init__(self) -> None:
        self._loaded = False
        self._model = None
        self._preprocessor = None
        self._device = "cpu"
        self._image_model = "resnet50"
        self._load_error: Optional[str] = None

    def load(self) -> bool:
        if self._loaded:
            return True
        if self._load_error is not None:
            return False
        try:
            import torch

            from src.models.resnet_coolant import PatchedCOOLANT

            ckpt_path = _resolve_ckpt()
            if ckpt_path is None:
                self._load_error = (
                    "No COOLANT checkpoint found. Set COOLANT_CKPT_PATH or place a run under "
                    f"{settings.coolant_search_root()}."
                )
                return False

            if torch.cuda.is_available():
                self._device = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                self._device = "mps"
            else:
                self._device = "cpu"
            ckpt = torch.load(ckpt_path, map_location=self._device)
            model_cfg = ckpt["config"]["model"]
            self._image_model = (
                ckpt["config"].get("data", {}).get("image_model", self._image_model)
            )

            model = PatchedCOOLANT.from_config(model_cfg, device=self._device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            model.eval()
            self._model = model
            self._loaded = True
            return True
        except Exception as exc:  # pragma: no cover - defensive
            self._load_error = f"COOLANT load failed: {exc}"
            return False

    def _ensure_preprocessor(self):
        if self._preprocessor is not None:
            return
        from src.preprocessing.combined_preprocessing import CombinedPreprocessor

        self._preprocessor = CombinedPreprocessor(
            text_model_name="vinai/phobert-base-v2",
            image_model_name=self._image_model,
            device=self._device,
        )

    def predict(self, statement: str, image_path: Optional[str]) -> ModelResult:
        if not image_path:
            warnings.warn(
                "COOLANT skipped: no image provided. Multimodal verification requires an image. "
                "System will continue with PhoBERT-only verification.",
                UserWarning,
                stacklevel=2,
            )
            return ModelResult(
                model="coolant",
                available=False,
                note="skipped: multimodal model requires an image alongside the statement",
            )
        if not Path(image_path).exists():
            warnings.warn(
                f"COOLANT skipped: image not found ({image_path}). "
                "System will continue with PhoBERT-only verification.",
                UserWarning,
                stacklevel=2,
            )
            return ModelResult(
                model="coolant", available=False, note=f"image not found: {image_path}"
            )
        if not self.load():
            warnings.warn(
                f"COOLANT model unavailable: {self._load_error or 'unknown error'}. "
                "System will continue with PhoBERT-only verification. "
                "For full multimodal verification, ensure COOLANT checkpoint is properly configured.",
                UserWarning,
                stacklevel=2,
            )
            return ModelResult(
                model="coolant", available=False, note=self._load_error or "unavailable"
            )
        try:
            import torch
            import torch.nn.functional as F

            self._ensure_preprocessor()
            text_feat, image_feat = self._preprocessor.preprocess_sample(
                statement, image_path
            )

            # Fix based on notebook inference pattern:
            # Text: [1, seq, 768] -> [1, 768, seq] (permute from [1, seq, 768])
            # Image: [2048] -> [1, 2048] (2D tensor, not 3D!)
            # Batch size workaround: model needs batch >= 2, so duplicate samples

            # Text preprocessing
            text_raw = torch.tensor(text_feat, dtype=torch.float32)
            if text_raw.dim() == 2:
                text_raw = text_raw.unsqueeze(0)  # [1, seq, 768]
            text_raw = text_raw.permute(0, 2, 1).to(self._device)  # [1, 768, seq]

            # Image preprocessing - keep as 2D [1, 2048]
            image_raw = torch.tensor(image_feat, dtype=torch.float32)
            if image_raw.dim() == 1:
                image_raw = image_raw.unsqueeze(0)  # [1, 2048]
            image_raw = image_raw.to(self._device)

            # Batch size workaround: duplicate samples (model needs batch >= 2)
            text_raw = text_raw.expand(2, -1, -1)  # [2, 768, seq]
            image_raw = image_raw.expand(2, -1)  # [2, 2048]

            with torch.no_grad():
                out = self._model(text_raw, image_raw)
                logits = out["detection_logits"] if isinstance(out, dict) else out
                # Take first sample since we duplicated for batch workaround
                logits = logits[0]  # [2] -> [2] after model, take first sample
                probs = F.softmax(logits, dim=-1).cpu().tolist()

            label_id = int(max(range(len(probs)), key=lambda i: probs[i]))
            prob_map = {
                _LABELS.get(i, str(i)): round(p, 4) for i, p in enumerate(probs)
            }

            # Build workflow steps for UI
            workflow_steps = [
                {
                    "step": "1. Preprocess text",
                    "description": "Encode statement with PhoBERT tokenizer to get embeddings",
                    "input": f"Statement ({len(statement)} chars)",
                    "output": f"text_feat shape: {text_feat.shape if hasattr(text_feat, 'shape') else 'list'}",
                },
                {
                    "step": "2. Preprocess image",
                    "description": f"Extract ResNet50 features from image ({self._image_model})",
                    "input": f"Image path: {image_path}",
                    "output": f"image_feat shape: {image_feat.shape if hasattr(image_feat, 'shape') else 'list'}",
                },
                {
                    "step": "3. Reshape tensors",
                    "description": "Text: [seq, 768] -> [1, 768, seq] | Image: [2048] -> [1, 2048]",
                    "input": f"text_raw: {text_raw.shape}, image_raw: {image_raw.shape}",
                    "output": f"batch_size workaround: duplicate to batch=2",
                },
                {
                    "step": "4. Multimodal fusion",
                    "description": "Pass through COOLANT model (text + image fusion)",
                    "input": f"text_raw [2, 768, seq], image_raw [2, 2048] (device: {self._device})",
                    "output": f"detection_logits: {logits.shape}",
                },
                {
                    "step": "5. Classify",
                    "description": "Apply softmax to logits to get probabilities",
                    "input": f"logits -> softmax",
                    "output": f"probabilities: {prob_map}",
                },
            ]

            return ModelResult(
                model="coolant",
                available=True,
                label=_LABELS.get(label_id, str(label_id)),
                label_id=label_id,
                probabilities=prob_map,
                confidence=round(probs[label_id], 4),
                note="multimodal (statement + image) prediction",
                workflow_steps=workflow_steps,
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"COOLANT inference error: {exc}. Continuing with PhoBERT-only verification.",
                UserWarning,
                stacklevel=3,
            )
            return ModelResult(
                model="coolant", available=False, note=f"inference error: {exc}"
            )
