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

import sys
from pathlib import Path
from typing import Optional

from ..config import settings
from ..state import ModelResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_LABELS = {0: "REAL", 1: "FAKE"}


def _resolve_ckpt() -> Optional[Path]:
    if settings.coolant_ckpt_path:
        p = Path(settings.coolant_ckpt_path)
        return p if p.exists() else None
    root = settings.coolant_search_root()
    if not root.exists():
        return None
    runs = [d for d in root.iterdir() if d.is_dir() and (d / "best_model.pth").exists()]
    if not runs:
        return None
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

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(ckpt_path, map_location=self._device)
            model_cfg = ckpt["config"]["model"]
            self._image_model = ckpt["config"].get("data", {}).get("image_model", self._image_model)

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
            return ModelResult(
                model="coolant",
                available=False,
                note="skipped: multimodal model requires an image alongside the statement",
            )
        if not Path(image_path).exists():
            return ModelResult(model="coolant", available=False, note=f"image not found: {image_path}")
        if not self.load():
            return ModelResult(model="coolant", available=False, note=self._load_error or "unavailable")
        try:
            import torch
            import torch.nn.functional as F

            self._ensure_preprocessor()
            text_feat, image_feat = self._preprocessor.preprocess_sample(statement, image_path)

            # text: [1, seq, 768] -> [1, 768, seq] for the patched FastCNN
            text_raw = torch.tensor(text_feat, dtype=torch.float32)
            if text_raw.dim() == 2:
                text_raw = text_raw.unsqueeze(0)
            text_raw = text_raw.permute(0, 2, 1).to(self._device)

            image_raw = torch.tensor(image_feat, dtype=torch.float32).reshape(1, -1).to(self._device)

            with torch.no_grad():
                out = self._model(text_raw, image_raw)
                logits = out["detection_logits"] if isinstance(out, dict) else out
                probs = F.softmax(logits, dim=-1)[0].cpu().tolist()

            label_id = int(max(range(len(probs)), key=lambda i: probs[i]))
            prob_map = {_LABELS.get(i, str(i)): round(p, 4) for i, p in enumerate(probs)}
            return ModelResult(
                model="coolant",
                available=True,
                label=_LABELS.get(label_id, str(label_id)),
                label_id=label_id,
                probabilities=prob_map,
                confidence=round(probs[label_id], 4),
                note="multimodal (statement + image) prediction",
            )
        except Exception as exc:  # pragma: no cover - defensive
            return ModelResult(model="coolant", available=False, note=f"inference error: {exc}")
