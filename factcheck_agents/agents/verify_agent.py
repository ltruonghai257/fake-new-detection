"""Verify agent: run the two trained models concurrently on the statement.

- PhoBERT ViFactCheck: text model scoring (statement + evidence text).
- COOLANT: multimodal model, only when an image is supplied.

Both models run inside a ThreadPoolExecutor so their inference overlaps where
checkpoints exist. A missing checkpoint or any inference error produces an
``unavailable`` ModelResult — the pipeline never crashes. ``reliability_signal``
is True only when at least one model is available, confident, and not NEI.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Optional

from ..config import settings
from ..models import CoolantChecker, PhoBERTChecker
from ..models.phobert_checker import build_evidence_text
from ..state import FactCheckState, ModelResult

_PHOBERT_WEIGHT = float(os.getenv("FACTCHECK_PHOBERT_WEIGHT", "0.4"))
_COOLANT_WEIGHT = float(os.getenv("FACTCHECK_COOLANT_WEIGHT", "0.6"))


@lru_cache(maxsize=1)
def _phobert() -> PhoBERTChecker:
    return PhoBERTChecker()


@lru_cache(maxsize=1)
def _coolant() -> CoolantChecker:
    return CoolantChecker()


def _run_phobert(
    statement: str, evidence_text: str, evidence_count: int = 0
) -> ModelResult:
    try:
        return _phobert().predict(statement, evidence_text, evidence_count)
    except Exception as exc:
        return ModelResult(model="phobert_vifactcheck", available=False, note=str(exc))


def _run_coolant(statement: str, image_path: Optional[str]) -> ModelResult:
    try:
        result = _coolant().predict(statement, image_path)
        # Log warning if COOLANT is unavailable
        if not result.get("available"):
            import warnings

            warnings.warn(
                f"COOLANT model unavailable: {result.get('note', 'unknown')}. "
                "Continuing with PhoBERT-only verification.",
                UserWarning,
                stacklevel=3,
            )
        return result
    except Exception as exc:
        import warnings

        warnings.warn(
            f"COOLANT inference error: {exc}. Continuing with PhoBERT-only verification.",
            UserWarning,
            stacklevel=3,
        )
        return ModelResult(model="coolant", available=False, note=str(exc))


def _compute_reliability_signal(results: List[ModelResult]) -> bool:
    """Compute reliability_signal from model results (D-01 through D-05).

    Formula: weighted_score = sum(w_i * conf_i) / sum(w_i) for available models.
    NEI label from any available model -> False (D-03).
    All unavailable -> False (D-04).
    """
    weights = {"phobert_vifactcheck": _PHOBERT_WEIGHT, "coolant": _COOLANT_WEIGHT}

    available = [r for r in results if r.get("available")]
    if not available:
        return False

    for r in available:
        if r.get("label") == "NEI":
            return False

    total_weight = sum(weights.get(r["model"], 0.0) for r in available)
    if total_weight == 0.0:
        return False

    weighted_score = (
        sum(weights.get(r["model"], 0.0) * r.get("confidence", 0.0) for r in available)
        / total_weight
    )

    return weighted_score >= settings.reliability_threshold


def verify_agent(state: FactCheckState) -> dict:
    statement = state["statement"]
    evidence = state.get("evidence", []) or []
    image_path = state.get("image_path")

    if not image_path and evidence:
        for e in evidence:
            if e.get("image_path"):
                image_path = e["image_path"]
                break

    evidence_text = build_evidence_text(evidence, statement)
    evidence_count = len(evidence)

    # Pre-import transformers classes in main thread to avoid race condition
    # with lazy loading in ThreadPoolExecutor (transformers 5.x _LazyModule is
    # not thread-safe — concurrent imports from 2 threads cause BertTokenizer
    # to be missing from the transformers namespace).
    if image_path:
        try:
            from transformers import BertTokenizer, BertModel  # noqa: F401
        except ImportError:
            pass  # COOLANT will handle the error gracefully

    use_phobert = state.get("use_phobert", True)
    use_coolant = state.get("use_coolant", True)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_phobert = (
            executor.submit(_run_phobert, statement, evidence_text, evidence_count)
            if use_phobert
            else None
        )
        future_coolant = (
            executor.submit(_run_coolant, statement, image_path)
            if use_coolant
            else None
        )

    phobert_result = (
        future_phobert.result()
        if future_phobert
        else ModelResult(model="phobert_vifactcheck", available=False, note="disabled by ablation toggle")
    )
    coolant_result = (
        future_coolant.result()
        if future_coolant
        else ModelResult(model="coolant", available=False, note="disabled by ablation toggle")
    )

    results = [phobert_result, coolant_result]
    signal = _compute_reliability_signal(results)

    summary = ", ".join(
        f"{r['model']}={r.get('label', 'n/a')}"
        + (f"({r.get('confidence'):.2f})" if r.get("available") else " [unavailable]")
        for r in results
    )
    return {
        "reliability_signal": signal,
        "model_results": results,
        "messages": [("assistant", f"[Verify] {summary} -> signal={signal}")],
    }
