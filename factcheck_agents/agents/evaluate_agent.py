"""Evaluate agent: run the two trained models on the statement.

- PhoBERT ViFactCheck: text model scoring (statement + retrieved evidence).
- COOLANT: multimodal model, only when an image is supplied.

Model wrappers are cached across calls and never raise: a missing checkpoint
becomes an ``unavailable`` result so the pipeline keeps flowing.
"""

from __future__ import annotations

from functools import lru_cache

from ..models import CoolantChecker, PhoBERTChecker
from ..models.phobert_checker import build_evidence_text
from ..state import FactCheckState


@lru_cache(maxsize=1)
def _phobert() -> PhoBERTChecker:
    return PhoBERTChecker()


@lru_cache(maxsize=1)
def _coolant() -> CoolantChecker:
    return CoolantChecker()


def evaluate_agent(state: FactCheckState) -> dict:
    statement = state["statement"]
    evidence = state.get("evidence", []) or []
    image_path = state.get("image_path")

    # If no explicit image was provided, use the first evidence page image saved by the search agent.
    if not image_path and evidence:
        for e in evidence:
            if e.get("image_path"):
                image_path = e["image_path"]
                break

    evidence_text = build_evidence_text(evidence)

    results = [
        _phobert().predict(statement, evidence_text),
        _coolant().predict(statement, image_path),
    ]

    summary = ", ".join(
        f"{r['model']}={r.get('label', 'n/a')}"
        + (f"({r.get('confidence'):.2f})" if r.get("available") else " [unavailable]")
        for r in results
    )
    return {
        "model_results": results,
        "messages": [("assistant", f"[Evaluate] {summary}")],
    }
