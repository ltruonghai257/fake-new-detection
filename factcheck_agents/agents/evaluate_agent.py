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

    evidence_text = build_evidence_text(evidence, statement)
    evidence_count = len(evidence)

    results = [
        _phobert().predict(statement, evidence_text, evidence_count),
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


# ── A2A service wrapper ─────────────────────────────────────────────────────
from ..a2a_server import AgentCardConfig, BaseTaskHandler, run_server
from ..config import settings


class EvaluateAgentHandler(BaseTaskHandler):
    """A2A TaskHandler exposing :func:`evaluate_agent` over HTTP (port 9002)."""

    agent_card_config = AgentCardConfig(
        name="evaluate_agent",
        description="Runs PhoBERT and COOLANT models on the statement",
        version="1.0",
        skills=[
            {
                "id": "phobert",
                "name": "PhoBERT ViFactCheck",
                "description": "Text model scoring",
            },
            {
                "id": "coolant",
                "name": "COOLANT",
                "description": "Multimodal image+text model",
            },
        ],
        port=settings.a2a_port_evaluate,
    )

    async def agent_fn(self, state: FactCheckState) -> dict:
        return evaluate_agent(state)


if __name__ == "__main__":
    run_server(EvaluateAgentHandler(), EvaluateAgentHandler.agent_card_config)
