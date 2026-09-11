"""Agreement gate: evidence-credibility gate that optionally skips debate.

Models (PhoBERT/COOLANT) are no longer inputs to this gate — they are
evidence-grade signals consumed by the advocates and judge, not a gateway.
The gate scores only the retrieved evidence; model confidences are reported
in ``weight_breakdown`` for observability but do not affect the score.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..config import settings
from ..state import FactCheckState


def agreement_gate(state: FactCheckState) -> dict:
    """Compute evidence-credibility score. Skip debate if above threshold (AGREE-03)."""
    # Model results are reported for observability only — they never gate.
    ph_conf = 0.0
    co_conf = 0.0
    for result in state.get("model_results", []):
        if not result.get("available"):
            continue
        if result.get("model") == "phobert_vifactcheck":
            ph_conf = result.get("confidence", 0.0)
        elif result.get("model") == "coolant":
            co_conf = result.get("confidence", 0.0)

    # AGREE-02: Evidence credibility score
    evidence_real = state.get("evidence_real") or []
    evidence_fake = state.get("evidence_fake") or []

    tier_score = 0.0
    if evidence_real:
        trusted_count = len(
            [e for e in evidence_real if e.get("source_tier") == "trusted"]
        )
        tier_score = trusted_count / len(evidence_real)

    count_score = min(1.0, (len(evidence_real) + len(evidence_fake)) / 5)

    # D-07: consistency_score floor at 0.1
    consistency_score = max(0.1, state.get("consistency_score", 0.1))

    cred = 0.40 * tier_score + 0.30 * count_score + 0.30 * consistency_score

    agreement_score = cred

    # AGREE-03: Log skipped debates to logs/debates/<request_id>.jsonl
    if agreement_score >= settings.agreement_threshold:
        try:
            Path("logs/debates").mkdir(parents=True, exist_ok=True)
            request_id = state.get("request_id", "")
            log_entry = {
                "debate_skipped": True,
                "request_id": request_id,
                "agreement_score": agreement_score,
                "timestamp": datetime.utcnow().isoformat(),
            }
            log_path = Path("logs/debates") / f"{request_id}.jsonl"
            import json

            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass  # Silently fail logging per AGREE-03

    return {
        "agreement_score": round(agreement_score, 4),
        "weight_breakdown": {
            "phobert": ph_conf,
            "coolant": co_conf,
            "evidence": round(cred, 4),
        },
        "debate_exit_reason": (
            "skipped_high_agreement"
            if agreement_score >= settings.agreement_threshold
            else ""
        ),
    }


def route_after_agreement(state: FactCheckState) -> str:
    """Route after agreement gate.

    Debate is only meaningful when all three reference signals are present:
    - COOLANT says the claim is REAL with high confidence (>= threshold).
    - PhoBERT is available to cross-check the textual claim.
    - Evidence has been retrieved.

    If any of these is missing, or COOLANT is FAKE / low-confidence REAL, route
    straight to judge (verdict with evidence + PhoBERT).
    """
    model_results = state.get("model_results", [])
    coolant = None
    phobert_available = False
    for m in model_results:
        if m.get("model") == "coolant" and m.get("available"):
            coolant = m
        elif m.get("model") == "phobert_vifactcheck" and m.get("available"):
            phobert_available = True

    evidence_real = state.get("evidence_real") or []
    evidence_fake = state.get("evidence_fake") or []
    has_evidence = bool(evidence_real or evidence_fake)

    if coolant is not None:
        label = str(coolant.get("label", "")).upper()
        conf = coolant.get("confidence", 0.0)
        if (
            label == "REAL"
            and conf >= settings.coolant_debate_threshold
            and phobert_available
            and has_evidence
        ):
            return "debate"
        # COOLANT FAKE / low-confidence / missing PhoBERT or evidence
        return "judge"

    # No COOLANT signal — use the evidence-credibility gate
    agreement_score = state.get("agreement_score", 0.0)
    return "judge" if agreement_score >= settings.agreement_threshold else "debate"


# ── A2A service wrapper ─────────────────────────────────────────────────────
from ..a2a_server import AgentCardConfig, BaseTaskHandler, run_server


class AgreementGateHandler(BaseTaskHandler):
    """A2A TaskHandler exposing :func:`agreement_gate` over HTTP (port 9006)."""

    agent_card_config = AgentCardConfig(
        name="agreement_gate",
        description="Computes evidence-credibility score; decides whether to skip debate",
        version="1.0",
        skills=[
            {
                "id": "agreement",
                "name": "Agreement Scoring",
                "description": "Evidence-credibility scoring (models are not gate inputs)",
            }
        ],
        port=settings.a2a_port_agreement_gate,
    )

    async def agent_fn(self, state: FactCheckState) -> dict:
        return agreement_gate(state)


if __name__ == "__main__":
    run_server(AgreementGateHandler(), AgreementGateHandler.agent_card_config)
