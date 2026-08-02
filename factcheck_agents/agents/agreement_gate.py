"""Agreement gate: compute weighted agreement score, optionally skip debate."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..config import settings
from ..state import FactCheckState


def agreement_gate(state: FactCheckState) -> dict:
    """Compute weighted agreement score (AGREE-01/02). Skip debate if above threshold (AGREE-03)."""
    model_results = state.get("model_results", [])

    # Extract model confidences
    ph_conf = 0.0
    co_conf = 0.0
    ph_available = False
    co_available = False

    # AGREE-01: NEI forces agreement_score = 0.0 immediately
    for result in model_results:
        if result.get("available") and result.get("label") == "NEI":
            # Force zero agreement if any available model returns NEI
            return {
                "agreement_score": 0.0,
                "weight_breakdown": {"phobert": 0.0, "coolant": 0.0, "evidence": 0.0},
                "debate_exit_reason": "",
            }

        if result.get("model") == "phobert_vifactcheck":
            ph_available = result.get("available", False)
            if ph_available:
                ph_conf = result.get("confidence", 0.0)

        if result.get("model") == "coolant":
            co_available = result.get("available", False)
            if co_available:
                co_conf = result.get("confidence", 0.0)

    # AGREE-02: Evidence credibility score
    evidence_real = state.get("evidence_real") or []
    evidence_fake = state.get("evidence_fake") or []

    tier_score = 0.0
    if evidence_real:
        trusted_count = len([e for e in evidence_real if e.get("source_tier") == "trusted"])
        tier_score = trusted_count / len(evidence_real)

    count_score = min(1.0, (len(evidence_real) + len(evidence_fake)) / 5)

    # D-07: consistency_score floor at 0.1
    consistency_score = max(0.1, state.get("consistency_score", 0.1))

    cred = 0.40 * tier_score + 0.30 * count_score + 0.30 * consistency_score

    # AGREE-01: Weighted agreement score, normalized over available signals
    w_ph = 0.30 if ph_available else 0.0
    w_co = 0.30 if co_available else 0.0
    w_ev = 0.40  # Always included

    total_weight = w_ph + w_co + w_ev
    if total_weight == 0:
        agreement_score = 0.0
    else:
        agreement_score = (w_ph * ph_conf + w_co * co_conf + w_ev * cred) / total_weight

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
        "debate_exit_reason": "skipped_high_agreement"
        if agreement_score >= settings.agreement_threshold
        else "",
    }


def route_after_agreement(state: FactCheckState) -> str:
    """Route after agreement gate: skip to judge if agreement >= threshold (AGREE-03)."""
    agreement_score = state.get("agreement_score", 0.0)
    if agreement_score >= settings.agreement_threshold:
        return "judge"
    return "debate"