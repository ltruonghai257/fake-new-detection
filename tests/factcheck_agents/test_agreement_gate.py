"""Unit tests for agreement_gate (evidence-credibility gate).

Models are no longer gate inputs — agreement_score is derived purely from
retrieved evidence credibility. Model confidences are still reported in
``weight_breakdown`` for observability.
"""

from __future__ import annotations

from factcheck_agents.agents.agreement_gate import agreement_gate, route_after_agreement


def _make_state(
    evidence_real, evidence_fake=None, consistency_score=0.5, model_results=None
):
    return {
        "model_results": model_results or [],
        "evidence_real": evidence_real,
        "evidence_fake": evidence_fake or [],
        "consistency_score": consistency_score,
        "request_id": "test-123",
    }


def test_agreement_score_is_evidence_only():
    """High-trust, high-count, high-consistency evidence scores high."""
    evidence_real = [
        {"source_tier": "trusted"},
        {"source_tier": "trusted"},
        {"source_tier": "unknown"},
    ]
    evidence_fake = [{"source_tier": "flagged"}, {"source_tier": "unknown"}]
    state = _make_state(evidence_real, evidence_fake, consistency_score=0.7)

    result = agreement_gate(state)

    # cred = 0.4*(2/3) + 0.3*(5/5) + 0.3*0.7 = 0.2667 + 0.3 + 0.21 ≈ 0.7767
    assert 0.7 < result["agreement_score"] < 0.8
    assert "evidence" in result["weight_breakdown"]


def test_model_confidence_does_not_affect_score():
    """Model results are observability-only: identical evidence gives identical score."""
    evidence = [{"source_tier": "trusted"}]
    with_models = _make_state(
        evidence,
        consistency_score=0.7,
        model_results=[
            {
                "model": "phobert_vifactcheck",
                "available": True,
                "confidence": 0.99,
                "label": "FAKE",
            },
            {
                "model": "coolant",
                "available": True,
                "confidence": 0.01,
                "label": "REAL",
            },
        ],
    )
    without_models = _make_state(evidence, consistency_score=0.7)

    a = agreement_gate(with_models)
    b = agreement_gate(without_models)

    assert a["agreement_score"] == b["agreement_score"]
    # confidences still surfaced for observability
    assert a["weight_breakdown"]["phobert"] == 0.99
    assert a["weight_breakdown"]["coolant"] == 0.01
    assert b["weight_breakdown"]["phobert"] == 0.0


def test_nei_label_no_longer_forces_zero():
    """NEI from a model does not force the gate — evidence decides."""
    evidence = [{"source_tier": "trusted"}, {"source_tier": "trusted"}]
    state = _make_state(
        evidence,
        consistency_score=0.9,
        model_results=[
            {
                "model": "phobert_vifactcheck",
                "available": True,
                "confidence": 0.9,
                "label": "NEI",
            },
        ],
    )

    result = agreement_gate(state)

    assert result["agreement_score"] > 0


def test_empty_evidence_scores_near_zero():
    """No evidence → only the consistency floor contributes."""
    state = _make_state([], consistency_score=0.1)

    result = agreement_gate(state)

    # cred = 0.4*0 + 0.3*0 + 0.3*0.1 = 0.03
    assert result["agreement_score"] == 0.03
    assert result["weight_breakdown"]["evidence"] == 0.03


def test_consistency_floor_at_point_one():
    """D-07: missing/negative consistency_score floors at 0.1."""
    state = _make_state([{"source_tier": "trusted"}], consistency_score=-5.0)

    result = agreement_gate(state)

    # cred = 0.4*1.0 + 0.3*0.2 + 0.3*0.1 = 0.4 + 0.06 + 0.03 = 0.49
    assert abs(result["agreement_score"] - 0.49) < 1e-4


def test_route_skips_debate_above_threshold():
    """AGREE-03: route_after_agreement returns 'judge' above threshold."""
    state = {"agreement_score": 0.9}
    assert route_after_agreement(state) == "judge"


def test_route_to_debate_below_threshold():
    """AGREE-03: route_after_agreement returns 'debate' below threshold."""
    state = {"agreement_score": 0.3}
    assert route_after_agreement(state) == "debate"


def test_route_coolant_real_high_confidence_forces_debate():
    """COOLANT REAL + conf >= threshold + PhoBERT available + evidence → debate."""
    state = {
        "agreement_score": 0.9,
        "model_results": [
            {
                "model": "coolant",
                "available": True,
                "label": "REAL",
                "confidence": 0.85,
            },
            {
                "model": "phobert_vifactcheck",
                "available": True,
                "label": "FAKE",
                "confidence": 0.9,
            },
        ],
        "evidence_real": [{"source_tier": "trusted"}],
    }
    assert route_after_agreement(state) == "debate"


def test_route_coolant_real_low_confidence_skips_debate():
    """COOLANT REAL with confidence below threshold goes straight to judge."""
    state = {
        "agreement_score": 0.3,
        "model_results": [
            {"model": "coolant", "available": True, "label": "REAL", "confidence": 0.5},
            {
                "model": "phobert_vifactcheck",
                "available": True,
                "label": "FAKE",
                "confidence": 0.9,
            },
        ],
        "evidence_real": [{"source_tier": "trusted"}],
    }
    assert route_after_agreement(state) == "judge"


def test_route_coolant_fake_skips_debate():
    """COOLANT FAKE goes straight to judge (verdict with evidence + PhoBERT)."""
    state = {
        "agreement_score": 0.3,
        "model_results": [
            {
                "model": "coolant",
                "available": True,
                "label": "FAKE",
                "confidence": 0.95,
            },
            {
                "model": "phobert_vifactcheck",
                "available": True,
                "label": "REAL",
                "confidence": 0.9,
            },
        ],
        "evidence_real": [{"source_tier": "trusted"}],
    }
    assert route_after_agreement(state) == "judge"


def test_route_coolant_real_missing_phobert_skips_debate():
    """Debate requires PhoBERT to be available."""
    state = {
        "agreement_score": 0.3,
        "model_results": [
            {
                "model": "coolant",
                "available": True,
                "label": "REAL",
                "confidence": 0.95,
            },
            {"model": "phobert_vifactcheck", "available": False, "label": "N/A"},
        ],
        "evidence_real": [{"source_tier": "trusted"}],
    }
    assert route_after_agreement(state) == "judge"


def test_route_coolant_real_missing_evidence_skips_debate():
    """Debate requires retrieved evidence."""
    state = {
        "agreement_score": 0.3,
        "model_results": [
            {
                "model": "coolant",
                "available": True,
                "label": "REAL",
                "confidence": 0.95,
            },
            {
                "model": "phobert_vifactcheck",
                "available": True,
                "label": "FAKE",
                "confidence": 0.9,
            },
        ],
        "evidence_real": [],
        "evidence_fake": [],
    }
    assert route_after_agreement(state) == "judge"


def test_route_no_coolant_falls_back_to_agreement_score():
    """If COOLANT is unavailable, routing falls back to evidence-credibility score."""
    state = {"agreement_score": 0.9, "model_results": []}
    assert route_after_agreement(state) == "judge"

    state = {"agreement_score": 0.3, "model_results": []}
    assert route_after_agreement(state) == "debate"
