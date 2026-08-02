"""Unit tests for agreement_gate."""
from __future__ import annotations

import pytest

from factcheck_agents.agents.agreement_gate import agreement_gate, route_after_agreement


def _make_state(ph_conf, co_conf, evidence_real, consistency_score, ph_available=True, co_available=True, ph_label=None, co_label=None):
    """Helper to create test state for agreement_gate."""
    model_results = []
    if ph_available:
        model_results.append({
            "model": "phobert_vifactcheck",
            "available": True,
            "confidence": ph_conf,
            "label": ph_label or "SUPPORTED",
        })
    else:
        model_results.append({
            "model": "phobert_vifactcheck",
            "available": False,
        })

    if co_available:
        model_results.append({
            "model": "coolant",
            "available": True,
            "confidence": co_conf,
            "label": co_label or "SUPPORTED",
        })
    else:
        model_results.append({
            "model": "coolant",
            "available": False,
        })

    return {
        "model_results": model_results,
        "evidence_real": evidence_real,
        "evidence_fake": [],
        "consistency_score": consistency_score,
        "request_id": "test-123",
    }


def test_agreement_formula_weighted():
    """AGREE-01: Test weighted formula with both models available."""
    evidence_real = [
        {"source_tier": "trusted"},
        {"source_tier": "trusted"},
        {"source_tier": "unknown"},
    ]
    state = _make_state(ph_conf=0.8, co_conf=0.8, evidence_real=evidence_real, consistency_score=0.7)

    result = agreement_gate(state)

    # With high confidences and decent evidence, agreement should be moderately high
    assert 0.6 < result["agreement_score"] < 0.9
    assert result["weight_breakdown"]["phobert"] == 0.8
    assert result["weight_breakdown"]["coolant"] == 0.8
    assert "evidence" in result["weight_breakdown"]


def test_agreement_nei_forces_zero():
    """AGREE-01: Test that NEI label forces agreement_score to 0.0."""
    evidence_real = [{"source_tier": "trusted"}]
    state = _make_state(
        ph_conf=0.99,
        co_conf=0.99,
        evidence_real=evidence_real,
        consistency_score=0.9,
        ph_label="NEI",
    )

    result = agreement_gate(state)

    assert result["agreement_score"] == 0.0


def test_agreement_credibility_floor():
    """AGREE-02: Test that consistency_score has floor at 0.1."""
    evidence_real = [{"source_tier": "trusted"}]
    # State without consistency_score key should use floor of 0.1
    state = _make_state(
        ph_conf=0.5,
        co_conf=0.5,
        evidence_real=evidence_real,
        consistency_score=0.1,
    )

    result = agreement_gate(state)

    # Evidence credibility should be at least 0.04 (floor from 0.1 consistency)
    assert result["weight_breakdown"]["evidence"] >= 0.04


def test_agreement_unavailable_model_treated_as_zero():
    """AGREE-01: Test that unavailable model is treated as zero confidence."""
    evidence_real = [{"source_tier": "trusted"}]
    state = _make_state(
        ph_conf=0.0,
        co_conf=0.8,
        evidence_real=evidence_real,
        consistency_score=0.7,
        ph_available=False,
    )

    result = agreement_gate(state)

    # Should still have positive agreement from coolant + evidence
    assert result["agreement_score"] > 0
    assert result["weight_breakdown"]["phobert"] == 0.0
    assert result["weight_breakdown"]["coolant"] == 0.8


def test_route_skips_debate_above_threshold():
    """AGREE-03: Test that route_after_agreement returns 'judge' above threshold."""
    state = {"agreement_score": 0.9}
    result = route_after_agreement(state)
    assert result == "judge"


def test_route_to_debate_below_threshold():
    """AGREE-03: Test that route_after_agreement returns 'debate' below threshold."""
    state = {"agreement_score": 0.3}
    result = route_after_agreement(state)
    assert result == "debate"