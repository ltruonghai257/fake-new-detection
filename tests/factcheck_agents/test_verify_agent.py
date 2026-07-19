from unittest.mock import MagicMock, patch

import pytest

from factcheck_agents.agents.verify_agent import _compute_reliability_signal
from factcheck_agents.state import ModelResult


# ── _compute_reliability_signal unit tests (TEST-03) ───────────────────────


def _phobert_result(available: bool, label: str = "SUPPORTED", confidence: float = 0.9) -> ModelResult:
    if not available:
        return ModelResult(model="phobert_vifactcheck", available=False, note="unavailable")
    return ModelResult(
        model="phobert_vifactcheck", available=True, label=label, confidence=confidence
    )


def _coolant_result(available: bool, label: str = "REAL", confidence: float = 0.9) -> ModelResult:
    if not available:
        return ModelResult(model="coolant", available=False, note="unavailable")
    return ModelResult(
        model="coolant", available=True, label=label, confidence=confidence
    )


def test_signal_both_available_above_threshold():
    results = [_phobert_result(True, confidence=0.8), _coolant_result(True, confidence=0.8)]
    assert _compute_reliability_signal(results) is True


def test_signal_both_available_below_threshold():
    results = [_phobert_result(True, confidence=0.1), _coolant_result(True, confidence=0.1)]
    assert _compute_reliability_signal(results) is False


def test_signal_phobert_only_above_threshold():
    # COOLANT unavailable -> full weight on PhoBERT -> phobert_conf >= threshold
    results = [_phobert_result(True, confidence=0.8), _coolant_result(False)]
    assert _compute_reliability_signal(results) is True


def test_signal_coolant_only_above_threshold():
    # PhoBERT unavailable -> full weight on COOLANT
    results = [_phobert_result(False), _coolant_result(True, confidence=0.8)]
    assert _compute_reliability_signal(results) is True


def test_signal_nei_label_always_false():
    # High confidence NEI still -> False
    results = [_phobert_result(True, label="NEI", confidence=0.99), _coolant_result(False)]
    assert _compute_reliability_signal(results) is False


def test_signal_both_unavailable_returns_false():
    results = [_phobert_result(False), _coolant_result(False)]
    assert _compute_reliability_signal(results) is False


# ── verify_agent integration tests ─────────────────────────────────────────


def _make_state(statement="test statement", image_path=None, evidence=None):
    state = {"statement": statement}
    if image_path is not None:
        state["image_path"] = image_path
    if evidence is not None:
        state["evidence"] = evidence
    return state


@patch("factcheck_agents.agents.verify_agent._run_coolant")
@patch("factcheck_agents.agents.verify_agent._run_phobert")
def test_verify_agent_runs_both_models(mock_phobert, mock_coolant):
    mock_phobert.return_value = _phobert_result(True, confidence=0.9)
    mock_coolant.return_value = _coolant_result(False)
    from factcheck_agents.agents.verify_agent import verify_agent
    result = verify_agent(_make_state())
    mock_phobert.assert_called_once()
    mock_coolant.assert_called_once()
    assert "model_results" in result
    assert len(result["model_results"]) == 2


@patch("factcheck_agents.agents.verify_agent._run_coolant")
@patch("factcheck_agents.agents.verify_agent._run_phobert")
def test_verify_agent_no_checkpoints_no_crash(mock_phobert, mock_coolant):
    mock_phobert.return_value = _phobert_result(False)
    mock_coolant.return_value = _coolant_result(False)
    from factcheck_agents.agents.verify_agent import verify_agent
    result = verify_agent(_make_state())
    assert result["reliability_signal"] is False
    assert "model_results" in result


@patch("factcheck_agents.agents.verify_agent._run_coolant")
@patch("factcheck_agents.agents.verify_agent._run_phobert")
def test_verify_agent_returns_reliability_signal_key(mock_phobert, mock_coolant):
    mock_phobert.return_value = _phobert_result(True, confidence=0.9)
    mock_coolant.return_value = _coolant_result(False)
    from factcheck_agents.agents.verify_agent import verify_agent
    result = verify_agent(_make_state())
    assert "reliability_signal" in result
    assert isinstance(result["reliability_signal"], bool)


@patch("factcheck_agents.agents.verify_agent._run_coolant")
@patch("factcheck_agents.agents.verify_agent._run_phobert")
def test_verify_agent_image_fallback_from_evidence(mock_phobert, mock_coolant):
    """If state has no image_path, verify_agent picks the first evidence image."""
    evidence = [{"url": "http://x.com", "snippet": "s", "image_path": "/tmp/img.jpg"}]
    mock_phobert.return_value = _phobert_result(False)
    mock_coolant.return_value = _coolant_result(False)
    from factcheck_agents.agents.verify_agent import verify_agent
    verify_agent(_make_state(evidence=evidence))
    # _run_coolant must have been called with the evidence image path
    call_args = mock_coolant.call_args
    assert call_args[0][1] == "/tmp/img.jpg"


# ── build_evidence_text tier ordering tests (EVGRAPH-03) ───────────────────


def test_build_evidence_text_trusted_before_unknown():
    from factcheck_agents.models.phobert_checker import build_evidence_text

    evidence = [
        {"snippet": "unknown snippet", "source_tier": "unknown"},
        {"snippet": "trusted snippet", "source_tier": "trusted"},
    ]
    result = build_evidence_text(evidence)
    assert result.index("trusted snippet") < result.index("unknown snippet")


def test_build_evidence_text_missing_tier_treated_as_unknown():
    from factcheck_agents.models.phobert_checker import build_evidence_text

    evidence = [
        {"snippet": "no tier snippet"},
        {"snippet": "trusted snippet", "source_tier": "trusted"},
    ]
    result = build_evidence_text(evidence)
    assert result.index("trusted snippet") < result.index("no tier snippet")
