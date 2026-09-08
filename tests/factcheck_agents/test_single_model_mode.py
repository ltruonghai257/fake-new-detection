"""Tests for single-model mode: debate convergence must not be treated as
independent agreement when only one model produced a result."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from factcheck_agents.agents.debate_utils import _format_model_results_verdict
from factcheck_agents.agents.expert_agent import expert_agent


def _model(name="coolant", label="REAL", confidence=0.9, available=True):
    return {
        "model": name,
        "available": available,
        "label": label,
        "confidence": confidence,
        "probabilities": {"REAL": confidence, "FAKE": 1 - confidence},
        "note": "",
    }


class TestSingleModelWarning:
    def test_single_model_adds_weak_signal_warning(self):
        out = _format_model_results_verdict(
            [_model("coolant"), _model("phobert_vifactcheck", available=False)]
        )
        assert "TÍN HIỆU YẾU" in out
        assert "MỘT model" in out

    def test_two_models_no_warning(self):
        out = _format_model_results_verdict(
            [_model("coolant"), _model("phobert_vifactcheck")]
        )
        assert "TÍN HIỆU YẾU" not in out

    def test_no_models_uses_no_available_header(self):
        out = _format_model_results_verdict(
            [_model("coolant", available=False)]
        )
        assert "KHÔNG CÓ KẾT QUẢ MODEL KHẢ DỤNG" in out


def _expert_state(model_results, converged=True, agreed="REAL"):
    return {
        "statement": "Tuyên bố mẫu",
        "model_results": model_results,
        "evidence_real": [],
        "evidence_fake": [],
        "debate_turns": [{"agent": "real_advocate", "round": 0, "verdict": agreed}],
        "debate_converged": converged,
        "debate_agreed_verdict": agreed,
        "weight_breakdown": {"argument_scores": []},
        "verdict": {},
        "request_id": "test-single-model",
    }


def _fake_llm(label="TRUE", confidence=0.9):
    resp = MagicMock()
    resp.content = json.dumps(
        {"label": label, "confidence": confidence, "rationale": "x" * 220}
    )
    llm = MagicMock()
    llm.invoke.return_value = resp
    return llm


class TestExpertConvergenceCap:
    @patch("factcheck_agents.agents.expert_agent.get_llm")
    def test_single_model_convergence_caps_confidence(self, mock_get_llm):
        """Converged debate with only one model → confidence capped at 0.7."""
        mock_get_llm.return_value = _fake_llm(label="TRUE", confidence=0.9)
        state = _expert_state([_model("coolant")])

        result = expert_agent(state)

        assert result["verdict"]["verdict_binary"] == "REAL"
        assert result["verdict"]["confidence"] <= 0.7

    @patch("factcheck_agents.agents.expert_agent.get_llm")
    def test_two_model_convergence_boosts_confidence(self, mock_get_llm):
        """Converged debate with ≥2 models → confidence boosted to ≥0.85."""
        mock_get_llm.return_value = _fake_llm(label="TRUE", confidence=0.6)
        state = _expert_state([_model("coolant"), _model("phobert_vifactcheck")])

        result = expert_agent(state)

        assert result["verdict"]["verdict_binary"] == "REAL"
        assert result["verdict"]["confidence"] >= 0.85
