import json
from unittest.mock import MagicMock, patch

import pytest

from factcheck_agents.agents.conclusion_agent import (
    _has_cross_source_conflict,
    _map_to_binary,
    conclusion_agent,
)
from factcheck_agents.graph_utils import EvidenceGraph
from factcheck_agents.prompts import CONCLUSION_SYSTEM_PROMPT


def _make_state(model_results=None, evidence=None, evidence_graph=None):
    return {
        "statement": "Tuyên bố mẫu",
        "model_results": model_results or [],
        "evidence": evidence or [],
        "evidence_graph": evidence_graph,
    }


def _make_model(model, label, confidence=0.9):
    return {
        "model": model,
        "available": True,
        "label": label,
        "label_id": 0,
        "probabilities": {},
        "confidence": confidence,
        "note": "",
    }


def _build_graph(*tiers):
    eg = EvidenceGraph()
    eg.graph.add_node("statement", node_type="statement")
    for i, tier in enumerate(tiers):
        node_id = f"https://example{i}.com/"
        eg.add_node(
            node_id,
            {
                "node_type": "evidence",
                "title": "T",
                "snippet": "S",
                "source_tier": tier,
            },
        )
        eg.add_edge("statement", node_id, type="mentions")
    return eg


class TestBinaryHelpers:
    def test_map_to_binary_supported_no_conflict(self):
        assert _map_to_binary("SUPPORTED", False) == ("REAL", "Thật")

    def test_map_to_binary_refuted_no_conflict(self):
        assert _map_to_binary("REFUTED", False) == ("FAKE", "Giả")

    def test_map_to_binary_true_maps_to_real(self):
        assert _map_to_binary("TRUE", False) == ("REAL", "Thật")

    def test_map_to_binary_unverified_maps_to_fake(self):
        assert _map_to_binary("UNVERIFIED", False) == ("FAKE", "Giả")

    def test_map_to_binary_conflict_overrides_true(self):
        assert _map_to_binary("TRUE", True) == ("FAKE", "Giả")

    def test_conflict_detection_no_graph(self):
        assert _has_cross_source_conflict(None) is False

    def test_conflict_detection_trusted_and_flagged(self):
        eg = _build_graph("trusted", "flagged")
        assert _has_cross_source_conflict(eg) is True

    def test_conflict_detection_trusted_and_social(self):
        eg = _build_graph("trusted", "social")
        assert _has_cross_source_conflict(eg) is True

    def test_conflict_detection_trusted_and_unknown(self):
        eg = _build_graph("trusted", "unknown")
        assert _has_cross_source_conflict(eg) is False

    def test_conflict_detection_only_trusted(self):
        eg = _build_graph("trusted", "trusted")
        assert _has_cross_source_conflict(eg) is False


class TestFallbackVerdict:
    @patch("factcheck_agents.agents.conclusion_agent.get_llm", return_value=None)
    def test_fallback_no_models_returns_unverified_and_fake(self, _mock_llm):
        result = conclusion_agent(_make_state())
        verdict = result["verdict"]
        assert verdict["label"] == "UNVERIFIED"
        assert verdict["verdict_binary"] == "FAKE"
        assert verdict["verdict_label_vi"] == "Giả"

    @patch("factcheck_agents.agents.conclusion_agent.get_llm", return_value=None)
    def test_fallback_supported_maps_to_real(self, _mock_llm):
        state = _make_state(model_results=[_make_model("phobert_vifactcheck", "SUPPORTED")])
        result = conclusion_agent(state)
        verdict = result["verdict"]
        assert verdict["label"] == "TRUE"
        assert verdict["verdict_binary"] == "REAL"
        assert verdict["verdict_label_vi"] == "Thật"

    @patch("factcheck_agents.agents.conclusion_agent.get_llm", return_value=None)
    def test_fallback_refuted_maps_to_fake(self, _mock_llm):
        state = _make_state(model_results=[_make_model("phobert_vifactcheck", "REFUTED")])
        result = conclusion_agent(state)
        verdict = result["verdict"]
        assert verdict["label"] == "FALSE"
        assert verdict["verdict_binary"] == "FAKE"
        assert verdict["verdict_label_vi"] == "Giả"

    @patch("factcheck_agents.agents.conclusion_agent.get_llm", return_value=None)
    def test_fallback_conflict_overrides_supported_to_fake(self, _mock_llm):
        eg = _build_graph("trusted", "flagged")
        state = _make_state(
            model_results=[_make_model("phobert_vifactcheck", "SUPPORTED")],
            evidence_graph=eg,
        )
        result = conclusion_agent(state)
        verdict = result["verdict"]
        assert verdict["verdict_binary"] == "FAKE"
        assert verdict["verdict_label_vi"] == "Giả"


class TestLlmVerdict:
    @patch("factcheck_agents.agents.conclusion_agent.get_llm")
    def test_llm_true_no_conflict_maps_to_real(self, mock_llm):
        mock_llm.return_value = MagicMock(
            invoke=lambda _msgs: MagicMock(
                content=json.dumps(
                    {
                        "label": "TRUE",
                        "confidence": 0.95,
                        "rationale": "Có bằng chứng đáng tin cậy.",
                        "citations": ["https://vnexpress.net/a"],
                        "recommendation": "Tin tức này là thật.",
                    }
                )
            )
        )
        result = conclusion_agent(_make_state())
        verdict = result["verdict"]
        assert verdict["label"] == "TRUE"
        assert verdict["verdict_binary"] == "REAL"
        assert verdict["verdict_label_vi"] == "Thật"

    @patch("factcheck_agents.agents.conclusion_agent.get_llm")
    def test_llm_true_with_conflict_maps_to_fake(self, mock_llm):
        mock_llm.return_value = MagicMock(
            invoke=lambda _msgs: MagicMock(
                content=json.dumps(
                    {
                        "label": "TRUE",
                        "confidence": 0.95,
                        "rationale": "Có bằng chứng xung khắc.",
                        "citations": ["https://vnexpress.net/a"],
                        "recommendation": "Cần xem xét thêm.",
                    }
                )
            )
        )
        eg = _build_graph("trusted", "social")
        result = conclusion_agent(_make_state(evidence_graph=eg))
        verdict = result["verdict"]
        assert verdict["label"] == "TRUE"
        assert verdict["verdict_binary"] == "FAKE"
        assert verdict["verdict_label_vi"] == "Giả"

    @patch("factcheck_agents.agents.conclusion_agent.get_llm")
    def test_llm_unverified_maps_to_fake(self, mock_llm):
        mock_llm.return_value = MagicMock(
            invoke=lambda _msgs: MagicMock(
                content=json.dumps(
                    {
                        "label": "UNVERIFIED",
                        "confidence": 0.4,
                        "rationale": "Không đủ bằng chứng.",
                        "citations": [],
                        "recommendation": "Chưa thể kết luận.",
                    }
                )
            )
        )
        result = conclusion_agent(_make_state())
        verdict = result["verdict"]
        assert verdict["label"] == "UNVERIFIED"
        assert verdict["verdict_binary"] == "FAKE"
        assert verdict["verdict_label_vi"] == "Giả"


class TestPrompt:
    def test_prompt_requests_vietnamese_and_binary(self):
        assert "in Vietnamese" in CONCLUSION_SYSTEM_PROMPT
        assert "Thật" in CONCLUSION_SYSTEM_PROMPT
        assert "Giả" in CONCLUSION_SYSTEM_PROMPT
        assert "4-class" in CONCLUSION_SYSTEM_PROMPT or "original 4-class label" in CONCLUSION_SYSTEM_PROMPT
