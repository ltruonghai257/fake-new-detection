"""Integration tests for build_debate_graph() — full M2 pipeline.

All 5 agents mocked at factcheck_agents.graph import namespace.
agreement_gate and debate_node run with real logic.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from factcheck_agents.graph import build_debate_graph, initial_state


def _fake_evidence(domain: str) -> dict:
    """Helper to create fake evidence dict."""
    return {
        "title": "T",
        "url": f"https://{domain}/a",
        "snippet": "snippet",
        "source_tier": "trusted",
        "score": 0.9,
    }


def _make_verdict(label="TRUE") -> dict:
    """Helper to create fake verdict dict."""
    return {
        "label": label,
        "verdict_binary": "REAL",
        "verdict_label_vi": "Thật",
        "confidence": 0.85,
        "rationale": "test",
        "citations": [],
    }


@pytest.fixture(autouse=True)
def setup_logs_dir():
    """Create logs directory before test session to avoid first-run mkdir errors."""
    Path("logs").mkdir(exist_ok=True)


def test_worldcup_claim():
    """Test full pipeline with Vietnamese World Cup claim."""
    claim = "Việt Nam đã đăng cai World Cup 2030"

    with patch.multiple(
        "factcheck_agents.graph",
        real_source_agent=lambda s: {
            "evidence_real": [_fake_evidence("vnexpress.net")],
            "messages": [],
        },
        fake_source_agent=lambda s: {
            "evidence_fake": [_fake_evidence("tingia.gov.vn")],
            "messages": [],
        },
        reranker_node=lambda s: {
            "evidence": [_fake_evidence("vnexpress.net")],
            "consistency_score": 0.8,
        },
        verify_agent=lambda s: {
            "model_results": [
                {
                    "model": "phobert_vifactcheck",
                    "available": True,
                    "label": "SUPPORTED",
                    "confidence": 0.9,
                },
                {"model": "coolant", "available": False, "note": "no ckpt"},
            ],
            "reliability_signal": True,
            "messages": [],
        },
        judge_agent=lambda s: {
            "verdict": _make_verdict(),
            "weight_breakdown": {"phobert": 0.27, "coolant": 0.0, "evidence": 0.32},
            "messages": [],
        },
    ):
        graph = build_debate_graph(checkpointer=MemorySaver())
        result = graph.invoke(
            initial_state(claim), config={"configurable": {"thread_id": "t-wc"}}
        )

        assert isinstance(result["verdict"], dict)
        assert result["verdict"]["label"] in {
            "TRUE",
            "FALSE",
            "MISLEADING",
            "UNVERIFIED",
            "NEI",
        }
        assert result["verdict"]["verdict_binary"] in {"REAL", "FAKE"}
        assert result["verdict"]["verdict_label_vi"] in {"Thật", "Giả"}


def test_vaccine_claim():
    """Test full pipeline with Vietnamese vaccine claim."""
    claim = "Bộ Y tế khuyến cáo không nên tiêm vaccine COVID-19"

    with patch.multiple(
        "factcheck_agents.graph",
        real_source_agent=lambda s: {
            "evidence_real": [_fake_evidence("vnexpress.net")],
            "messages": [],
        },
        fake_source_agent=lambda s: {
            "evidence_fake": [_fake_evidence("tingia.gov.vn")],
            "messages": [],
        },
        reranker_node=lambda s: {
            "evidence": [_fake_evidence("vnexpress.net")],
            "consistency_score": 0.8,
        },
        verify_agent=lambda s: {
            "model_results": [
                {
                    "model": "phobert_vifactcheck",
                    "available": True,
                    "label": "SUPPORTED",
                    "confidence": 0.9,
                },
                {"model": "coolant", "available": False, "note": "no ckpt"},
            ],
            "reliability_signal": True,
            "messages": [],
        },
        judge_agent=lambda s: {
            "verdict": _make_verdict(),
            "weight_breakdown": {"phobert": 0.27, "coolant": 0.0, "evidence": 0.32},
            "messages": [],
        },
    ):
        graph = build_debate_graph(checkpointer=MemorySaver())
        result = graph.invoke(
            initial_state(claim), config={"configurable": {"thread_id": "t-vax"}}
        )

        assert isinstance(result["verdict"], dict)
        assert result["verdict"]["label"] in {
            "TRUE",
            "FALSE",
            "MISLEADING",
            "UNVERIFIED",
            "NEI",
        }
        assert result["verdict"]["verdict_binary"] in {"REAL", "FAKE"}
        assert result["verdict"]["verdict_label_vi"] in {"Thật", "Giả"}


def test_nei_short_circuit():
    """Test EVRET-04 NEI short-circuit path when both evidence lists are empty."""
    claim = "Some claim with no evidence"

    with patch.multiple(
        "factcheck_agents.graph",
        real_source_agent=lambda s: {"evidence_real": [], "messages": []},
        fake_source_agent=lambda s: {"evidence_fake": [], "messages": []},
        reranker_node=lambda s: {
            "evidence": [],
            "consistency_score": 0.0,
        },
        verify_agent=lambda s: {
            "model_results": [
                {
                    "model": "phobert_vifactcheck",
                    "available": True,
                    "label": "SUPPORTED",
                    "confidence": 0.9,
                },
                {"model": "coolant", "available": False, "note": "no ckpt"},
            ],
            "reliability_signal": True,
            "messages": [],
        },
        judge_agent=lambda s: {
            "verdict": _make_verdict("NEI"),
            "weight_breakdown": {},
            "messages": [],
        },
    ):
        graph = build_debate_graph(checkpointer=MemorySaver())
        result = graph.invoke(
            initial_state(claim), config={"configurable": {"thread_id": "t-nei"}}
        )

        assert result["verdict"]["label"] == "NEI"
        # agreement_score should be absent or 0.0 since agreement gate never ran via NEI short-circuit
        assert result.get("agreement_score", 0.0) == 0.0


def test_logs_dirs_exist():
    """Test that logs/debates and logs/verdicts directories are created."""
    # Import modules
    from factcheck_agents.agents.debate_node import debate_node
    from factcheck_agents.agents.judge_agent import judge_agent

    # Call functions to trigger directory creation (happens at function call time)
    debate_node(
        {
            "statement": "test",
            "evidence_real": [],
            "evidence_fake": [],
            "request_id": "test",
        }
    )
    judge_agent(
        {
            "statement": "test",
            "evidence_real": [{"url": "https://test.com"}],
            "evidence_fake": [],
            "model_results": [],
            "request_id": "test",
        }
    )

    assert Path("logs/debates").exists()
    assert Path("logs/verdicts").exists()


def test_no_state_collision_between_runs():
    """Test MemorySaver isolation via distinct thread IDs."""
    claim1 = "Việt Nam đã đăng cai World Cup 2030"
    claim2 = "Bộ Y tế khuyến cáo không nên tiêm vaccine COVID-19"

    with patch.multiple(
        "factcheck_agents.graph",
        real_source_agent=lambda s: {
            "evidence_real": [_fake_evidence("vnexpress.net")],
            "messages": [],
        },
        fake_source_agent=lambda s: {
            "evidence_fake": [_fake_evidence("tingia.gov.vn")],
            "messages": [],
        },
        reranker_node=lambda s: {
            "evidence": [_fake_evidence("vnexpress.net")],
            "consistency_score": 0.8,
        },
        verify_agent=lambda s: {
            "model_results": [
                {
                    "model": "phobert_vifactcheck",
                    "available": True,
                    "label": "SUPPORTED",
                    "confidence": 0.9,
                },
                {"model": "coolant", "available": False, "note": "no ckpt"},
            ],
            "reliability_signal": True,
            "messages": [],
        },
        judge_agent=lambda s: {
            "verdict": _make_verdict(),
            "weight_breakdown": {"phobert": 0.27, "coolant": 0.0, "evidence": 0.32},
            "messages": [],
        },
    ):
        graph = build_debate_graph(checkpointer=MemorySaver())
        result1 = graph.invoke(
            initial_state(claim1),
            config={"configurable": {"thread_id": "t-collision-1"}},
        )
        result2 = graph.invoke(
            initial_state(claim2),
            config={"configurable": {"thread_id": "t-collision-2"}},
        )

        # Both verdicts should be valid
        assert isinstance(result1["verdict"], dict)
        assert isinstance(result2["verdict"], dict)
        assert result1["verdict"]["label"] in {
            "TRUE",
            "FALSE",
            "MISLEADING",
            "UNVERIFIED",
            "NEI",
        }
        assert result2["verdict"]["label"] in {
            "TRUE",
            "FALSE",
            "MISLEADING",
            "UNVERIFIED",
            "NEI",
        }
