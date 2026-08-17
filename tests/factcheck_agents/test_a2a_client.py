"""Persisted behavioral tests for the A2A client (requirement A2A-04).

Covers AgentUnavailableError, _timeout_for resolution, call_agent success and
error paths, the degrade_on_unavailable decorator (incl. shallow-copy
isolation), the search_agent EvidenceGraph rebuild (D-03), per-wrapper degrade
diffs, and the Settings timeout defaults. All HTTP is mocked — no servers.
"""

from unittest.mock import patch

import httpx
import pytest

from factcheck_agents import a2a_client
from factcheck_agents.a2a_client import (
    AgentUnavailableError,
    _timeout_for,
    agreement_gate,
    conclusion_agent,
    evaluate_agent,
    judge_agent,
    real_advocate,
    search_agent,
    social_loop_agent,
)
from factcheck_agents.config import Settings, a2a_ports, settings
from factcheck_agents.graph_utils import EvidenceGraph


# ── AgentUnavailableError ──────────────────────────────────────────────────


def test_agent_unavailable_error_attributes_and_message():
    err = AgentUnavailableError("search_agent", 9001, "conn refused")
    assert err.agent_name == "search_agent"
    assert err.port == 9001
    assert err.cause == "conn refused"
    assert str(err) == "[search_agent:9001] unavailable: conn refused"


# ── _timeout_for ───────────────────────────────────────────────────────────


def test_timeout_for_per_agent_defaults(monkeypatch):
    monkeypatch.setattr(settings, "a2a_client_timeout", None)
    monkeypatch.setattr(settings, "a2a_client_timeout_search", 120)
    monkeypatch.setattr(settings, "a2a_client_timeout_evaluate", 60)
    monkeypatch.setattr(settings, "a2a_client_timeout_social_loop", 30)
    monkeypatch.setattr(settings, "a2a_client_timeout_agreement_gate", 30)
    monkeypatch.setattr(settings, "a2a_client_timeout_judge", 120)
    assert _timeout_for("search_agent") == 120.0
    assert _timeout_for("evaluate_agent") == 60.0
    assert _timeout_for("social_loop_agent") == 30.0
    assert _timeout_for("agreement_gate") == 30.0
    assert _timeout_for("judge_agent") == 120.0


def test_timeout_for_global_override_wins(monkeypatch):
    monkeypatch.setattr(settings, "a2a_client_timeout", 7)
    monkeypatch.setattr(settings, "a2a_client_timeout_search", 120)
    monkeypatch.setattr(settings, "a2a_client_timeout_judge", 120)
    assert _timeout_for("search_agent") == 7.0
    assert _timeout_for("judge_agent") == 7.0


# ── call_agent ─────────────────────────────────────────────────────────────


def test_call_agent_success_path(monkeypatch):
    monkeypatch.setattr(settings, "a2a_client_timeout", None)
    monkeypatch.setattr(settings, "a2a_client_timeout_search", 120)
    port = a2a_ports()["search_agent"]
    inner = {"evidence": [{"url": "https://a.com"}], "search_queries": ["q"]}
    response = {
        "task": {
            "status": {"state": "TASK_STATE_COMPLETED"},
            "artifacts": [{"name": "output", "parts": [{"data": inner}]}],
        }
    }
    with patch("factcheck_agents.a2a_client.httpx.Client") as mock_client:
        mock_client.return_value.post.return_value.json.return_value = response
        result = a2a_client.call_agent("search_agent", {"statement": "s"})

    assert result == inner
    mock_client.assert_called_once_with(timeout=120.0)
    args, kwargs = mock_client.return_value.post.call_args
    assert args[0] == f"http://localhost:{port}/message:send"
    assert kwargs["headers"] == {"A2A-Version": "1.0"}
    body = kwargs["json"]["message"]
    assert body["role"] == "ROLE_USER"
    assert body["messageId"].startswith("msg-")
    assert isinstance(body["parts"][0]["data"], dict)


def test_call_agent_http_error_raises_unavailable():
    port = a2a_ports()["search_agent"]
    with patch("factcheck_agents.a2a_client.httpx.Client") as mock_client:
        mock_client.return_value.post.side_effect = httpx.ConnectError("conn refused")
        with pytest.raises(AgentUnavailableError) as excinfo:
            a2a_client.call_agent("search_agent", {"statement": "s"})
    assert excinfo.value.agent_name == "search_agent"
    assert excinfo.value.port == port
    assert "conn refused" in excinfo.value.cause


def test_call_agent_failed_task_raises_with_server_error():
    with patch("factcheck_agents.a2a_client.httpx.Client") as mock_client:
        mock_client.return_value.post.return_value.json.return_value = {
            "task": {
                "status": {"state": "TASK_STATE_FAILED"},
                "artifacts": [
                    {"name": "output", "parts": [{"data": {"error": "boom"}}]}
                ],
            }
        }
        with pytest.raises(AgentUnavailableError) as excinfo:
            a2a_client.call_agent("search_agent", {"statement": "s"})
    assert excinfo.value.cause == "boom"


def test_call_agent_failed_task_without_error_artifact_falls_back():
    with patch("factcheck_agents.a2a_client.httpx.Client") as mock_client:
        mock_client.return_value.post.return_value.json.return_value = {
            "task": {"status": {"state": "TASK_STATE_FAILED"}, "artifacts": []}
        }
        with pytest.raises(AgentUnavailableError) as excinfo:
            a2a_client.call_agent("search_agent", {"statement": "s"})
    assert excinfo.value.cause == "server error"


def test_call_agent_working_task_returns_empty_diff():
    with patch("factcheck_agents.a2a_client.httpx.Client") as mock_client:
        mock_client.return_value.post.return_value.json.return_value = {
            "task": {"status": {"state": "TASK_STATE_WORKING"}, "artifacts": []}
        }
        result = a2a_client.call_agent("search_agent", {"statement": "s"})
    assert result == {}


# ── degrade_on_unavailable ─────────────────────────────────────────────────


def test_degrade_on_unavailable_returns_degrade_diff_on_error():
    degrade_diff = {"evidence": [], "evidence_graph": None}

    @a2a_client.degrade_on_unavailable("search_agent", degrade_diff)
    def fn(state):
        raise AgentUnavailableError("search_agent", 9001, "conn refused")

    assert fn({"statement": "s"}) == degrade_diff


def test_degrade_on_unavailable_passes_through_success():
    degrade_diff = {"evidence": []}

    @a2a_client.degrade_on_unavailable("search_agent", degrade_diff)
    def fn(state):
        return {"evidence": [{"url": "https://a.com"}]}

    assert fn({"statement": "s"}) == {"evidence": [{"url": "https://a.com"}]}


def test_degrade_on_unavailable_returns_shallow_copy():
    degrade_diff = {"evidence": [], "search_queries": []}

    @a2a_client.degrade_on_unavailable("search_agent", degrade_diff)
    def fn(state):
        raise AgentUnavailableError("search_agent", 9001, "conn refused")

    result = fn({"statement": "s"})
    assert result is not degrade_diff
    result["new_key"] = True
    del result["search_queries"]
    result["evidence"] = ["replaced"]
    assert "new_key" not in degrade_diff
    assert "search_queries" in degrade_diff
    assert degrade_diff["evidence"] == []


# ── search_agent wrapper (D-03 EvidenceGraph rebuild) ──────────────────────


def test_search_agent_builds_evidence_graph():
    evidence = [
        {
            "url": "https://a.com",
            "title": "A",
            "snippet": "sa",
            "source_tier": "trusted",
        },
        {
            "url": "https://b.com",
            "title": "B",
            "snippet": "sb",
            "source_tier": "flagged",
        },
    ]
    with patch(
        "factcheck_agents.a2a_client.call_agent",
        return_value={"evidence": evidence, "search_queries": ["q"]},
    ):
        result = search_agent({"statement": "s"})
    assert isinstance(result["evidence_graph"], EvidenceGraph)
    # statement node + 2 evidence nodes
    assert result["evidence_graph"].graph.number_of_nodes() == 3


def test_search_agent_empty_evidence_sets_graph_none():
    with patch(
        "factcheck_agents.a2a_client.call_agent",
        return_value={"evidence": []},
    ):
        result = search_agent({"statement": "s"})
    assert result["evidence_graph"] is None


# ── per-wrapper degrade diffs ──────────────────────────────────────────────


def test_wrapper_degrade_diffs():
    with patch(
        "factcheck_agents.a2a_client.call_agent",
        side_effect=AgentUnavailableError("any", 9001, "boom"),
    ):
        conc = conclusion_agent({"statement": "s"})
        assert conc["verdict"]["label"] == "UNVERIFIED"
        assert conc["verdict"]["confidence"] == 0.0

        ra = real_advocate({"statement": "s"})
        assert ra["debate_turn"]["error"] == "agent_unavailable"

        judge = judge_agent({"statement": "s"})
        assert judge["verdict"]["label"] == "UNVERIFIED"
        assert judge["weight_breakdown"] == {}

        ev = evaluate_agent({"statement": "s"})
        assert ev["model_results"] == []

        ag = agreement_gate({"statement": "s"})
        assert ag["agreement_score"] == 0.0

        sl = social_loop_agent({"statement": "s"})
        assert sl["social_loop_fired"] is True


# ── config.py timeout settings ─────────────────────────────────────────────


def test_settings_timeout_defaults(monkeypatch):
    for name in (
        "A2A_CLIENT_TIMEOUT",
        "A2A_CLIENT_TIMEOUT_SEARCH",
        "A2A_CLIENT_TIMEOUT_EVALUATE",
        "A2A_CLIENT_TIMEOUT_SOCIAL_LOOP",
        "A2A_CLIENT_TIMEOUT_AGREEMENT_GATE",
        "A2A_CLIENT_TIMEOUT_JUDGE",
        "A2A_CLIENT_TIMEOUT_CONCLUSION",
        "A2A_CLIENT_TIMEOUT_REAL_SOURCE",
        "A2A_CLIENT_TIMEOUT_FAKE_SOURCE",
        "A2A_CLIENT_TIMEOUT_REAL_ADVOCATE",
        "A2A_CLIENT_TIMEOUT_FAKE_ADVOCATE",
    ):
        monkeypatch.delenv(name, raising=False)
    s = Settings()
    assert s.a2a_client_timeout is None
    assert s.a2a_client_timeout_search == 120
    assert s.a2a_client_timeout_evaluate == 60
    assert s.a2a_client_timeout_social_loop == 30
    assert s.a2a_client_timeout_agreement_gate == 30
    assert s.a2a_client_timeout_judge == 120


def test_settings_timeout_global_override_env(monkeypatch):
    # Per 04-01-PLAN the global override is stored as-is (no int() cast);
    # _timeout_for casts to float at use time.
    monkeypatch.setenv("A2A_CLIENT_TIMEOUT", "5")
    assert Settings().a2a_client_timeout == "5"
    monkeypatch.setattr(settings, "a2a_client_timeout", "5")
    assert _timeout_for("search_agent") == 5.0
