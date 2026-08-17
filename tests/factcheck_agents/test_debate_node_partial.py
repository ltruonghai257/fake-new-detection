"""Behavioral tests for debate_node partial-debate semantics (requirement A2A-05b / D-05).

Covers: both advocates down -> exit_reason "agent_unavailable"; one-sided
debates still give the available advocate full turns; convergence preserved
when both are available; and IN-07 (None turns are never appended).

Patch notes:
- Advocates are patched at ``factcheck_agents.graph.<name>`` (from-import
  bindings — patching the a2a_client namespace does NOT intercept graph's
  bindings, a documented phase-04 lesson).
- ``get_llm`` is imported function-locally inside debate_node, so the patch
  target is the source module ``factcheck_agents.agents.llm.get_llm``.
- ``max_debate_rounds`` is read from the shared Settings instance at loop
  time, so monkeypatching the instance attribute keeps tests fast.
"""

from unittest.mock import patch

import pytest

from factcheck_agents.a2a_client import AgentUnavailableError
from factcheck_agents.config import settings
from factcheck_agents.graph import debate_node


def _state(**overrides):
    state = {
        "statement": "Tuyên bố kiểm tra",
        "debate_turns": [],
        "evidence_real": [{"url": "https://real.com"}],
        "evidence_fake": [{"url": "https://fake.com"}],
        "request_id": "test-debate",
    }
    state.update(overrides)
    return state


@pytest.fixture(autouse=True)
def _fast_llm():
    with patch("factcheck_agents.agents.llm.get_llm", return_value=object()):
        yield


def test_both_advocates_down_exits_agent_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "max_debate_rounds", 3)
    with patch(
        "factcheck_agents.graph.real_advocate",
        side_effect=AgentUnavailableError("real_advocate", 9007, "down"),
    ), patch(
        "factcheck_agents.graph.fake_advocate",
        side_effect=AgentUnavailableError("fake_advocate", 9008, "down"),
    ):
        result = debate_node(_state())

    assert result["debate_exit_reason"] == "agent_unavailable"
    assert result["debate_converged"] is False
    turns = result["debate_turns"]
    assert {"agent": "real_advocate", "error": "agent_unavailable"} in turns
    assert {"agent": "fake_advocate", "error": "agent_unavailable"} in turns


def test_real_down_fake_still_gets_turns(monkeypatch):
    monkeypatch.setattr(settings, "max_debate_rounds", 2)
    fake_turn = {"agent": "fake_advocate", "verdict": "FAKE", "rationale": "r"}
    with patch(
        "factcheck_agents.graph.real_advocate",
        side_effect=AgentUnavailableError("real_advocate", 9007, "down"),
    ) as mock_real, patch(
        "factcheck_agents.graph.fake_advocate",
        return_value={"debate_turn": fake_turn},
    ) as mock_fake:
        result = debate_node(_state())

    assert result["debate_exit_reason"] == "max_rounds"
    assert result["debate_converged"] is False
    fake_turns = [
        t
        for t in result["debate_turns"]
        if t.get("agent") == "fake_advocate" and "error" not in t
    ]
    assert len(fake_turns) == 2  # one full turn per round for the available side
    assert mock_real.call_count == 1  # marked down after first failure
    assert mock_fake.call_count == 2


def test_fake_down_real_still_gets_turns(monkeypatch):
    monkeypatch.setattr(settings, "max_debate_rounds", 2)
    real_turn = {"agent": "real_advocate", "verdict": "REAL", "rationale": "r"}
    with patch(
        "factcheck_agents.graph.fake_advocate",
        side_effect=AgentUnavailableError("fake_advocate", 9008, "down"),
    ) as mock_fake, patch(
        "factcheck_agents.graph.real_advocate",
        return_value={"debate_turn": real_turn},
    ) as mock_real:
        result = debate_node(_state())

    assert result["debate_exit_reason"] == "max_rounds"
    assert result["debate_converged"] is False
    real_turns = [
        t
        for t in result["debate_turns"]
        if t.get("agent") == "real_advocate" and "error" not in t
    ]
    assert len(real_turns) == 2  # one full turn per round for the available side
    assert mock_fake.call_count == 1  # marked down after first failure
    assert mock_real.call_count == 2


def test_both_available_agree_converges(monkeypatch):
    monkeypatch.setattr(settings, "max_debate_rounds", 5)
    with patch(
        "factcheck_agents.graph.real_advocate",
        return_value={"debate_turn": {"agent": "real_advocate", "verdict": "REAL"}},
    ), patch(
        "factcheck_agents.graph.fake_advocate",
        return_value={"debate_turn": {"agent": "fake_advocate", "verdict": "REAL"}},
    ):
        result = debate_node(_state())

    assert result["debate_exit_reason"] == "converged"
    assert result["debate_converged"] is True
    assert result["debate_agreed_verdict"] == "REAL"


def test_none_turn_not_appended_real_side(monkeypatch):
    monkeypatch.setattr(settings, "max_debate_rounds", 3)
    with patch(
        "factcheck_agents.graph.real_advocate", return_value={"debate_turn": None}
    ), patch(
        "factcheck_agents.graph.fake_advocate",
        return_value={"debate_turn": {"agent": "fake_advocate", "verdict": "FAKE"}},
    ):
        result = debate_node(_state())

    assert all(t is not None for t in result["debate_turns"])
    assert result["debate_turns"] == []


def test_none_turn_not_appended_fake_side(monkeypatch):
    monkeypatch.setattr(settings, "max_debate_rounds", 3)
    with patch(
        "factcheck_agents.graph.real_advocate",
        return_value={"debate_turn": {"agent": "real_advocate", "verdict": "REAL"}},
    ), patch(
        "factcheck_agents.graph.fake_advocate", return_value={"debate_turn": None}
    ):
        result = debate_node(_state())

    assert all(t is not None for t in result["debate_turns"])
    assert len(result["debate_turns"]) == 1
    assert result["debate_turns"][0]["agent"] == "real_advocate"
