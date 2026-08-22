"""In-process HTTP tests for all 10 A2A TaskHandlers (requirement A2A-07).

Each test starts a real uvicorn server in a background daemon thread,
sends a real A2A Task via httpx, and asserts TASK_STATE_COMPLETED.
All tests are marked integration — excluded from the default
pytest run (`pytest tests/ -m "not integration"`).
"""

from __future__ import annotations

import socket
import threading
import time
import uuid

import httpx
import pytest
import uvicorn

from factcheck_agents.a2a_server import create_app
from factcheck_agents.agents.agreement_gate import AgreementGateHandler
from factcheck_agents.agents.conclusion_agent import ConclusionAgentHandler
from factcheck_agents.agents.evaluate_agent import EvaluateAgentHandler
from factcheck_agents.agents.fake_advocate import FakeAdvocateHandler
from factcheck_agents.agents.fake_source_agent import FakeSourceAgentHandler
from factcheck_agents.agents.judge_agent import JudgeAgentHandler
from factcheck_agents.agents.real_advocate import RealAdvocateHandler
from factcheck_agents.agents.real_source_agent import RealSourceAgentHandler
from factcheck_agents.agents.search_agent import SearchAgentHandler
from factcheck_agents.agents.social_loop_agent import SocialLoopAgentHandler
from factcheck_agents.config import a2a_ports


# ── Shared helpers ──────────────────────────────────────────────────────────


def _port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start_agent_server(handler_cls, port: int) -> uvicorn.Server:
    app = create_app(handler_cls(), handler_cls.agent_card_config)
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/.well-known/agent.json", timeout=1)
            if r.status_code == 200:
                return server
        except Exception:
            time.sleep(0.1)
    pytest.fail(f"Server on port {port} did not become ready within 10 s")
    return server  # unreachable; silences mypy


def _send_task(port: int, state: dict) -> dict:
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"http://127.0.0.1:{port}/message:send",
            json={
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"data": state}],
                    "messageId": f"msg-{uuid.uuid4()}",
                }
            },
            headers={"A2A-Version": "1.0"},
        )
    resp.raise_for_status()
    return resp.json()


# ── Minimal state stubs per agent ───────────────────────────────────────────

_SEARCH_STATE = {
    "statement": "Hà Nội là thủ đô của Việt Nam",
    "claim_variants": [],
    "search_queries": [],
    "messages": [],
}

_EVALUATE_STATE = {
    "statement": "test",
    "evidence": [],
    "messages": [],
}

_SOURCE_STATE = {
    "statement": "Hà Nội là thủ đô của Việt Nam",
    "evidence_real": [],
    "evidence_fake": [],
    "messages": [],
}

_SOCIAL_STATE = {
    "statement": "test",
    "evidence_real": [],
    "evidence_fake": [],
    "social_loop_fired": False,
    "messages": [],
}

_AGREEMENT_STATE = {
    "statement": "test",
    "model_results": [
        {"label": "TRUE", "confidence": 0.9, "model": "phobert"},
        {"label": "TRUE", "confidence": 0.85, "model": "coolant"},
    ],
    "evidence": [],
    "messages": [],
}

_ADVOCATE_STATE = {
    "statement": "test",
    "evidence_real": [
        {
            "title": "T",
            "url": "https://vnexpress.net/a",
            "snippet": "s",
            "source_tier": "trusted",
            "score": 0.9,
        }
    ],
    "evidence_fake": [],
    "debate_turns": [],
    "messages": [],
}

_JUDGE_STATE = {
    "statement": "test",
    "debate_turns": [],
    "model_results": [{"label": "TRUE", "confidence": 0.9, "model": "phobert"}],
    "evidence": [],
    "messages": [],
}

_CONCLUSION_STATE = {
    "statement": "test",
    "verdict": {},
    "weight_breakdown": {},
    "model_results": [],
    "messages": [],
}


# ── 10 integration test functions ───────────────────────────────────────────


@pytest.mark.integration
def test_search_agent_http():
    port = a2a_ports()["search_agent"]
    if _port_in_use(port):
        pytest.skip(
            f"Port {port} already in use — skip to avoid conflict with running agent server"
        )
    server = _start_agent_server(SearchAgentHandler, port)
    try:
        resp = _send_task(port, _SEARCH_STATE)
        task = resp.get("task") or resp
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        diff = task["artifacts"][0]["parts"][0]["data"]
        assert isinstance(diff, dict)
        assert "evidence" in diff or "search_queries" in diff
    finally:
        server.should_exit = True


@pytest.mark.integration
def test_evaluate_agent_http():
    import os

    if not os.getenv("VIFACTCHECK_CKPT_DIR") and not os.getenv("COOLANT_CKPT_PATH"):
        pytest.skip(
            "VIFACTCHECK_CKPT_DIR not set — evaluate_agent test requires model checkpoints"
        )
    port = a2a_ports()["evaluate_agent"]
    if _port_in_use(port):
        pytest.skip(
            f"Port {port} already in use — skip to avoid conflict with running agent server"
        )
    server = _start_agent_server(EvaluateAgentHandler, port)
    try:
        resp = _send_task(port, _EVALUATE_STATE)
        task = resp.get("task") or resp
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        diff = task["artifacts"][0]["parts"][0]["data"]
        assert isinstance(diff, dict)
        assert "model_results" in diff
    finally:
        server.should_exit = True


@pytest.mark.integration
def test_real_source_agent_http():
    port = a2a_ports()["real_source_agent"]
    if _port_in_use(port):
        pytest.skip(
            f"Port {port} already in use — skip to avoid conflict with running agent server"
        )
    server = _start_agent_server(RealSourceAgentHandler, port)
    try:
        resp = _send_task(port, _SOURCE_STATE)
        task = resp.get("task") or resp
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        diff = task["artifacts"][0]["parts"][0]["data"]
        assert isinstance(diff, dict)
        assert "evidence_real" in diff
    finally:
        server.should_exit = True


@pytest.mark.integration
def test_fake_source_agent_http():
    port = a2a_ports()["fake_source_agent"]
    if _port_in_use(port):
        pytest.skip(
            f"Port {port} already in use — skip to avoid conflict with running agent server"
        )
    server = _start_agent_server(FakeSourceAgentHandler, port)
    try:
        resp = _send_task(port, _SOURCE_STATE)
        task = resp.get("task") or resp
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        diff = task["artifacts"][0]["parts"][0]["data"]
        assert isinstance(diff, dict)
        assert "evidence_fake" in diff
    finally:
        server.should_exit = True


@pytest.mark.integration
def test_social_loop_agent_http():
    port = a2a_ports()["social_loop_agent"]
    if _port_in_use(port):
        pytest.skip(
            f"Port {port} already in use — skip to avoid conflict with running agent server"
        )
    server = _start_agent_server(SocialLoopAgentHandler, port)
    try:
        resp = _send_task(port, _SOCIAL_STATE)
        task = resp.get("task") or resp
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        diff = task["artifacts"][0]["parts"][0]["data"]
        assert isinstance(diff, dict)
        assert "social_loop_fired" in diff
    finally:
        server.should_exit = True


@pytest.mark.integration
def test_agreement_gate_http():
    port = a2a_ports()["agreement_gate"]
    if _port_in_use(port):
        pytest.skip(
            f"Port {port} already in use — skip to avoid conflict with running agent server"
        )
    server = _start_agent_server(AgreementGateHandler, port)
    try:
        resp = _send_task(port, _AGREEMENT_STATE)
        task = resp.get("task") or resp
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        diff = task["artifacts"][0]["parts"][0]["data"]
        assert isinstance(diff, dict)
        assert "agreement_score" in diff
    finally:
        server.should_exit = True


@pytest.mark.integration
def test_real_advocate_http():
    port = a2a_ports()["real_advocate"]
    if _port_in_use(port):
        pytest.skip(
            f"Port {port} already in use — skip to avoid conflict with running agent server"
        )
    server = _start_agent_server(RealAdvocateHandler, port)
    try:
        resp = _send_task(port, _ADVOCATE_STATE)
        task = resp.get("task") or resp
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        diff = task["artifacts"][0]["parts"][0]["data"]
        assert isinstance(diff, dict)
        assert "debate_turn" in diff or "messages" in diff
    finally:
        server.should_exit = True


@pytest.mark.integration
def test_fake_advocate_http():
    port = a2a_ports()["fake_advocate"]
    if _port_in_use(port):
        pytest.skip(
            f"Port {port} already in use — skip to avoid conflict with running agent server"
        )
    server = _start_agent_server(FakeAdvocateHandler, port)
    try:
        resp = _send_task(port, _ADVOCATE_STATE)
        task = resp.get("task") or resp
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        diff = task["artifacts"][0]["parts"][0]["data"]
        assert isinstance(diff, dict)
        assert "debate_turn" in diff or "messages" in diff
    finally:
        server.should_exit = True


@pytest.mark.integration
def test_judge_agent_http():
    port = a2a_ports()["judge_agent"]
    if _port_in_use(port):
        pytest.skip(
            f"Port {port} already in use — skip to avoid conflict with running agent server"
        )
    server = _start_agent_server(JudgeAgentHandler, port)
    try:
        resp = _send_task(port, _JUDGE_STATE)
        task = resp.get("task") or resp
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        diff = task["artifacts"][0]["parts"][0]["data"]
        assert isinstance(diff, dict)
        assert "weight_breakdown" in diff
    finally:
        server.should_exit = True


@pytest.mark.integration
def test_conclusion_agent_http():
    port = a2a_ports()["conclusion_agent"]
    if _port_in_use(port):
        pytest.skip(
            f"Port {port} already in use — skip to avoid conflict with running agent server"
        )
    server = _start_agent_server(ConclusionAgentHandler, port)
    try:
        resp = _send_task(port, _CONCLUSION_STATE)
        task = resp.get("task") or resp
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        diff = task["artifacts"][0]["parts"][0]["data"]
        assert isinstance(diff, dict)
        assert "verdict" in diff
    finally:
        server.should_exit = True
