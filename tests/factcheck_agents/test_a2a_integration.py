"""Session-scoped graph integration tests for the A2A debate pipeline (A2A-07b).

The ``a2a_agent_servers`` fixture starts 8 graph-path agents (ports 9003–9010)
as daemon threads and tears them down after the session. Both tests invoke
``build_debate_graph()`` with real Vietnamese claims.

All tests are marked integration — excluded from the default pytest run.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from factcheck_agents.a2a_server import create_app
from factcheck_agents.agents.agreement_gate import AgreementGateHandler
from factcheck_agents.agents.conclusion_agent import ConclusionAgentHandler
from factcheck_agents.agents.fake_advocate import FakeAdvocateHandler
from factcheck_agents.agents.fake_source_agent import FakeSourceAgentHandler
from factcheck_agents.agents.judge_agent import JudgeAgentHandler
from factcheck_agents.agents.real_advocate import RealAdvocateHandler
from factcheck_agents.agents.real_source_agent import RealSourceAgentHandler
from factcheck_agents.agents.social_loop_agent import SocialLoopAgentHandler
from factcheck_agents.config import a2a_ports
from factcheck_agents.graph import build_debate_graph, initial_state


@pytest.fixture(scope="session")
def a2a_agent_servers():
    """Start 8 graph-path A2A agents (ports 9003–9010) for the test session."""
    ports = a2a_ports()
    agents_to_start = [
        (RealSourceAgentHandler, ports["real_source_agent"]),  # 9003
        (FakeSourceAgentHandler, ports["fake_source_agent"]),  # 9004
        (SocialLoopAgentHandler, ports["social_loop_agent"]),  # 9005
        (AgreementGateHandler, ports["agreement_gate"]),  # 9006
        (RealAdvocateHandler, ports["real_advocate"]),  # 9007
        (FakeAdvocateHandler, ports["fake_advocate"]),  # 9008
        (JudgeAgentHandler, ports["judge_agent"]),  # 9009
        (ConclusionAgentHandler, ports["conclusion_agent"]),  # 9010
    ]
    servers = []
    for handler_cls, port in agents_to_start:
        with socket.socket() as s:
            result = s.connect_ex(("127.0.0.1", port))
        if result == 0:
            pytest.skip(f"Port {port} already in use — cannot start session fixture")
        app = create_app(handler_cls(), handler_cls.agent_card_config)
        cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(cfg)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        # Poll readiness up to 10 s
        deadline = time.time() + 10
        ready = False
        while time.time() < deadline:
            try:
                r = httpx.get(
                    f"http://127.0.0.1:{port}/.well-known/agent.json", timeout=1
                )
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(0.1)
        if not ready:
            pytest.fail(f"Agent on port {port} did not become ready within 10 s")
        servers.append(server)
    yield
    for server in servers:
        server.should_exit = True


@pytest.fixture(autouse=True)
def _ensure_logs_dir():
    Path("logs").mkdir(exist_ok=True)


@pytest.mark.integration
def test_factcheck_true_claim_a2a(a2a_agent_servers):
    graph = build_debate_graph(checkpointer=None)
    result = graph.invoke(
        initial_state("Hà Nội là thủ đô của Việt Nam"),
        config={"configurable": {"thread_id": "integ-hanoi"}},
    )
    assert isinstance(result, dict)
    assert "verdict" in result
    verdict = result["verdict"]
    assert isinstance(verdict, dict)
    assert verdict.get("label") in {"TRUE", "FALSE", "MISLEADING", "UNVERIFIED", "NEI"}


@pytest.mark.integration
def test_factcheck_false_claim_a2a(a2a_agent_servers):
    graph = build_debate_graph(checkpointer=None)
    result = graph.invoke(
        initial_state("Mặt trăng làm từ phô mai"),
        config={"configurable": {"thread_id": "integ-moon"}},
    )
    assert isinstance(result, dict)
    assert "verdict" in result
    verdict = result["verdict"]
    assert isinstance(verdict, dict)
    assert verdict.get("label") in {"TRUE", "FALSE", "MISLEADING", "UNVERIFIED", "NEI"}
