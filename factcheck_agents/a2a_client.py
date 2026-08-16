"""A2A client module for calling agent services over HTTP."""

from __future__ import annotations

import functools
import logging
import uuid
from typing import Any

import httpx

from .a2a_server import serialize_state
from .config import a2a_ports, settings
from .state import FactCheckState

logger = logging.getLogger(__name__)


class AgentUnavailableError(Exception):
    """Raised when an A2A agent service is unreachable or returns a failed task."""

    def __init__(self, agent_name: str = "", port: int = 0, cause: str = ""):
        self.agent_name = agent_name
        self.port = port
        self.cause = cause
        super().__init__(f"[{agent_name}:{port}] unavailable: {cause}")


# Agent name -> Settings field suffix for per-agent timeouts.
_SUFFIX_BY_AGENT = {
    "search_agent": "search",
    "evaluate_agent": "evaluate",
    "real_source_agent": "real_source",
    "fake_source_agent": "fake_source",
    "social_loop_agent": "social_loop",
    "agreement_gate": "agreement_gate",
    "real_advocate": "real_advocate",
    "fake_advocate": "fake_advocate",
    "judge_agent": "judge",
    "conclusion_agent": "conclusion",
}


def _timeout_for(agent_name: str) -> float:
    """Resolve the httpx timeout for an agent: global override -> per-agent field."""
    if settings.a2a_client_timeout is not None:
        return float(settings.a2a_client_timeout)
    suffix = _SUFFIX_BY_AGENT.get(agent_name, agent_name)
    return float(getattr(settings, f"a2a_client_timeout_{suffix}"))


def _extract_artifact_data(artifacts: list) -> dict:
    """Extract ``parts[0]["data"]`` from the ``"output"`` artifact of a Task."""
    for artifact in artifacts:
        if artifact.get("name") == "output":
            parts = artifact.get("parts", [])
            if parts and "data" in parts[0]:
                return parts[0]["data"]
    return {}


def call_agent(agent_name: str, state: FactCheckState) -> dict:
    """Send the serialized ``FactCheckState`` to an A2A agent and return its diff.

    Posts ``POST /message:send`` on the agent's port and returns the
    deserialized state diff from the ``output`` artifact. Raises
    :class:`AgentUnavailableError` when the agent is unreachable or the task
    ends in the failed state.
    """
    port = a2a_ports()[agent_name]
    timeout = _timeout_for(agent_name)
    payload = serialize_state(dict(state))
    request_body = {
        "message": {
            "role": "user",
            "parts": [{"data": payload}],
            "messageId": f"msg-{uuid.uuid4()}",
        }
    }
    try:
        resp = httpx.Client(timeout=timeout).post(
            f"http://localhost:{port}/message:send",
            json=request_body,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise AgentUnavailableError(agent_name, port, str(exc)) from exc

    task = resp.json()
    task_state = task.get("status", {}).get("state", "")
    artifacts = task.get("artifacts", [])

    if task_state == "TASK_STATE_FAILED":
        error_data = _extract_artifact_data(artifacts)
        raise AgentUnavailableError(
            agent_name, port, error_data.get("error", "server error")
        )
    if task_state == "TASK_STATE_WORKING":
        logger.warning(
            "[%s:%s] task still working after POST returned — treating as empty diff",
            agent_name,
            port,
        )
        return {}
    # TASK_STATE_COMPLETED (or unknown) — extract the diff.
    return _extract_artifact_data(artifacts)


def degrade_on_unavailable(agent_name: str, degrade_diff: dict):
    """Decorator factory: catch :class:`AgentUnavailableError`, return ``degrade_diff``."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            try:
                return fn(state, *args, **kwargs)
            except AgentUnavailableError as exc:
                logger.warning(
                    "[%s] agent unavailable (port %s): %s — degraded",
                    agent_name,
                    exc.port,
                    exc.cause,
                )
                return dict(degrade_diff)  # shallow copy to avoid mutation

        return wrapper

    return decorator


# ── Per-agent degrade diffs (total=False — only keys the agent produces) ──


_DEGRADE_SEARCH = {
    "evidence": [],
    "search_queries": [],
    "evidence_graph": None,
    "claim_variants": [],
    "messages": [("assistant", "[Search] agent unavailable — degraded")],
}

_DEGRADE_CONCLUSION = {
    "verdict": {
        "label": "UNVERIFIED",
        "confidence": 0.0,
        "verdict_binary": "NEI",
        "verdict_label_vi": "Chưa xác thực",
        "rationale": "Conclusion agent unavailable",
        "citations": [],
        "recommendation": "",
    }
}

_DEGRADE_REAL_SOURCE = {
    "evidence_real": [],
    "messages": [("assistant", "[RealSource] agent unavailable — degraded")],
}

_DEGRADE_FAKE_SOURCE = {
    "evidence_fake": [],
    "messages": [("assistant", "[FakeSource] agent unavailable — degraded")],
}

_DEGRADE_SOCIAL_LOOP = {
    "evidence_social": [],
    "social_loop_fired": True,  # prevents re-entry (SOCLOOP-03)
    "messages": [("assistant", "[SocialLoop] agent unavailable — degraded")],
}

_DEGRADE_AGREEMENT = {
    "agreement_score": 0.0,
    "messages": [("assistant", "[AgreementGate] agent unavailable — degraded")],
}

_DEGRADE_REAL_ADVOCATE = {
    "debate_turn": {"agent": "real_advocate", "error": "agent_unavailable"},
    "messages": [("assistant", "[RealAdvocate] agent unavailable — degraded")],
}

_DEGRADE_FAKE_ADVOCATE = {
    "debate_turn": {"agent": "fake_advocate", "error": "agent_unavailable"},
    "messages": [("assistant", "[FakeAdvocate] agent unavailable — degraded")],
}

_DEGRADE_JUDGE = {
    "verdict": {
        "label": "UNVERIFIED",
        "confidence": 0.0,
        "verdict_binary": "NEI",
        "verdict_label_vi": "Chưa xác thực",
        "rationale": "Judge agent unavailable",
        "citations": [],
        "recommendation": "",
    },
    "weight_breakdown": {},
}

_DEGRADE_EVALUATE = {
    "model_results": [],
    "messages": [("assistant", "[Evaluate] agent unavailable — degraded")],
}
