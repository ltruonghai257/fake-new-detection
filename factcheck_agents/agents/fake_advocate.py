"""Fake advocate agent: defends the claim as FAKE in the adversarial debate.

Split from the former ``debate_node.py`` (D-07). This module is a SINGLE-TURN
advocate: it reads the current debate history from state, produces one turn
arguing the claim is FAKE, and returns that turn. The LangGraph debate node
(orchestrating convergence / round limits) lives in ``graph.py``; Phase 4 will
call this service over A2A instead.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from ..config import settings
from ..state import Evidence, FactCheckState
from .debate_utils import (
    FAKE_ADVOCATE_PROMPT,
    _append_turn,
    _build_advocate_user_message,
    _parse_advocate_json,
)
from .llm import get_llm


def fake_advocate(state: FactCheckState) -> dict:
    """Produce a single debate turn defending the claim as FAKE.

    Returns:
        dict with keys:
            - debate_turn: structured turn dict (or None when no LLM is
              configured / on parse failure the turn carries an "error" key)
            - messages: list of message tuples
    """
    llm = get_llm()
    if llm is None:
        return {
            "debate_turn": None,
            "messages": [("assistant", "[FakeAdvocate] skipped — no LLM configured")],
        }

    statement = state["statement"]
    evidence_real: List[Evidence] = state.get("evidence_real") or []
    evidence_fake: List[Evidence] = state.get("evidence_fake") or []
    all_evidence = evidence_real + evidence_fake
    model_results = state.get("model_results") or []
    turns = state.get("debate_turns") or []
    request_id = state.get("request_id", "unknown")
    round_num = len(turns) // 2

    user_message = _build_advocate_user_message(
        statement, model_results, all_evidence, turns
    )
    prompt = settings.fake_advocate_prompt or FAKE_ADVOCATE_PROMPT

    try:
        resp = llm.invoke([("system", prompt), ("user", user_message)])
        content = str(getattr(resp, "content", ""))
        data = _parse_advocate_json(content) or {}
        turn = {
            "agent": "fake_advocate",
            "round": round_num,
            "timestamp": datetime.utcnow().isoformat(),
            "verdict": data.get("verdict", "FAKE"),
            "confidence": float(data.get("confidence", 0.5)),
            "argument": data.get("argument", content[:500]),
            "concession": data.get("concession"),
        }
        msg = f"[FakeAdvocate] round {round_num} → {turn['verdict']} ({turn['confidence']:.2f})"
    except Exception as exc:
        turn = {
            "agent": "fake_advocate",
            "round": round_num,
            "timestamp": datetime.utcnow().isoformat(),
            "verdict": None,
            "confidence": 0.0,
            "argument": "",
            "concession": None,
            "error": str(exc),
        }
        msg = f"[FakeAdvocate] round {round_num} failed: {exc}"

    _append_turn(request_id, turn)
    return {"debate_turn": turn, "messages": [("assistant", msg)]}


# ── A2A service wrapper ─────────────────────────────────────────────────────
from ..a2a_server import AgentCardConfig, BaseTaskHandler, run_server


class FakeAdvocateHandler(BaseTaskHandler):
    """A2A TaskHandler exposing :func:`fake_advocate` over HTTP (port 9008)."""

    agent_card_config = AgentCardConfig(
        name="fake_advocate",
        description="Defends claim as FAKE in adversarial debate",
        version="1.0",
        skills=[
            {"id": "advocacy", "name": "Fake Advocacy", "description": "Argue the claim is FAKE in a single debate turn"}
        ],
        port=settings.a2a_port_fake_advocate,
    )

    async def agent_fn(self, state: FactCheckState) -> dict:
        # Contract D-05: the task input carries debate_role; enforce it so a
        # misrouted task fails loudly instead of arguing the wrong side.
        role = state.get("debate_role")
        if role is not None and role != "fake":
            raise ValueError(
                f"FakeAdvocateHandler got debate_role={role!r}; expected 'fake'"
            )
        return fake_advocate(state)


if __name__ == "__main__":
    run_server(FakeAdvocateHandler(), FakeAdvocateHandler.agent_card_config)
