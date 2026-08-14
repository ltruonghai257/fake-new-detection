"""Real advocate agent: defends the claim as REAL in the adversarial debate.

Split from the former ``debate_node.py`` (D-07). This module is a SINGLE-TURN
advocate: it reads the current debate history from state, produces one turn
arguing the claim is REAL, and returns that turn. The LangGraph debate node
(orchestrating convergence / round limits) lives in ``graph.py``; Phase 4 will
call this service over A2A instead.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from ..config import settings
from ..state import Evidence, FactCheckState
from .debate_utils import (
    REAL_ADVOCATE_PROMPT,
    _append_turn,
    _build_advocate_user_message,
    _parse_advocate_json,
)
from .llm import get_llm


def real_advocate(state: FactCheckState) -> dict:
    """Produce a single debate turn defending the claim as REAL.

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
            "messages": [("assistant", "[RealAdvocate] skipped — no LLM configured")],
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
    prompt = settings.real_advocate_prompt or REAL_ADVOCATE_PROMPT

    try:
        resp = llm.invoke([("system", prompt), ("user", user_message)])
        content = str(getattr(resp, "content", ""))
        data = _parse_advocate_json(content) or {}
        turn = {
            "agent": "real_advocate",
            "round": round_num,
            "timestamp": datetime.utcnow().isoformat(),
            "verdict": data.get("verdict", "REAL"),
            "confidence": float(data.get("confidence", 0.5)),
            "argument": data.get("argument", content[:500]),
            "concession": data.get("concession"),
        }
        msg = f"[RealAdvocate] round {round_num} → {turn['verdict']} ({turn['confidence']:.2f})"
    except Exception as exc:
        turn = {
            "agent": "real_advocate",
            "round": round_num,
            "timestamp": datetime.utcnow().isoformat(),
            "verdict": None,
            "confidence": 0.0,
            "argument": "",
            "concession": None,
            "error": str(exc),
        }
        msg = f"[RealAdvocate] round {round_num} failed: {exc}"

    _append_turn(request_id, turn)
    return {"debate_turn": turn, "messages": [("assistant", msg)]}


# ── A2A service wrapper ─────────────────────────────────────────────────────
from ..a2a_server import AgentCardConfig, BaseTaskHandler, run_server


class RealAdvocateHandler(BaseTaskHandler):
    """A2A TaskHandler exposing :func:`real_advocate` over HTTP (port 9007)."""

    agent_card_config = AgentCardConfig(
        name="real_advocate",
        description="Defends claim as REAL in adversarial debate",
        version="1.0",
        skills=[
            {"id": "advocacy", "name": "Real Advocacy", "description": "Argue the claim is REAL in a single debate turn"}
        ],
        port=settings.a2a_port_real_advocate,
    )

    async def agent_fn(self, state: FactCheckState) -> dict:
        # Contract D-05: the task input carries debate_role; enforce it so a
        # misrouted task fails loudly instead of arguing the wrong side.
        role = state.get("debate_role")
        if role is not None and role != "real":
            raise ValueError(
                f"RealAdvocateHandler got debate_role={role!r}; expected 'real'"
            )
        return real_advocate(state)


if __name__ == "__main__":
    run_server(RealAdvocateHandler(), RealAdvocateHandler.agent_card_config)
