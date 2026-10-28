"""LangGraph wiring: Search -> Verify -> (Social?) -> Conclusion.

A single ``FactCheckState`` is threaded through the agent nodes, mirroring
the shared-state design used by TradingAgents.
"""

from __future__ import annotations

import logging
import os
import uuid

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

from .agents import social_search_agent, verify_agent
from .agents.agreement_gate import route_after_agreement
from .agents.expert_agent import expert_agent
from .a2a_client import (
    AgentUnavailableError,
    agreement_gate,
    conclusion_agent,
    fake_advocate,
    fake_source_agent,
    judge_agent,
    real_advocate,
    real_source_agent,
    search_agent,
    social_loop_agent,
)
from .config import settings
from .reranker import reranker as reranker_node
from .state import FactCheckState


def route_after_verify(state: FactCheckState) -> str:
    """Route after verify based on reliability_signal."""
    if state.get("reliability_signal"):
        return "social_search"
    return "conclusion"


def route_nei_check(state: FactCheckState) -> str:
    """EVRET-04: NEI short-circuit when both evidence lists are empty after fan-out."""
    real = state.get("evidence_real") or []
    fake = state.get("evidence_fake") or []
    return "judge" if not real and not fake else "reranker"


def route_social_loop(state: FactCheckState) -> str:
    """SOCLOOP-01/02/03: fire social loop at most once, only when evidence is weak."""
    real = state.get("evidence_real") or []
    fake = state.get("evidence_fake") or []
    count = len(real) + len(fake)
    cred = state.get("consistency_score", 0.1)
    if (
        not state.get("social_loop_fired", False)
        and count < settings.social_loop_min_count
        and cred < settings.social_loop_min_credibility
    ):
        return "social_loop"
    return "verify"


def build_graph(checkpointer=None):
    """Compile and return the fact-checking graph."""
    if checkpointer is None:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            checkpointer = SqliteSaver.from_conn_string(settings.checkpoint_db)
        except ImportError:
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()

    g = StateGraph(FactCheckState)
    g.add_node("search", search_agent)
    g.add_node("verify", verify_agent)
    g.add_node("social_search", social_search_agent)
    g.add_node("conclusion", conclusion_agent)

    g.add_edge(START, "search")
    g.add_edge("search", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"social_search": "social_search", "conclusion": "conclusion"},
    )
    g.add_edge("social_search", "conclusion")
    g.add_edge("conclusion", END)
    # Note for Phase 7: g.invoke(state) without config={"configurable":
    # {"thread_id": "..."}} uses LangGraph's null default thread — safe for
    # existing callers. Wire thread_id into cli.py / run_fact_check() in Phase 7.
    return g.compile(checkpointer=checkpointer)


def debate_node(state: FactCheckState) -> dict:
    """Run convergence-driven debate between real/fake advocate agents.

    The advocates (``real_advocate`` / ``fake_advocate``) are single-turn
    services (D-07); this node orchestrates the loop: full history is passed
    each round, the debate exits when both advocates agree on the same
    verdict (REAL/FAKE) or when ``max_debate_rounds`` is hit. Phase 4 will
    replace these direct calls with A2A client invocations.

    Returns the same keys as the former ``debate_node``:
    ``debate_turns``, ``debate_exit_reason``, ``debate_converged``,
    ``debate_agreed_verdict``, ``messages``.
    """
    from pathlib import Path

    from .agents.llm import get_llm

    llm = get_llm()
    if llm is None:
        return {
            "debate_turns": [],
            "debate_exit_reason": "no_llm",
            "debate_converged": False,
            "debate_agreed_verdict": None,
            "messages": [("assistant", "[Debate] skipped — no LLM configured")],
        }

    Path("logs/debates").mkdir(parents=True, exist_ok=True)

    turns = list(state.get("debate_turns") or [])
    exit_reason = None
    converged = False
    agreed_verdict = None
    down = {"real": False, "fake": False}
    agents = {"real": real_advocate, "fake": fake_advocate}
    last_verdict: dict = {}
    passes = 0  # consecutive passes; both sides passing ends the debate
    max_turns = max(2, 2 * settings.max_debate_rounds)
    judge_interject = os.getenv("FACTCHECK_JUDGE_INTERJECT", "1") != "0"
    speaker = "real"
    judge_reason = ""

    # Free-floor debate: no fixed alternation. A side may pass (nothing new to
    # say) and then the other side keeps the floor; two passes in a row end the
    # debate. The judge may cut in after any turn and conclude early.
    while len(turns) < max_turns:
        if down["real"] and down["fake"]:
            exit_reason = "agent_unavailable"
            break
        if down[speaker]:
            speaker = "fake" if speaker == "real" else "real"
        try:
            out = agents[speaker]({**state, "debate_turns": turns, "debate_role": speaker})
        except AgentUnavailableError:
            down[speaker] = True
            turns.append({"agent": f"{speaker}_advocate", "error": "agent_unavailable"})
            logger.warning("[Debate] %s_advocate unavailable — partial debate", speaker)
            continue
        turn = out.get("debate_turn")
        if turn is None or turn.get("error"):
            if turn is not None:
                turns.append(turn)
                logger.error(
                    "[Debate] %s_advocate LLM error at turn %d: %s",
                    speaker, len(turns), turn.get("error"),
                )
            exit_reason = "llm_error" if turn else "no_llm"
            break

        if turn.get("passed"):
            passes += 1
            logger.info("[Debate] %s_advocate passes", speaker)
            if passes >= 2 or (down["real"] or down["fake"]):
                exit_reason = "both_passed"
                break
            speaker = "fake" if speaker == "real" else "real"
            continue

        passes = 0
        turns.append(turn)
        last_verdict[speaker] = str(turn.get("verdict", "")).upper()

        rv, fv = last_verdict.get("real"), last_verdict.get("fake")
        if rv and rv == fv and rv in {"REAL", "FAKE"}:
            converged = True
            agreed_verdict = rv
            exit_reason = "converged"
            break

        if judge_interject:
            from .agents.judge_agent import judge_should_conclude

            stop, reason = judge_should_conclude(state, turns)
            if stop:
                exit_reason = "judge_concluded"
                judge_reason = reason
                logger.info("[Debate] judge concluded early: %s", reason)
                break

        # The opponent normally answers, but may be skipped if it just passed.
        speaker = "fake" if speaker == "real" else "real"

    return {
        "debate_turns": turns,
        "debate_exit_reason": exit_reason or "max_rounds",
        "debate_converged": converged,
        "debate_agreed_verdict": agreed_verdict,
        "messages": [
            (
                "assistant",
                f"[Debate] {len(turns)} turns ({exit_reason or 'max_rounds'})"
                + (f" → agreed={agreed_verdict}" if converged else "")
                + (f" — judge: {judge_reason}" if judge_reason else ""),
            )
        ],
    }


def route_after_start(state: FactCheckState) -> str:
    """Skip evidence retrieval when use_evidence=False (ablation)."""
    if not state.get("use_evidence", True):
        return "nei_gate"
    return "fan_out"


def build_debate_graph(checkpointer=None):
    """Build M2 debate graph topology with static fan-out (D-12).

    Topology: START -> conditional(use_evidence?) -> (real_source, fake_source) or nei_gate -> (reranker -> social_loop? -> verify -> agreement_gate -> debate? -> judge) -> END
    """
    from langgraph.checkpoint.memory import MemorySaver

    if checkpointer is None:
        checkpointer = MemorySaver()

    g = StateGraph(FactCheckState)
    g.add_node("real_source", real_source_agent)
    g.add_node("fake_source", fake_source_agent)
    g.add_node("nei_gate", lambda state: {})  # barrier; routing via conditional edge
    g.add_node("reranker", reranker_node)
    g.add_node("social_loop", social_loop_agent)
    g.add_node("verify", verify_agent)
    g.add_node("agreement_gate", agreement_gate)
    g.add_node("debate", debate_node)
    g.add_node("judge", judge_agent)
    g.add_node("expert", expert_agent)

    # LangGraph supports both conditional + static edges from START for fan-out
    g.add_conditional_edges(
        START, route_after_start, {"fan_out": "real_source", "nei_gate": "nei_gate"}
    )
    g.add_edge(
        START, "fake_source"
    )  # static fan-out; still fires even when use_evidence=False
    g.add_edge("real_source", "nei_gate")  # implicit barrier merge
    g.add_edge("fake_source", "nei_gate")
    g.add_conditional_edges(
        "nei_gate", route_nei_check, {"reranker": "reranker", "judge": "judge"}
    )
    g.add_conditional_edges(
        "reranker",
        route_social_loop,
        {"social_loop": "social_loop", "verify": "verify"},
    )
    g.add_edge("social_loop", "verify")
    g.add_edge("verify", "agreement_gate")
    g.add_conditional_edges(
        "agreement_gate",
        route_after_agreement,
        {"debate": "debate", "judge": "judge"},
    )
    g.add_edge("debate", "judge")
    g.add_edge("judge", "expert")
    g.add_edge("expert", END)
    return g.compile(checkpointer=checkpointer)


def initial_state(
    statement: str, image_path: str | None = None, language: str = "auto"
) -> FactCheckState:
    return FactCheckState(
        statement=statement,
        image_path=image_path,
        language=language,
        search_queries=[],
        claim_variants=[],
        evidence=[],
        model_results=[],
        verdict={},
        messages=[],
        errors=[],
        meta={},
        request_id=str(uuid.uuid4()),
        social_loop_fired=False,
        evidence_real=[],
        evidence_fake=[],
        evidence_social=[],
        debate_turns=[],
        debate_converged=False,
        debate_agreed_verdict=None,
        use_phobert=settings.use_phobert,
        use_coolant=settings.use_coolant,
        use_evidence=settings.use_evidence,
    )
