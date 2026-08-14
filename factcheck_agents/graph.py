"""LangGraph wiring: Search -> Verify -> (Social?) -> Conclusion.

A single ``FactCheckState`` is threaded through the agent nodes, mirroring
the shared-state design used by TradingAgents.
"""

from __future__ import annotations

import uuid

from langgraph.graph import END, START, StateGraph

from .agents import (
    conclusion_agent,
    search_agent,
    social_search_agent,
    verify_agent,
)
from .agents.agreement_gate import agreement_gate, route_after_agreement
from .agents.expert_agent import expert_agent
from .agents.fake_advocate import fake_advocate
from .agents.fake_source_agent import fake_source_agent
from .agents.judge_agent import judge_agent
from .agents.real_advocate import real_advocate
from .agents.real_source_agent import real_source_agent
from .agents.social_loop_agent import social_loop_agent
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

    for round_num in range(settings.max_debate_rounds):
        real_out = real_advocate(
            {**state, "debate_turns": turns, "debate_role": "real"}
        )
        real_turn = real_out["debate_turn"]
        turns.append(real_turn)
        if real_turn is None or real_turn.get("error"):
            exit_reason = "llm_error" if real_turn else "no_llm"
            break

        fake_out = fake_advocate(
            {**state, "debate_turns": turns, "debate_role": "fake"}
        )
        fake_turn = fake_out["debate_turn"]
        turns.append(fake_turn)
        if fake_turn is None or fake_turn.get("error"):
            exit_reason = "llm_error" if fake_turn else "no_llm"
            break

        real_v = str(real_turn.get("verdict", "")).upper()
        fake_v = str(fake_turn.get("verdict", "")).upper()
        if real_v == fake_v and real_v in {"REAL", "FAKE"}:
            converged = True
            agreed_verdict = real_v
            exit_reason = "converged"
            break

    return {
        "debate_turns": turns,
        "debate_exit_reason": exit_reason or "max_rounds",
        "debate_converged": converged,
        "debate_agreed_verdict": agreed_verdict,
        "messages": [
            (
                "assistant",
                f"[Debate] {len(turns)} turns ({exit_reason or 'max_rounds'})"
                + (f" → agreed={agreed_verdict}" if converged else ""),
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
        use_phobert=True,
        use_coolant=True,
        use_evidence=True,
    )
