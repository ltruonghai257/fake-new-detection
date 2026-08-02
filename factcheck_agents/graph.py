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
from .agents.debate_node import debate_node
from .agents.fake_source_agent import fake_source_agent
from .agents.judge_agent import judge_agent
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


def build_debate_graph(checkpointer=None):
    """Build M2 debate graph topology with static fan-out (D-12).

    Topology: START -> (real_source, fake_source) -> nei_gate -> (reranker -> social_loop? -> verify -> agreement_gate -> debate? -> judge) -> END
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

    g.add_edge(START, "real_source")  # D-12: static fan-out
    g.add_edge(START, "fake_source")
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
    g.add_edge("judge", END)
    return g.compile(checkpointer=checkpointer)


def initial_state(
    statement: str, image_path: str | None = None, language: str = "auto"
) -> FactCheckState:
    return FactCheckState(
        statement=statement,
        image_path=image_path,
        language=language,
        search_queries=[],
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
    )
