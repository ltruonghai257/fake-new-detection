"""LangGraph wiring: Search -> Verify -> (Social?) -> Conclusion.

A single ``FactCheckState`` is threaded through the agent nodes, mirroring
the shared-state design used by TradingAgents.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import (
    conclusion_agent,
    search_agent,
    social_search_agent,
    verify_agent,
)
from .state import FactCheckState


def route_after_verify(state: FactCheckState) -> str:
    """Route after verify based on reliability_signal."""
    if state.get("reliability_signal"):
        return "social_search"
    return "conclusion"


def build_graph():
    """Compile and return the fact-checking graph."""
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
    return g.compile()


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
    )
