"""LangGraph wiring: Search -> Evaluate -> Conclusion.

A single ``FactCheckState`` is threaded through the three agent nodes, mirroring
the shared-state design used by TradingAgents.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import conclusion_agent, evaluate_agent, search_agent
from .state import FactCheckState


def build_graph():
    """Compile and return the fact-checking graph."""
    g = StateGraph(FactCheckState)
    g.add_node("search", search_agent)
    g.add_node("evaluate", evaluate_agent)
    g.add_node("conclusion", conclusion_agent)

    g.add_edge(START, "search")
    g.add_edge("search", "evaluate")
    g.add_edge("evaluate", "conclusion")
    g.add_edge("conclusion", END)
    return g.compile()


def initial_state(statement: str, image_path: str | None = None, language: str = "auto") -> FactCheckState:
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
