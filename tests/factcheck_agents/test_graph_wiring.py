import pytest
from langgraph.checkpoint.memory import MemorySaver

from factcheck_agents.graph import build_graph, route_after_verify


@pytest.mark.parametrize("signal,expected", [
    (True, "social_search"),
    (False, "conclusion"),
    (None, "conclusion"),
])
def test_route_after_verify(signal, expected):
    state = {"reliability_signal": signal}
    assert route_after_verify(state) == expected


def test_build_graph_compiles():
    graph = build_graph(checkpointer=MemorySaver())
    assert graph is not None
