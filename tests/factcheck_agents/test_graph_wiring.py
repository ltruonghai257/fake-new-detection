import ast
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from factcheck_agents.graph import build_graph, route_after_verify

GRAPH_SRC = Path(__file__).resolve().parents[2] / "factcheck_agents" / "graph.py"


@pytest.mark.parametrize(
    "signal,expected",
    [
        (True, "social_search"),
        (False, "conclusion"),
        (None, "conclusion"),
    ],
)
def test_route_after_verify(signal, expected):
    state = {"reliability_signal": signal}
    assert route_after_verify(state) == expected


def test_build_graph_compiles():
    graph = build_graph(checkpointer=MemorySaver())
    assert graph is not None


# ── Source structure: A2A wiring (requirement A2A-05) ─────────────────────

FORBIDDEN_AGENT_IMPORTS = [
    "agents.fake_advocate",
    "agents.real_advocate",
    "agents.judge_agent",
    "agents.real_source_agent",
    "agents.fake_source_agent",
    "agents.social_loop_agent",
    "agents.search_agent",
    "agents.conclusion_agent",
]


def _graph_imports():
    """Return [(module_or_name, [imported_names or None]), ...] from graph.py."""
    tree = ast.parse(GRAPH_SRC.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, None))
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "", [a.name for a in node.names]))
    return imports


def test_graph_has_no_direct_agent_imports():
    imports = _graph_imports()
    for module, names in imports:
        for forbidden in FORBIDDEN_AGENT_IMPORTS:
            assert (
                forbidden not in module
            ), f"graph.py still imports A2A-wrapped agent directly: {forbidden!r}"
            for name in names or []:
                assert (
                    forbidden not in name
                ), f"graph.py still binds A2A-wrapped agent directly: {forbidden!r}"


def test_graph_agreement_gate_import_binds_only_route():
    imports = _graph_imports()
    agreement_imports = [
        names for module, names in imports if module.endswith("agents.agreement_gate")
    ]
    assert agreement_imports, "graph.py must import from agents.agreement_gate"
    assert all(names == ["route_after_agreement"] for names in agreement_imports)


def test_graph_imports_from_a2a_client():
    imports = _graph_imports()
    a2a_imports = [names for module, names in imports if module.endswith("a2a_client")]
    assert a2a_imports, "graph.py must import A2A wrappers from a2a_client"
