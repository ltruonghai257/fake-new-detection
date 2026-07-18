import pytest
from factcheck_agents.graph_utils import EvidenceGraph


def test_empty_evidence_creates_statement_node():
    eg = EvidenceGraph.build_from_evidence([])
    assert "statement" in eg.graph.nodes
    assert eg.graph.nodes["statement"]["node_type"] == "statement"


def test_single_evidence_creates_two_nodes():
    ev = [{"url": "https://vnexpress.net/a", "snippet": "test", "source_tier": "trusted"}]
    eg = EvidenceGraph.build_from_evidence(ev)
    assert len(eg.graph.nodes) == 2


def test_evidence_node_attributes():
    ev = [{"url": "https://vnexpress.net/a", "title": "Title", "snippet": "Snip", "source_tier": "trusted"}]
    eg = EvidenceGraph.build_from_evidence(ev)
    node = eg.graph.nodes["https://vnexpress.net/a"]
    assert node["source_tier"] == "trusted"
    assert node["title"] == "Title"
    assert node["node_type"] == "evidence"


def test_default_edge_type_is_mentions():
    ev = [{"url": "https://vnexpress.net/a", "snippet": "s"}]
    eg = EvidenceGraph.build_from_evidence(ev)
    edges = list(eg.graph.edges(data=True))
    assert any(d.get("type") == "mentions" for _, _, d in edges)


def test_add_edge_with_custom_type():
    eg = EvidenceGraph()
    eg.add_node("a", {"node_type": "evidence"})
    eg.add_node("b", {"node_type": "evidence"})
    eg.add_edge("a", "b", type="supports")
    edge_data = eg.graph.edges["a", "b"]
    assert edge_data["type"] == "supports"


def test_to_evidence_list_returns_evidence_nodes():
    ev = [
        {"url": "https://vnexpress.net/a", "title": "T1", "snippet": "S1", "source_tier": "trusted"},
        {"url": "https://example.com/b", "title": "T2", "snippet": "S2", "source_tier": "unknown"},
    ]
    eg = EvidenceGraph.build_from_evidence(ev)
    result = eg.to_evidence_list()
    assert len(result) == 2
    urls = {item["url"] for item in result}
    assert "https://vnexpress.net/a" in urls
    assert "https://example.com/b" in urls
