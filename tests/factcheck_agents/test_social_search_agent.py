from unittest.mock import MagicMock, patch

import pytest

from factcheck_agents.agents.social_search_agent import social_search_agent
from factcheck_agents.graph_utils import EvidenceGraph
from factcheck_agents.state import Evidence


def _make_evidence(url: str, snippet: str = "s") -> Evidence:
    return Evidence(title="T", url=url, snippet=snippet, source="tavily", score=0.5)


def _make_graph_with_url(url: str) -> EvidenceGraph:
    eg = EvidenceGraph()
    eg.graph.add_node("statement", node_type="statement")
    eg.add_node(url, {"node_type": "evidence", "source_tier": "trusted"})
    eg.add_edge("statement", url, type="mentions")
    return eg


def _make_state(queries=None, evidence_graph=None):
    state = {"statement": "test"}
    if queries is not None:
        state["search_queries"] = queries
    if evidence_graph is not None:
        state["evidence_graph"] = evidence_graph
    return state


# ── source_tier tagging ─────────────────────────────────────────────────────


@patch(
    "factcheck_agents.agents.social_search_agent._fetch_evidence_image",
    return_value=(None, None),
)
@patch("factcheck_agents.agents.social_search_agent.web_search")
def test_social_items_tagged_social(mock_search, mock_img):
    mock_search.return_value = [_make_evidence("https://twitter.com/post/1")]
    eg = EvidenceGraph.build_from_evidence([])
    result = social_search_agent(_make_state(queries=["query"], evidence_graph=eg))
    node = result["evidence_graph"].graph.nodes.get("https://twitter.com/post/1")
    assert node is not None
    assert node["source_tier"] == "social"


# ── graph merge ──────────────────────────────────────────────────────────────


@patch(
    "factcheck_agents.agents.social_search_agent._fetch_evidence_image",
    return_value=(None, None),
)
@patch("factcheck_agents.agents.social_search_agent.web_search")
def test_merges_into_existing_graph(mock_search, mock_img):
    existing_url = "https://vnexpress.net/existing"
    eg = _make_graph_with_url(existing_url)
    mock_search.return_value = [_make_evidence("https://twitter.com/post/2")]
    result = social_search_agent(_make_state(queries=["q"], evidence_graph=eg))
    nodes = result["evidence_graph"].graph.nodes
    assert existing_url in nodes
    assert "https://twitter.com/post/2" in nodes


# ── dedup ────────────────────────────────────────────────────────────────────


@patch(
    "factcheck_agents.agents.social_search_agent._fetch_evidence_image",
    return_value=(None, None),
)
@patch("factcheck_agents.agents.social_search_agent.web_search")
def test_dedup_skips_existing_urls(mock_search, mock_img):
    dup_url = "https://twitter.com/already/here"
    eg = _make_graph_with_url(dup_url)
    initial_count = eg.graph.number_of_nodes()
    mock_search.return_value = [_make_evidence(dup_url)]
    result = social_search_agent(_make_state(queries=["q"], evidence_graph=eg))
    assert result["evidence_graph"].graph.number_of_nodes() == initial_count


# ── web_search call parameters ───────────────────────────────────────────────


@patch(
    "factcheck_agents.agents.social_search_agent._fetch_evidence_image",
    return_value=(None, None),
)
@patch("factcheck_agents.agents.social_search_agent.web_search", return_value=[])
def test_max_results_3_per_query(mock_search, mock_img):
    eg = EvidenceGraph.build_from_evidence([])
    social_search_agent(_make_state(queries=["q1"], evidence_graph=eg))
    _, kwargs = mock_search.call_args
    assert kwargs.get("max_results") == 3 or mock_search.call_args[0][1] == 3


@patch(
    "factcheck_agents.agents.social_search_agent._fetch_evidence_image",
    return_value=(None, None),
)
@patch("factcheck_agents.agents.social_search_agent.web_search", return_value=[])
def test_include_domains_passed(mock_search, mock_img):
    eg = EvidenceGraph.build_from_evidence([])
    social_search_agent(_make_state(queries=["q1"], evidence_graph=eg))
    _, kwargs = mock_search.call_args
    domains = kwargs.get("include_domains") or mock_search.call_args[0][2]
    assert "twitter.com" in domains
    assert "facebook.com" in domains


# ── no-results graceful degrade ──────────────────────────────────────────────


@patch(
    "factcheck_agents.agents.social_search_agent._fetch_evidence_image",
    return_value=(None, None),
)
@patch("factcheck_agents.agents.social_search_agent.web_search", return_value=[])
def test_no_results_returns_unchanged_graph(mock_search, mock_img):
    eg = EvidenceGraph.build_from_evidence(
        [{"url": "https://vnexpress.net/a", "snippet": "s", "source_tier": "trusted"}]
    )
    before = eg.graph.number_of_nodes()
    result = social_search_agent(_make_state(queries=["q"], evidence_graph=eg))
    assert result["evidence_graph"].graph.number_of_nodes() == before


# ── return contract ──────────────────────────────────────────────────────────


@patch(
    "factcheck_agents.agents.social_search_agent._fetch_evidence_image",
    return_value=(None, None),
)
@patch("factcheck_agents.agents.social_search_agent.web_search", return_value=[])
def test_returns_evidence_graph_and_messages(mock_search, mock_img):
    eg = EvidenceGraph.build_from_evidence([])
    result = social_search_agent(_make_state(queries=["q"], evidence_graph=eg))
    assert "evidence_graph" in result
    assert "messages" in result
    assert isinstance(result["evidence_graph"], EvidenceGraph)


@patch(
    "factcheck_agents.agents.social_search_agent._fetch_evidence_image",
    return_value=(None, None),
)
@patch("factcheck_agents.agents.social_search_agent.web_search", return_value=[])
def test_no_flat_evidence_written(mock_search, mock_img):
    eg = EvidenceGraph.build_from_evidence([])
    result = social_search_agent(_make_state(queries=["q"], evidence_graph=eg))
    assert "evidence" not in result
