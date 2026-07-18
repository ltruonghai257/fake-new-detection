from unittest.mock import patch, MagicMock
import pytest
from factcheck_agents.agents.search_agent import search_agent
from factcheck_agents.agents.search_agent import _fetch_evidence_image
from factcheck_agents.graph_utils import EvidenceGraph


def _make_state(statement="test statement"):
    return {"statement": statement}


def _mock_evidence(url, score=0.5):
    return {
        "title": "T",
        "url": url,
        "snippet": "S",
        "source": "tavily",
        "score": score,
    }


@pytest.fixture(autouse=True)
def _stub_fetch_evidence_image():
    """Prevent search_agent tests from making real HTTP requests for images."""
    with patch(
        "factcheck_agents.agents.search_agent._fetch_evidence_image",
        return_value=(None, None),
    ) as p:
        yield p


@patch("factcheck_agents.agents.search_agent.web_search")
@patch("factcheck_agents.agents.search_agent._draft_queries", return_value=["q1"])
def test_three_passes_calls_web_search_three_times(mock_draft, mock_ws):
    mock_ws.return_value = []
    search_agent(_make_state())
    assert mock_ws.call_count == 3  # 1 query x 3 passes


@patch("factcheck_agents.agents.search_agent.web_search")
@patch("factcheck_agents.agents.search_agent._draft_queries", return_value=["q1"])
def test_source_tier_tagged_by_classify_domain(mock_draft, mock_ws):
    mock_ws.return_value = [_mock_evidence("https://vnexpress.net/article")]
    result = search_agent(_make_state())
    # First pass returns trusted URL; dedup means only one item total
    tiers = {e["source_tier"] for e in result["evidence"]}
    assert "trusted" in tiers


@patch("factcheck_agents.agents.search_agent.web_search")
@patch("factcheck_agents.agents.search_agent._draft_queries", return_value=["q1"])
def test_dedup_first_occurrence_wins(mock_draft, mock_ws):
    url = "https://example.com/page"
    # All 3 passes return same URL
    mock_ws.return_value = [_mock_evidence(url)]
    result = search_agent(_make_state())
    urls = [e["url"] for e in result["evidence"]]
    assert urls.count(url) == 1


@patch("factcheck_agents.agents.search_agent.web_search")
@patch("factcheck_agents.agents.search_agent._draft_queries", return_value=["q1"])
def test_evidence_capped_at_two_times_max_results(mock_draft, mock_ws):
    from factcheck_agents.config import settings

    # Return many unique URLs
    mock_ws.side_effect = [
        [_mock_evidence(f"https://a{i}.com/") for i in range(20)],
        [_mock_evidence(f"https://b{i}.com/") for i in range(20)],
        [_mock_evidence(f"https://c{i}.com/") for i in range(20)],
    ]
    result = search_agent(_make_state())
    assert len(result["evidence"]) <= 2 * settings.max_results


@patch("factcheck_agents.agents.search_agent.web_search")
@patch("factcheck_agents.agents.search_agent._draft_queries", return_value=["q1"])
def test_evidence_sorted_by_tier_priority_then_score(mock_draft, mock_ws):
    mock_ws.side_effect = [
        [
            _mock_evidence("https://unknown1.com/", score=0.9)
        ],  # pass1 trusted filter -> classify -> unknown
        [
            _mock_evidence("https://vnexpress.net/", score=0.1)
        ],  # pass2 flagged filter -> classify -> trusted
        [],
    ]
    result = search_agent(_make_state())
    tiers = [e["source_tier"] for e in result["evidence"]]
    # trusted should appear before unknown
    assert tiers.index("trusted") < tiers.index("unknown")


@patch("factcheck_agents.agents.search_agent.web_search")
@patch("factcheck_agents.agents.search_agent._draft_queries", return_value=["q1"])
def test_evidence_graph_populated(mock_draft, mock_ws):
    mock_ws.return_value = [_mock_evidence("https://vnexpress.net/a")]
    result = search_agent(_make_state())
    assert result["evidence_graph"] is not None


@patch("factcheck_agents.agents.search_agent.web_search")
@patch("factcheck_agents.agents.search_agent._draft_queries", return_value=["q1"])
def test_evidence_graph_is_evidence_graph_instance(mock_draft, mock_ws):
    mock_ws.return_value = [_mock_evidence("https://vnexpress.net/a")]
    result = search_agent(_make_state())
    assert isinstance(result["evidence_graph"], EvidenceGraph)


@patch("factcheck_agents.agents.search_agent.web_search")
@patch("factcheck_agents.agents.search_agent._draft_queries", return_value=["q1"])
def test_backward_compat_evidence_key_present(mock_draft, mock_ws):
    mock_ws.return_value = []
    result = search_agent(_make_state())
    assert "evidence" in result
    assert isinstance(result["evidence"], list)


@patch("factcheck_agents.agents.search_agent.web_search")
@patch("factcheck_agents.agents.search_agent._draft_queries", return_value=["q1"])
@patch(
    "factcheck_agents.agents.search_agent._fetch_evidence_image",
    return_value=("/tmp/evidence.jpg", "A caption"),
)
def test_evidence_image_path_and_caption_set(mock_fetch, mock_draft, mock_ws):
    mock_ws.return_value = [_mock_evidence("https://vnexpress.net/article")]
    result = search_agent(_make_state())
    assert result["evidence"][0]["image_path"] == "/tmp/evidence.jpg"
    assert result["evidence"][0]["image_caption"] == "A caption"


def test_fetch_evidence_image_downloads_og_image(tmp_path):
    html_resp = MagicMock()
    html_resp.raise_for_status.return_value = None
    html_resp.text = (
        '<html><meta property="og:image" content="http://example.com/img.jpg">'
        '<meta property="og:image:alt" content="Sample caption"></html>'
    )
    img_resp = MagicMock()
    img_resp.raise_for_status.return_value = None
    img_resp.content = b"fake-image-bytes"

    with patch("factcheck_agents.agents.search_agent.settings.data_root", tmp_path):
        with patch("factcheck_agents.agents.search_agent.requests.get") as mock_get:
            mock_get.side_effect = [html_resp, img_resp]
            path, caption = _fetch_evidence_image("http://example.com/article")

    assert path is not None
    assert path.endswith(".jpg")
    assert caption is not None


def test_fetch_evidence_image_returns_none_on_failure(tmp_path):
    with (
        patch("factcheck_agents.agents.search_agent.settings.data_root", tmp_path),
        patch(
            "factcheck_agents.agents.search_agent.requests.get",
            side_effect=Exception("network"),
        ),
    ):
        assert _fetch_evidence_image("http://example.com/article") == (None, None)
