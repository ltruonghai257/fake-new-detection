"""Unit tests for social_loop_agent."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from factcheck_agents.agents.social_loop_agent import social_loop_agent
from factcheck_agents.graph import route_social_loop


@pytest.fixture(autouse=True)
def mock_fetch_image():
    """Autouse fixture to mock _fetch_evidence_image."""
    with patch(
        "factcheck_agents.agents.social_loop_agent._fetch_evidence_image",
        return_value=(None, None),
    ):
        yield


def test_social_loop_sets_fired_true():
    """Test that social_loop_agent sets social_loop_fired=True and returns evidence_social."""
    state = {
        "statement": "Test statement",
        "search_queries": ["query1"],
    }

    mock_result = {
        "title": "Test",
        "url": "https://example.com/test",
        "snippet": "Test snippet",
        "content": "Test content",
        "source": "test",
        "score": 0.9,
    }

    with patch(
        "factcheck_agents.agents.social_loop_agent.web_search", return_value=[mock_result]
    ):
        result = social_loop_agent(state)

    assert result["social_loop_fired"] is True
    assert "evidence_social" in result
    assert len(result["evidence_social"]) == 1
    assert result["evidence_social"][0]["url"] == "https://example.com/test"


def test_social_loop_targets_tiktok():
    """Test that social_loop_agent includes tiktok.com in include_domains."""
    state = {
        "statement": "Test statement",
        "search_queries": ["query1"],
    }

    with patch("factcheck_agents.agents.social_loop_agent.web_search") as mock_search:
        mock_search.return_value = []
        social_loop_agent(state)

        # Check that web_search was called with tiktok.com in include_domains
        call_kwargs = mock_search.call_args[1]
        assert "include_domains" in call_kwargs
        assert "tiktok.com" in call_kwargs["include_domains"]


def test_social_loop_fire_once_guard():
    """SOCLOOP-03: Test that route_social_loop returns 'verify' when social_loop_fired=True."""
    state = {
        "social_loop_fired": True,
        "evidence_real": [],
        "evidence_fake": [],
        "consistency_score": 0.1,
    }
    result = route_social_loop(state)
    assert result == "verify"


def test_social_loop_exception_returns_empty_fired():
    """Test that social_loop_agent returns empty evidence_social and sets fired=True on exception."""
    state = {
        "statement": "Test statement",
        "search_queries": ["query1"],
    }

    with patch(
        "factcheck_agents.agents.social_loop_agent.web_search",
        side_effect=Exception("fail"),
    ):
        result = social_loop_agent(state)

    assert result["evidence_social"] == []
    assert result["social_loop_fired"] is True
    assert len(result["errors"]) > 0


def test_social_loop_does_not_write_to_evidence_real_or_fake():
    """EVRET-03: Test that social_loop_agent does not write to evidence_real or evidence_fake."""
    state = {
        "statement": "Test statement",
        "search_queries": ["query1"],
    }

    mock_result = {
        "title": "Test",
        "url": "https://example.com/test",
        "snippet": "Test snippet",
        "content": "Test content",
        "source": "test",
        "score": 0.9,
    }

    with patch(
        "factcheck_agents.agents.social_loop_agent.web_search", return_value=[mock_result]
    ):
        result = social_loop_agent(state)

    # Should not have evidence_real or evidence_fake keys
    assert "evidence_real" not in result
    assert "evidence_fake" not in result
    # Should only have evidence_social
    assert "evidence_social" in result