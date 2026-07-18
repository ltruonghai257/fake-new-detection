from unittest.mock import patch, MagicMock
import pytest


@patch("factcheck_agents.tools.web_search.requests.post")
def test_tavily_include_domains_in_payload_when_non_empty(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": []}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    from factcheck_agents.tools.web_search import _search_tavily
    from factcheck_agents.config import settings

    original = settings.tavily_api_key
    settings.tavily_api_key = "test-key"
    try:
        _search_tavily("query", 5, include_domains=["vnexpress.net"])
    finally:
        settings.tavily_api_key = original

    call_kwargs = mock_post.call_args
    payload = call_kwargs[1]["json"] if call_kwargs[1] else call_kwargs[0][1]
    assert "include_domains" in payload
    assert payload["include_domains"] == ["vnexpress.net"]


@patch("factcheck_agents.tools.web_search.requests.post")
def test_tavily_no_include_domains_when_none(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": []}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    from factcheck_agents.tools.web_search import _search_tavily
    from factcheck_agents.config import settings

    original = settings.tavily_api_key
    settings.tavily_api_key = "test-key"
    try:
        _search_tavily("query", 5, include_domains=None)
    finally:
        settings.tavily_api_key = original

    call_kwargs = mock_post.call_args
    payload = call_kwargs[1]["json"] if call_kwargs[1] else call_kwargs[0][1]
    assert "include_domains" not in payload


@patch("factcheck_agents.tools.web_search.requests.get")
def test_google_cse_site_filter_appended_to_query(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"items": []}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    from factcheck_agents.tools.web_search import _search_google_cse
    from factcheck_agents.config import settings

    orig_key = settings.google_cse_api_key
    orig_id = settings.google_cse_id
    settings.google_cse_api_key = "test-key"
    settings.google_cse_id = "test-cx"
    try:
        _search_google_cse(
            "my query", 5, include_domains=["vnexpress.net", "tuoitre.vn"]
        )
    finally:
        settings.google_cse_api_key = orig_key
        settings.google_cse_id = orig_id

    call_kwargs = mock_get.call_args
    params = call_kwargs[1]["params"] if call_kwargs[1] else call_kwargs[0][1]
    assert "site:vnexpress.net" in params["q"]
    assert "site:tuoitre.vn" in params["q"]
    assert "my query" in params["q"]


@patch("factcheck_agents.tools.web_search.requests.get")
def test_google_cse_no_site_filter_when_none(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"items": []}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    from factcheck_agents.tools.web_search import _search_google_cse
    from factcheck_agents.config import settings

    orig_key = settings.google_cse_api_key
    orig_id = settings.google_cse_id
    settings.google_cse_api_key = "test-key"
    settings.google_cse_id = "test-cx"
    try:
        _search_google_cse("my query", 5, include_domains=None)
    finally:
        settings.google_cse_api_key = orig_key
        settings.google_cse_id = orig_id

    call_kwargs = mock_get.call_args
    params = call_kwargs[1]["params"] if call_kwargs[1] else call_kwargs[0][1]
    assert params["q"] == "my query"


def test_search_query_prompt_contains_vietnamese():
    from factcheck_agents.prompts import SEARCH_QUERY_PROMPT

    assert "Vietnamese" in SEARCH_QUERY_PROMPT
