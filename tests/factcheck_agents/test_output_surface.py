from unittest.mock import MagicMock, patch

import pytest

from factcheck_agents.cli import _print_human
from factcheck_agents import run_fact_check


def test_print_human_shows_verdict_label_vi(capsys):
    _print_human({"verdict": {"label": "TRUE", "verdict_label_vi": "Thật", "confidence": 0.9}})
    captured = capsys.readouterr()
    assert "Thật" in captured.out


def test_print_human_vi_label_includes_4class_parenthetical(capsys):
    _print_human({"verdict": {"label": "TRUE", "verdict_label_vi": "Thật", "confidence": 0.9}})
    captured = capsys.readouterr()
    assert "TRUE" in captured.out


def test_print_human_fallback_no_vi_label(capsys):
    _print_human({"verdict": {"label": "UNVERIFIED", "confidence": 0.0}})
    captured = capsys.readouterr()
    assert "UNVERIFIED" in captured.out
    assert "Thật" not in captured.out
    assert "Giả" not in captured.out


@patch("factcheck_agents.graph.build_graph")
def test_run_fact_check_promotes_verdict_binary(mock_build):
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "verdict": {"verdict_binary": "REAL", "verdict_label_vi": "Thật", "label": "TRUE"},
        "model_results": [],
        "evidence": [],
        "search_queries": [],
        "messages": [],
        "errors": [],
        "meta": {},
    }
    mock_build.return_value = mock_graph
    result = run_fact_check("Tuyên bố kiểm tra")
    assert result["verdict_binary"] == "REAL"
    assert result["verdict_label_vi"] == "Thật"


@patch("factcheck_agents.graph.build_graph")
def test_run_fact_check_verdict_dict_still_present(mock_build):
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "verdict": {"verdict_binary": "FAKE", "verdict_label_vi": "Giả", "label": "FALSE"},
        "model_results": [],
        "evidence": [],
        "search_queries": [],
        "messages": [],
        "errors": [],
        "meta": {},
    }
    mock_build.return_value = mock_graph
    result = run_fact_check("Tuyên bố kiểm tra")
    assert "verdict" in result
    assert isinstance(result["verdict"], dict)


@patch("factcheck_agents.mcp_server.build_graph")
def test_mcp_fact_check_includes_verdict_binary(mock_build):
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "verdict": {"verdict_binary": "FAKE", "verdict_label_vi": "Giả"},
        "model_results": [],
        "evidence": [],
        "search_queries": [],
    }
    mock_build.return_value = mock_graph
    from factcheck_agents.mcp_server import fact_check
    result = fact_check("Tuyên bố kiểm tra")
    assert "verdict_binary" in result
    assert "verdict_label_vi" in result


@patch("factcheck_agents.mcp_server.build_graph")
def test_mcp_fact_check_preserves_existing_keys(mock_build):
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "verdict": {"verdict_binary": "REAL", "verdict_label_vi": "Thật"},
        "model_results": [],
        "evidence": [],
        "search_queries": [],
    }
    mock_build.return_value = mock_graph
    from factcheck_agents.mcp_server import fact_check
    result = fact_check("Tuyên bố kiểm tra")
    for key in ("statement", "verdict", "model_results", "evidence", "search_queries"):
        assert key in result, f"existing key '{key}' missing from mcp response"
