"""Unit tests for the BM25+PhoBERT reranker.

Tests recall@k, graceful degradation, and edge cases (RERANK-02, D-01 through D-07).
"""

import pytest

from factcheck_agents.reranker import reranker


def test_reranker_bm25_fallback_recall():
    """BM25-only fallback: relevant snippet appears in selected evidence (RERANK-02).

    Target recall@k ≥ 1/5 on BM25-only; BM25 advantage on keyword-matched Vietnamese pairs.
    """
    # Five labeled claim–snippet pairs (keyword-matched for BM25)
    state = {
        "statement": "Chủ tịch nước Nguyễn Xuân Phúc",
        "evidence_real": [
            {
                "snippet": "Chủ tịch nước Nguyễn Xuân Phúc đã đến thăm tỉnh Bình Dương",
                "url": "http://example.com/1",
            },
            {
                "snippet": "Thủ tướng Phạm Minh Chính làm việc tại Hà Nội",
                "url": "http://example.com/2",
            },
            {
                "snippet": "Báo cáo kinh tế quý 3 năm 2023",
                "url": "http://example.com/3",
            },
            {
                "snippet": "Giá vàng tăng mạnh trong tuần qua",
                "url": "http://example.com/4",
            },
            {
                "snippet": "Đội tuyển Việt Nam thắng Thái Lan",
                "url": "http://example.com/5",
            },
        ],
        "evidence_fake": [],
        "evidence_social": [],
    }

    result = reranker(state)
    assert "evidence" in result
    assert len(result["evidence"]) > 0

    # The first snippet is keyword-matched to the claim (contains "Chủ tịch nước Nguyễn Xuân Phúc")
    # It should appear in the selected evidence
    selected_urls = [e.get("url") for e in result["evidence"]]
    assert "http://example.com/1" in selected_urls


def test_consistency_score_floor_when_phobert_unavailable():
    """When PhoBERT is unavailable, consistency_score should be 0.1 (D-07)."""
    state = {
        "statement": "Test claim",
        "evidence_real": [{"snippet": "Test evidence", "url": "http://example.com/1"}],
        "evidence_fake": [],
        "evidence_social": [],
    }

    result = reranker(state)
    assert result["consistency_score"] == 0.1


def test_reranker_empty_pool_returns_empty_dict():
    """Empty evidence pool should return empty dict (D-05)."""
    state = {
        "statement": "Test claim",
        "evidence_real": [],
        "evidence_fake": [],
        "evidence_social": [],
    }

    result = reranker(state)
    assert result == {}


def test_reranker_single_item_no_division_by_zero():
    """Single-item pool should not cause division by zero."""
    state = {
        "statement": "Test claim",
        "evidence_real": [{"snippet": "Test evidence", "url": "http://example.com/1"}],
        "evidence_fake": [],
        "evidence_social": [],
    }

    result = reranker(state)
    assert "evidence" in result
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["url"] == "http://example.com/1"


def test_reranker_writes_to_evidence_field():
    """Result should write to 'evidence' field, not 'evidence_real' or 'evidence_fake' (D-05)."""
    state = {
        "statement": "Test claim",
        "evidence_real": [{"snippet": "Test evidence", "url": "http://example.com/1"}],
        "evidence_fake": [],
        "evidence_social": [],
    }

    result = reranker(state)
    assert "evidence" in result
    assert "evidence_real" not in result
    assert "evidence_fake" not in result
    assert "evidence_social" not in result
