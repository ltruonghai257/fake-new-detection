"""Web search over open truth sources.

Tavily is preferred (returns clean snippets tuned for LLM use); Google Custom
Search is used as a fallback. Both are optional — if neither is configured the
function returns an empty list and the pipeline continues on model output only.
"""

from __future__ import annotations

from typing import List

import requests

from ..config import settings
from ..state import Evidence


def _search_tavily(
    query: str, max_results: int, include_domains: list | None = None
) -> List[Evidence]:
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    out: List[Evidence] = []
    for r in data.get("results", []):
        out.append(
            Evidence(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                source="tavily",
                score=float(r.get("score", 0.0) or 0.0),
            )
        )
    return out


def _search_google_cse(
    query: str, max_results: int, include_domains: list | None = None
) -> List[Evidence]:
    effective_query = query
    if include_domains:
        site_filter = " OR ".join(f"site:{d}" for d in include_domains)
        effective_query = f"{query} {site_filter}"
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": settings.google_cse_api_key,
                "cx": settings.google_cse_id,
                "q": effective_query,
                "num": min(max_results, 10),
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    out: List[Evidence] = []
    for r in data.get("items", []):
        out.append(
            Evidence(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
                source="google_cse",
                score=0.0,
            )
        )
    return out


def web_search(
    query: str, max_results: int | None = None, include_domains: list | None = None
) -> List[Evidence]:
    """Return evidence for a single query using the first configured provider."""
    n = max_results or settings.max_results
    if settings.tavily_api_key:
        results = _search_tavily(query, n, include_domains=include_domains)
        if results:
            return results
    if settings.google_cse_api_key and settings.google_cse_id:
        return _search_google_cse(query, n, include_domains=include_domains)
    return []
