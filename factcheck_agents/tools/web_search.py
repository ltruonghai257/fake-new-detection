"""Web search over open truth sources.

Tavily is preferred (returns clean snippets tuned for LLM use); Google Custom
Search is used as a fallback. Both are optional — if neither is configured the
function returns an empty list and the pipeline continues on model output only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import List

import requests

from ..config import settings
from ..state import Evidence


def _is_recent(published: str | None) -> bool:
    """False only when a parseable date is older than the window.

    Results without a date are kept: the provider-side filter already
    restricted them, and many pages simply expose no date.
    """
    days = settings.search_max_age_days
    if days <= 0 or not published:
        return True
    dt = None
    try:
        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(published)
        except (TypeError, ValueError):
            return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt >= datetime.now(timezone.utc) - timedelta(days=days)


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
    days = settings.search_max_age_days
    if days > 0:
        payload["time_range"] = (
            "day" if days <= 1 else "week" if days <= 7 else "month" if days <= 31 else "year"
        )
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
        if not _is_recent(r.get("published_date")):
            continue
        out.append(
            Evidence(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),  # Tavily's content → snippet
                source="tavily",
                score=float(r.get("score", 0.0) or 0.0),
                content=r.get("content", ""),  # Keep original content field
                published_date=r.get("published_date"),
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
    params = {
        "key": settings.google_cse_api_key,
        "cx": settings.google_cse_id,
        "q": effective_query,
        "num": min(max_results, 10),
    }
    if settings.search_max_age_days > 0:
        params["dateRestrict"] = f"d{settings.search_max_age_days}"
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
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
                content=r.get("snippet", ""),  # Google uses snippet for both
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
