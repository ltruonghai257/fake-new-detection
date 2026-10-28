"""Parallel search + crawl shared by real_source_agent and fake_source_agent.

Everything network-bound (web search, article crawl, image fetch) used to run
serially per query and per result; here each phase runs in a thread pool.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

from ..config import settings
from ..helpers import _fetch_evidence_image
from ..source_tier import classify_domain
from ..state import Evidence
from ..tools.article_crawler import crawl_article, is_within_days
from ..tools.web_search import web_search

_WORKERS = int(os.getenv("FACTCHECK_SEARCH_WORKERS", "8"))
_MAX_URLS = int(os.getenv("FACTCHECK_MAX_CRAWL_URLS", "12"))


def _safe_search(q: str, domains: Optional[list]) -> List[Evidence]:
    try:
        return web_search(q, max_results=settings.max_results, include_domains=domains)
    except Exception:
        return []


def _enrich(e: Evidence) -> Optional[Evidence]:
    url = e.get("url", "")
    full_content, publish_date = crawl_article(url)
    if not is_within_days(publish_date, days=settings.search_max_age_days or 7):
        return None
    e["source_tier"] = classify_domain(url) if url else "unknown"
    if full_content:
        e["content"] = full_content
        e["snippet"] = full_content[:500] + "..." if len(full_content) > 500 else full_content
        e["publish_date"] = publish_date.isoformat() if publish_date else None
    e["image_path"], e["image_caption"] = (
        _fetch_evidence_image(url) if url else (None, None)
    )
    return e


def parallel_search_and_crawl(
    queries: List[str],
    domain_sets: List[Optional[list]],
    seen: set,
    keep: Callable[[Evidence], bool] = lambda e: True,
) -> List[Evidence]:
    """Search every (query, domain set) pair concurrently, then crawl unique
    URLs concurrently. `keep` pre-filters results before the costly crawl."""
    tasks = [(q, d) for q in queries for d in domain_sets]
    if not tasks:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(_WORKERS, len(tasks)))) as ex:
        batches = list(ex.map(lambda t: _safe_search(*t), tasks))

    unique: List[Evidence] = []
    for batch in batches:
        for e in batch:
            url = e.get("url", "")
            if url and url in seen:
                continue
            if url:
                seen.add(url)
            if keep(e):
                unique.append(e)
    unique = unique[:_MAX_URLS]
    if not unique:
        return []

    with ThreadPoolExecutor(max_workers=max(1, min(_WORKERS, len(unique)))) as ex:
        enriched = list(ex.map(_enrich_safe, unique))
    return [e for e in enriched if e is not None]


def _enrich_safe(e: Evidence) -> Optional[Evidence]:
    try:
        return _enrich(e)
    except Exception:
        return None
