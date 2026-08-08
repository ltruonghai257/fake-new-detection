"""Real source agent: gather evidence from trusted Vietnamese news domains.

Searches a hardcoded list of trusted domains (vnexpress.net, tuoitre.vn, thanhnien.vn,
ttxvn.gov.vn, vtv.vn, dantri.com.vn) for evidence supporting the claim.
"""

from __future__ import annotations

from typing import List

from ..config import settings
from ..helpers import _fetch_evidence_image
from ..source_tier import classify_domain
from ..state import Evidence, FactCheckState
from ..tools.web_search import web_search
from ..tools.article_crawler import crawl_article, is_within_days

REAL_DOMAINS = [
    "vnexpress.net",
    "tuoitre.vn",
    "thanhnien.vn",
    "ttxvn.gov.vn",
    "vtv.vn",
    "dantri.com.vn",
]


def real_source_agent(state: FactCheckState) -> dict:
    """Search trusted Vietnamese news domains for real evidence.

    Crawls full article content and filters to last 7 days.
    Returns {"evidence_real": results, "messages": [...], "evidence_workflow_steps": [...]}.
    Never raises; returns empty list on all failures (EVRET-03).
    """
    queries = state.get("search_queries") or [state["statement"]]
    claim_variants = state.get("claim_variants") or []
    queries = list(dict.fromkeys(queries + claim_variants))  # deduplicate, preserve order
    results: List[Evidence] = []
    seen: set = set()
    workflow_steps = []

    for q in queries:
        try:
            workflow_steps.append(
                {
                    "step": f"Search query: {q[:50]}...",
                    "description": f"Search Tavily/Google CSE with include_domains={REAL_DOMAINS[:2]}",
                    "input": f"max_results={settings.max_results}",
                    "output": "",
                }
            )

            q_results = web_search(
                q, max_results=settings.max_results, include_domains=REAL_DOMAINS
            )
            workflow_steps[-1]["output"] = f"{len(q_results)} results found"

            crawled_count = 0
            filtered_count = 0
            for e in q_results:
                url = e.get("url", "")
                if url and url in seen:
                    continue
                if url:
                    seen.add(url)

                # Crawl full article content
                full_content, publish_date = crawl_article(url)
                crawled_count += 1

                # Filter: only include if within last 7 days
                if not is_within_days(publish_date, days=7):
                    filtered_count += 1
                    continue

                e["source_tier"] = classify_domain(url) if url else "unknown"
                e["image_path"], e["image_caption"] = (
                    _fetch_evidence_image(url) if url else (None, None)
                )

                # Use full content if available, otherwise snippet
                if full_content:
                    e["content"] = full_content
                    e["snippet"] = (
                        full_content[:500] + "..."
                        if len(full_content) > 500
                        else full_content
                    )
                    e["publish_date"] = (
                        publish_date.isoformat() if publish_date else None
                    )

                results.append(e)

            workflow_steps.append(
                {
                    "step": f"Crawl & filter for query: {q[:50]}...",
                    "description": "Crawl full article content + filter by last 7 days",
                    "input": f"{crawled_count} URLs crawled",
                    "output": f"{len(results) - (len(results) - crawled_count + filtered_count)} passed 7-day filter (filtered: {filtered_count})",
                }
            )
        except Exception as exc:
            errors = state.get("errors", [])
            errors.append(f"[RealSource] Query '{q}' failed: {exc}")

    msg = f"[RealSource] {len(results)} items from {REAL_DOMAINS[:2]}... (last 7 days)"
    return {
        "evidence_real": results,
        "messages": [("assistant", msg)],
        "evidence_workflow_steps": workflow_steps,
    }
