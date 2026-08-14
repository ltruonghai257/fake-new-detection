"""Fake source agent: gather evidence from non-official / flagged sources.

Searches flagged domains (kenh14.vn, etc.), open web, and Google Fact Check
API for evidence refuting or casting doubt on the claim.
"""

from __future__ import annotations

import requests
from typing import List
from dateutil import parser as date_parser

from ..config import settings
from ..helpers import _fetch_evidence_image
from ..source_tier import classify_domain
from ..state import Evidence, FactCheckState
from ..tools.web_search import web_search
from ..tools.article_crawler import crawl_article, is_within_days


def _search_and_crawl(
    query: str, include_domains: list | None, seen: set
) -> List[Evidence]:
    """Run web search, crawl article content, filter by 7 days, deduplicate."""
    try:
        q_results = web_search(
            query, max_results=settings.max_results, include_domains=include_domains
        )
    except Exception:
        return []

    out: List[Evidence] = []
    for e in q_results:
        url = e.get("url", "")
        if url and url in seen:
            continue
        if url:
            seen.add(url)

        full_content, publish_date = crawl_article(url)
        if not is_within_days(publish_date, days=7):
            continue

        e["source_tier"] = classify_domain(url) if url else "unknown"
        e["image_path"], e["image_caption"] = (
            _fetch_evidence_image(url) if url else (None, None)
        )
        if full_content:
            e["content"] = full_content
            e["snippet"] = (
                full_content[:500] + "..." if len(full_content) > 500 else full_content
            )
            e["publish_date"] = publish_date.isoformat() if publish_date else None

        out.append(e)
    return out


def fake_source_agent(state: FactCheckState) -> dict:
    """Search flagged/non-official sources for evidence against the claim.

    Returns {"evidence_fake": results, "messages": [...]}.
    Never raises; returns empty list on all failures (EVRET-03).
    """
    queries = state.get("search_queries") or [state["statement"]]
    claim_variants = state.get("claim_variants") or []
    queries = list(
        dict.fromkeys(queries + claim_variants)
    )  # deduplicate, preserve order

    flagged_list = [d.strip() for d in settings.flagged_domains.split(",") if d.strip()]
    results: List[Evidence] = []
    seen: set = set()

    for q in queries:
        # 1) flagged domains (kenh14.vn, etc.)
        if flagged_list:
            results.extend(_search_and_crawl(q, flagged_list, seen))

        # 2) open web (no domain filter) — catches non-official sources
        results.extend(_search_and_crawl(q, None, seen))

        # 3) Google Fact Check API
        if settings.google_factcheck_api_key is not None:
            try:
                api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
                params = {
                    "query": q,
                    "languageCode": "vi",
                    "key": settings.google_factcheck_api_key,
                }
                resp = requests.get(api_url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                claims = data.get("claims", [])
                for claim in claims:
                    claim_text = claim.get("text", "")
                    claimant = claim.get("claimant", "")
                    review_date = claim.get("claimReview", [{}])[0].get(
                        "reviewDate", ""
                    )
                    publisher = (
                        claim.get("claimReview", [{}])[0]
                        .get("publisher", {})
                        .get("name", "")
                    )
                    url = (
                        claim.get("claimReview", [{}])[0]
                        .get("publisher", {})
                        .get("site", "")
                    )
                    textual_rating = claim.get("claimReview", [{}])[0].get(
                        "textualRating", ""
                    )
                    title = f"{publisher}: {textual_rating}"
                    snippet = f"{claim_text} - {claimant} ({review_date})"

                    try:
                        parsed_date = date_parser.parse(review_date)
                        if not is_within_days(parsed_date, days=7):
                            continue
                    except Exception:
                        pass

                    if url and url in seen:
                        continue
                    if url:
                        seen.add(url)
                    e = Evidence(
                        title=title,
                        url=url,
                        snippet=snippet,
                        content=snippet,
                        source="google_factcheck",
                        score=0.0,
                    )
                    e["source_tier"] = classify_domain(url) if url else "unknown"
                    e["image_path"], e["image_caption"] = (
                        _fetch_evidence_image(url) if url else (None, None)
                    )
                    results.append(e)
            except Exception:
                pass

    msg = f"[FakeSource] {len(results)} items from flagged+open+factcheck (last 7 days)"
    return {"evidence_fake": results, "messages": [("assistant", msg)]}


# ── A2A service wrapper ─────────────────────────────────────────────────────
from ..a2a_server import AgentCardConfig, BaseTaskHandler, run_server
from ..config import settings


class FakeSourceAgentHandler(BaseTaskHandler):
    """A2A TaskHandler exposing :func:`fake_source_agent` over HTTP (port 9004)."""

    agent_card_config = AgentCardConfig(
        name="fake_source_agent",
        description="Searches flagged/non-official sources and Google Fact Check API",
        version="1.0",
        skills=[
            {
                "id": "flagged_search",
                "name": "Flagged Domain Search",
                "description": "Search flagged + open web",
            },
            {
                "id": "factcheck_api",
                "name": "Google Fact Check API",
                "description": "Query the factchecktools API",
            },
        ],
        port=settings.a2a_port_fake_source,
    )

    async def agent_fn(self, state: FactCheckState) -> dict:
        return fake_source_agent(state)


if __name__ == "__main__":
    run_server(FakeSourceAgentHandler(), FakeSourceAgentHandler.agent_card_config)
