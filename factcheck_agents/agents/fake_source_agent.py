"""Fake source agent: gather evidence from fact-checking sources.

Searches tingia.gov.vn and the Google Fact Check API for evidence refuting
or fact-checking the claim.
"""

from __future__ import annotations

import requests
from typing import List

from ..config import settings
from ..helpers import _fetch_evidence_image
from ..source_tier import classify_domain
from ..state import Evidence, FactCheckState
from ..tools.web_search import web_search


def fake_source_agent(state: FactCheckState) -> dict:
    """Search fact-checking sources for fake evidence.

    Returns {"evidence_fake": results, "messages": [...]}.
    Never raises; returns empty list on all failures (EVRET-03).
    """
    queries = state.get("search_queries") or [state["statement"]]
    results: List[Evidence] = []
    seen: set = set()

    for q in queries:
        # tingia.gov.vn path (EVRET-02)
        try:
            q_results = web_search(
                q, max_results=settings.max_results, include_domains=["tingia.gov.vn"]
            )
            for e in q_results:
                url = e.get("url", "")
                if url and url in seen:
                    continue
                if url:
                    seen.add(url)
                e["source_tier"] = classify_domain(url) if url else "unknown"
                e["image_path"], e["image_caption"] = (
                    _fetch_evidence_image(url) if url else (None, None)
                )
                results.append(e)
        except Exception:
            # Skip tingia.gov.vn on failure
            pass

        # Google Fact Check API path (EVRET-02)
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
                    # Extract relevant fields from Google Fact Check API response
                    claim_text = claim.get("text", "")
                    claimant = claim.get("claimant", "")
                    review_date = claim.get("claimReview", [{}])[0].get("reviewDate", "")
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
                    textual_rating = (
                        claim.get("claimReview", [{}])[0].get("textualRating", "")
                    )
                    title = f"{publisher}: {textual_rating}"
                    snippet = f"{claim_text} - {claimant} ({review_date})"
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
                # Stub to [] on Google Fact Check API failure
                pass

    msg = f"[FakeSource] {len(results)} items"
    return {"evidence_fake": results, "messages": [("assistant", msg)]}