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

    Returns {"evidence_real": results, "messages": [...]}.
    Never raises; returns empty list on all failures (EVRET-03).
    """
    queries = state.get("search_queries") or [state["statement"]]
    results: List[Evidence] = []
    seen: set = set()

    for q in queries:
        try:
            q_results = web_search(
                q, max_results=settings.max_results, include_domains=REAL_DOMAINS
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
        except Exception as exc:
            errors = state.get("errors", [])
            errors.append(f"[RealSource] Query '{q}' failed: {exc}")

    msg = f"[RealSource] {len(results)} items from {REAL_DOMAINS[:2]}..."
    return {"evidence_real": results, "messages": [("assistant", msg)]}