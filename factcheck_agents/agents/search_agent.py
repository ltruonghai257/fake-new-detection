"""Search agent: gather evidence from open truth sources.

Drafts a few focused search queries for the statement (LLM if available, else
a heuristic fallback), runs them through the web-search tool, and de-duplicates
the results into the shared state.
"""

from __future__ import annotations

from typing import List

from ..config import settings
from ..graph_utils import EvidenceGraph
from ..helpers import _fetch_evidence_image
from ..source_tier import classify_domain
from ..state import Evidence, FactCheckState
from ..tools.web_search import web_search
from .llm import get_llm, parse_json
from ..prompts import SEARCH_QUERY_PROMPT


def _draft_queries(statement: str) -> List[str]:
    llm = get_llm()
    if llm is not None:
        try:
            resp = llm.invoke(
                SEARCH_QUERY_PROMPT.format(n=settings.max_queries, statement=statement)
            )
            data = parse_json(getattr(resp, "content", "") or "")
            if data and isinstance(data.get("queries"), list):
                qs = [q.strip() for q in data["queries"] if q and q.strip()]
                if qs:
                    return qs[: settings.max_queries]
        except Exception:
            pass
    # heuristic fallback: use the statement itself
    return [statement.strip()]


def search_agent(state: FactCheckState) -> dict:
    statement = state["statement"]
    queries = _draft_queries(statement)

    trusted_list = [d.strip() for d in settings.trusted_domains.split(",") if d.strip()]
    flagged_list = [d.strip() for d in settings.flagged_domains.split(",") if d.strip()]

    seen: set = set()
    raw: List[Evidence] = []

    for include_domains in (trusted_list, flagged_list, None):
        for q in queries:
            results = web_search(q, include_domains=include_domains)
            for e in results:
                url = e.get("url", "")
                if url and url in seen:
                    continue
                if url:
                    seen.add(url)
                e["source_tier"] = classify_domain(url) if url else "unknown"
                e["image_path"], e["image_caption"] = (
                    _fetch_evidence_image(url) if url else (None, None)
                )
                raw.append(e)

    _tier_priority = {"trusted": 0, "flagged": 1, "social": 2, "unknown": 3}
    raw.sort(
        key=lambda e: (
            _tier_priority.get(e.get("source_tier", "unknown"), 3),
            -e.get("score", 0.0),
        )
    )
    cap = 2 * settings.max_results
    evidence = raw[:cap]

    evidence_graph = EvidenceGraph.build_from_evidence(evidence)

    msg = f"[Search] {len(queries)} queries x 3 passes -> {len(evidence)} evidence items (capped at {cap})"
    if not settings.has_search():
        msg += " (no search provider configured; continuing on model output only)"

    return {
        "search_queries": queries,
        "evidence": evidence,
        "evidence_graph": evidence_graph,
        "messages": [("assistant", msg)],
    }
