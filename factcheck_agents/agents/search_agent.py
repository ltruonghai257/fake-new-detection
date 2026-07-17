"""Search agent: gather evidence from open truth sources.

Drafts a few focused search queries for the statement (LLM if available, else
a heuristic fallback), runs them through the web-search tool, and de-duplicates
the results into the shared state.
"""

from __future__ import annotations

from typing import List

from ..config import settings
from ..state import Evidence, FactCheckState
from ..tools.web_search import web_search
from .llm import get_llm, parse_json

_QUERY_PROMPT = (
    "You are a fact-checking research assistant. Given a claim, produce up to "
    "{n} short, diverse web-search queries that would surface authoritative "
    "evidence for or against it. Prefer the claim's original language. "
    'Respond as JSON: {{"queries": ["...", "..."]}}\n\nClaim: {statement}'
)


def _draft_queries(statement: str) -> List[str]:
    llm = get_llm()
    if llm is not None:
        try:
            resp = llm.invoke(_QUERY_PROMPT.format(n=settings.max_queries, statement=statement))
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

    seen, evidence = set(), []  # type: ignore[var-annotated]
    for q in queries:
        for e in web_search(q):
            url = e.get("url", "")
            if url and url in seen:
                continue
            if url:
                seen.add(url)
            evidence.append(e)

    evidence = sorted(evidence, key=lambda e: e.get("score", 0.0), reverse=True)
    evidence = evidence[: settings.max_results]

    msg = f"[Search] {len(queries)} queries -> {len(evidence)} evidence items"
    if not settings.has_search():
        msg += " (no search provider configured; continuing on model output only)"

    return {
        "search_queries": queries,
        "evidence": evidence,
        "messages": [("assistant", msg)],
    }
