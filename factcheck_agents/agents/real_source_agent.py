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
    queries = state.get("search_queries")
    claim_variants = state.get("claim_variants") or []
    if not queries:  # debate graph has no search_agent: build keyword queries here
        from .search_agent import build_queries

        queries, variants = build_queries(state["statement"])
        queries, claim_variants = list(queries), list(variants)
    queries = list(
        dict.fromkeys(queries + claim_variants)
    )  # deduplicate, preserve order
    seen: set = set()
    from .source_utils import parallel_search_and_crawl

    results: List[Evidence] = parallel_search_and_crawl(queries, [REAL_DOMAINS], seen)
    workflow_steps = [
        {
            "step": f"Search {len(queries)} queries in parallel",
            "description": f"Tavily/Google CSE with include_domains={REAL_DOMAINS[:2]}, crawl + {settings.search_max_age_days or 7}-day filter",
            "input": f"queries={[q[:40] for q in queries]}",
            "output": f"{len(results)} items kept",
        }
    ]

    msg = f"[RealSource] {len(results)} items from {REAL_DOMAINS[:2]}... (last 7 days)"
    return {
        "evidence_real": results,
        "messages": [("assistant", msg)],
        "evidence_workflow_steps": workflow_steps,
    }


# ── A2A service wrapper ─────────────────────────────────────────────────────
from ..a2a_server import AgentCardConfig, BaseTaskHandler, run_server
from ..config import settings


class RealSourceAgentHandler(BaseTaskHandler):
    """A2A TaskHandler exposing :func:`real_source_agent` over HTTP (port 9003)."""

    agent_card_config = AgentCardConfig(
        name="real_source_agent",
        description="Searches trusted Vietnamese news domains for real evidence",
        version="1.0",
        skills=[
            {
                "id": "trusted_search",
                "name": "Trusted Domain Search",
                "description": "Search vnexpress, tuoitre, thanhnien, ...",
            }
        ],
        port=settings.a2a_port_real_source,
    )

    async def agent_fn(self, state: FactCheckState) -> dict:
        return real_source_agent(state)


if __name__ == "__main__":
    run_server(RealSourceAgentHandler(), RealSourceAgentHandler.agent_card_config)
