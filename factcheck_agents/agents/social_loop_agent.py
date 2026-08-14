"""Social loop agent: targeted search against tiktok.com + flagged domains.

Fires at most once (SOCLOOP-03) when evidence is weak, writes results to
evidence_social only (never merged into evidence_real/evidence_fake per EVRET-03).
"""

from __future__ import annotations

from typing import List

from ..config import settings
from ..helpers import _fetch_evidence_image
from ..source_tier import classify_domain
from ..state import Evidence, FactCheckState
from ..tools.web_search import web_search


def social_loop_agent(state: FactCheckState) -> dict:
    """Search tiktok.com + flagged domains, write to evidence_social, set social_loop_fired=True."""
    queries: List[str] = state.get("search_queries") or [state["statement"]]
    errors: List[str] = state.get("errors") or []

    # D-08: Target domains evaluated at call time
    targets = ["tiktok.com"] + [
        d.strip() for d in settings.flagged_domains.split(",") if d.strip()
    ]

    results: List[Evidence] = []
    seen: set = set()

    for q in queries:
        try:
            search_results = web_search(
                q, max_results=settings.max_results, include_domains=targets
            )
        except Exception as e:
            errors.append(f"[SocialLoop] web_search failed for '{q}': {e}")
            continue

        for r in search_results:
            url = r.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)

            img_path, img_caption = _fetch_evidence_image(url)

            evidence_item: Evidence = {
                "title": r.get("title", ""),
                "url": url,
                "snippet": r.get("snippet", ""),
                "content": r.get("content", ""),
                "source": r.get("source", ""),
                "score": r.get("score", 0.0),
                "source_tier": classify_domain(url),
                "image_path": img_path,
                "image_caption": img_caption,
            }
            results.append(evidence_item)

    # SOCLOOP-02: Always set social_loop_fired=True, even if 0 results or exception
    return {
        "evidence_social": results,
        "social_loop_fired": True,
        "messages": [("assistant", f"[SocialLoop] {len(results)} items")],
        "errors": errors,
    }


# ── A2A service wrapper ─────────────────────────────────────────────────────
from ..a2a_server import AgentCardConfig, BaseTaskHandler, run_server


class SocialLoopAgentHandler(BaseTaskHandler):
    """A2A TaskHandler exposing :func:`social_loop_agent` over HTTP (port 9005)."""

    agent_card_config = AgentCardConfig(
        name="social_loop_agent",
        description="Searches tiktok.com + flagged domains when evidence is weak",
        version="1.0",
        skills=[
            {
                "id": "social_search",
                "name": "Social Media Search",
                "description": "Search tiktok.com and flagged domains",
            }
        ],
        port=settings.a2a_port_social_loop,
    )

    async def agent_fn(self, state: FactCheckState) -> dict:
        return social_loop_agent(state)


if __name__ == "__main__":
    run_server(SocialLoopAgentHandler(), SocialLoopAgentHandler.agent_card_config)
