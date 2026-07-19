"""Social search sub-node: site-restricted queries against social platforms.

Reuses state["search_queries"] (drafted in Vietnamese by search_agent),
runs them through the existing web_search tool with include_domains restricted
to ["twitter.com", "facebook.com"], tags results source_tier="social", and
merges new nodes + "mentions" edges into the existing evidence_graph DiGraph.

This node assumes it is only reached when reliability_signal=True; the
conditional routing is handled in graph.py (Phase 6).
"""
from __future__ import annotations

from typing import List

from ..graph_utils import EvidenceGraph
from ..helpers import _fetch_evidence_image
from ..state import Evidence, FactCheckState
from ..tools.web_search import web_search

_SOCIAL_DOMAINS = ["twitter.com", "facebook.com"]
_MAX_RESULTS_PER_QUERY = 3


def social_search_agent(state: FactCheckState) -> dict:
    queries: List[str] = state.get("search_queries") or []
    eg: EvidenceGraph = state.get("evidence_graph") or EvidenceGraph()

    seen: set = set(eg.graph.nodes)

    new_count = 0
    for q in queries:
        results = web_search(q, max_results=_MAX_RESULTS_PER_QUERY, include_domains=_SOCIAL_DOMAINS)
        for r in results:
            url = r.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)

            img_path, img_caption = _fetch_evidence_image(url)

            eg.add_node(
                url,
                {
                    "node_type": "evidence",
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "source_tier": "social",
                    "image_path": img_path,
                    "image_caption": img_caption,
                },
            )
            eg.add_edge("statement", url, type="mentions")
            new_count += 1

    msg = (
        f"[SocialSearch] {new_count} new social items merged into graph "
        f"({len(queries)} queries x {_MAX_RESULTS_PER_QUERY} results max)"
    )
    return {
        "evidence_graph": eg,
        "messages": [("assistant", msg)],
    }
