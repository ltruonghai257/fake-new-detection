"""In-memory evidence graph backed by networkx.DiGraph."""

from __future__ import annotations

import networkx as nx
from typing import Any, Dict, List, Optional


class EvidenceGraph:
    def __init__(self, graph_data: Optional[Dict[str, Any]] = None) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        # graph_data enables checkpoint persistence: the langgraph checkpointer
        # serializes state via jsonplus msgpack, which reconstructs objects
        # with `_asdict()` via cls(**kwargs). Plain nodes/edges round-trip;
        # an in-memory nx.DiGraph alone is not msgpack-serializable.
        if graph_data:
            for node_id, attrs in graph_data.get("nodes", []):
                self.graph.add_node(node_id, **attrs)
            for src, dst, attrs in graph_data.get("edges", []):
                self.graph.add_edge(src, dst, **attrs)

    def _asdict(self) -> Dict[str, Any]:
        """Plain-data view for checkpoint serialization (jsonplus namedtuple protocol)."""
        return {
            "graph_data": {
                "nodes": [(n, dict(d)) for n, d in self.graph.nodes(data=True)],
                "edges": [(u, v, dict(d)) for u, v, d in self.graph.edges(data=True)],
            }
        }

    @classmethod
    def build_from_evidence(
        cls, evidence_list: List[Dict[str, Any]]
    ) -> "EvidenceGraph":
        eg = cls()
        eg.graph.add_node("statement", node_type="statement")
        for item in evidence_list:
            node_id = item.get("url") or str(hash(item.get("snippet", "")))
            eg.graph.add_node(
                node_id,
                node_type="evidence",
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                source_tier=item.get("source_tier", "unknown"),
            )
            eg.graph.add_edge("statement", node_id, type="mentions")
        return eg

    def add_node(self, id: Any, attrs: Dict[str, Any]) -> None:
        self.graph.add_node(id, **attrs)

    def add_edge(
        self, src: Any, dst: Any, type: str, attrs: Optional[Dict[str, Any]] = None
    ) -> None:
        self.graph.add_edge(src, dst, type=type, **(attrs or {}))

    def to_evidence_list(self) -> List[Dict[str, Any]]:
        result = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == "evidence":
                result.append(
                    {
                        "url": node_id,
                        "title": data.get("title", ""),
                        "snippet": data.get("snippet", ""),
                        "source_tier": data.get("source_tier", "unknown"),
                    }
                )
        return result
