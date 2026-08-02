# Phase 1: State, Config & Evidence Graph Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-19
**Phase:** 1-State, Config & Evidence Graph Foundation
**Areas discussed:** Graph library choice, EvidenceGraph API surface, source_tier field strictness, State typing for evidence_graph

---

## Graph library choice

| Option | Description | Selected |
|--------|-------------|----------|
| Add networkx | Matches EVGRAPH-01 exactly. Real DiGraph with typed edges, built-in graph algorithms. Adds ~3MB dep. Phase 5 conclusion agent can call G.in_edges(node, data=True) to find 'contradicts' edges naturally. | ✓ |
| Plain dict-of-dicts | Zero new deps. Works but Phase 5 code has to iterate manually. PROJECT.md says acceptable but EVGRAPH-01 says DiGraph explicitly. | |
| networkx optional import | Try import networkx, fall back to dict stub if not installed. Means two code paths for edge queries in Phase 5. | |

**User's choice:** Add networkx

| Option | Description | Selected |
|--------|-------------|----------|
| agents extra (alongside langgraph) | networkx only needed when running fact-check agents. Consistent with how langgraph, openai, etc. are gated. | ✓ |
| base dependencies | Makes networkx always available regardless of extras. Simpler but installs for training pipeline which doesn't need it. | |

**User's choice:** agents extra

---

## EvidenceGraph API surface

| Option | Description | Selected |
|--------|-------------|----------|
| Thin wrapper (4 methods only) | Expose only what EVGRAPH-01 specifies. Agents needing DiGraph-level access use .graph property. Keeps class minimal. | ✓ (Claude decided) |
| Richer wrapper (hide DiGraph entirely) | Add get_conflicts(), trusted_snippets(), merge(). More API surface to maintain. | |
| No wrapper — use DiGraph directly | Store bare networkx.DiGraph in state. Contradicts EVGRAPH-01 'EvidenceGraph wrapper' spec. | |

**User's choice:** Deferred to Claude — Claude chose thin wrapper (premature to add convenience methods before Phase 5 reveals what's needed).

| Option | Description | Selected |
|--------|-------------|----------|
| Classmethod: EvidenceGraph.build_from_evidence(evidence_list) → EvidenceGraph | Factory classmethod. Keeps construction logic inside the class. Consistent with factory pattern. | ✓ |
| Constructor: EvidenceGraph(evidence_list=None) | Build on init. Flexible but conflates construction vs. mutation. | |

**User's choice:** Classmethod

| Option | Description | Selected |
|--------|-------------|----------|
| Not needed — in-memory only per check-run | Graph built fresh per request and discarded after. No pause/resume requirement. | ✓ |
| Add __reduce__ or to_dict/from_dict for serialization | Needed for LangGraph state persistence (SQLite/Redis checkpointer). Adds complexity to Phase 1. | |

**User's choice:** Not needed — in-memory only

| Option | Description | Selected |
|--------|-------------|----------|
| Two node types: statement + evidence snippets | Statement node (id='statement') + one node per Evidence item. Edges connect statement ↔ snippets. | ✓ |
| Three node types: statement + source domains + snippets | Adds intermediate 'source' nodes grouping snippets. Overkill for Phase 1. | |

**User's choice:** Two node types

---

## source_tier field strictness

| Option | Description | Selected |
|--------|-------------|----------|
| Literal['trusted','flagged','social','unknown'] | Static type checking catches typos. MyPy/Pyright will flag invalid tier values. | ✓ |
| Plain str | No import needed. Consistent with how other Evidence fields are typed. | |

**User's choice:** Literal type

| Option | Description | Selected |
|--------|-------------|----------|
| Add to existing Evidence (total=False) | Additive change. total=False makes it optional for callers that don't set it. Simplest change. | ✓ |
| New SearchEvidence(Evidence) subtype | Evidence stays clean but TypedDict inheritance with total=False has compat subtleties. | |

**User's choice:** Add to existing Evidence

| Option | Description | Selected |
|--------|-------------|----------|
| Pure function only | classify_domain(url) -> str. Simple, testable. Tier values already documented via Literal in state.py. | ✓ |
| Function + SourceTier constants | Avoids magic strings at call sites. Enum adds a class with no real benefit for 4 values. | |

**User's choice:** Pure function only

---

## State typing for evidence_graph

| Option | Description | Selected |
|--------|-------------|----------|
| Optional[Any] — no import from graph_utils | state.py stays pure data-definition file. Agents that need EvidenceGraph methods import graph_utils directly. Matches ROADMAP spec. | ✓ |
| Optional['EvidenceGraph'] forward ref | Forward ref signals intent but state.py would depend on graph_utils at runtime; TYPE_CHECKING guards add noise. | |

**User's choice:** Optional[Any]

| Option | Description | Selected |
|--------|-------------|----------|
| Top-level field on FactCheckState | reliability_signal: Optional[bool] alongside evidence_graph. Simple flat access. Phase 6 routing reads it directly. | ✓ |
| Nested inside a verify_result dict | Groups verify outputs together but contradicts REQUIREMENTS spec for route_after_verify. | |

**User's choice:** Top-level field

---

## Claude's Discretion

- **EvidenceGraph wrapper thickness:** User said "you decide" — Claude chose thin wrapper (4 methods only). Rationale: premature to add convenience methods like `get_conflicts()` before Phase 5 reveals exact needs. Phase 5 can use `.graph.in_edges(node, data=True)` directly.

## Deferred Ideas

None — discussion stayed within phase scope.
