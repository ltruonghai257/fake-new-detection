# Phase 1: State, Config & Evidence Graph Foundation - Context

**Gathered:** 2026-07-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend existing `TypedDict` definitions in `state.py` and `Settings` dataclass in `config.py` with new fields for evidence graph and source tiers. Create two new modules — `factcheck_agents/graph_utils.py` (`EvidenceGraph` wrapper around `networkx.DiGraph`) and `factcheck_agents/source_tier.py` (`classify_domain()` pure function). No agent logic, no LangGraph wiring — pure data contracts and utility code that all downstream phases depend on.

</domain>

<decisions>
## Implementation Decisions

### Graph Library
- **D-01:** Use `networkx.DiGraph` (not plain dict) — add `networkx` to the `agents` extra in `pyproject.toml` alongside `langgraph`.
- **D-02:** The graph is **in-memory only per check-run** — no serialization, no LangGraph state checkpointing integration needed.

### EvidenceGraph Wrapper (`graph_utils.py`)
- **D-03:** Thin wrapper class — expose exactly 4 methods: `build_from_evidence(evidence_list)` (classmethod), `add_node(id, attrs)`, `add_edge(src, dst, type, attrs)`, `to_evidence_list()`. Expose `.graph` property for raw DiGraph access.
- **D-04:** Construction via classmethod: `EvidenceGraph.build_from_evidence(evidence_list) -> EvidenceGraph`. Constructor creates an empty graph for incremental mutation.
- **D-05:** Two node types only — `statement` node (id=`"statement"`, text=statement string) + one node per Evidence item (id=URL or hash, carrying title, snippet, source_tier). Edges connect statement ↔ snippets with typed edge attribute (`supports`/`contradicts`/`mentions`).

### source_tier Field (`state.py` + `source_tier.py`)
- **D-06:** `source_tier` added to **existing `Evidence` TypedDict** (not a new subtype) — `total=False` already makes it optional for callers that don't set it.
- **D-07:** Type annotation: `Literal["trusted", "flagged", "social", "unknown"]` (import `Literal` from `typing`).
- **D-08:** `classify_domain(url: str) -> str` is a **pure function** in `source_tier.py` — no enum/constants class. It reads from `settings` (imported from `config.py`).

### FactCheckState Fields (`state.py`)
- **D-09:** `evidence_graph: Optional[Any]` — typed as `Any` to avoid importing `networkx` or `graph_utils` in `state.py`. State stays a pure data-definition file.
- **D-10:** `reliability_signal: Optional[bool]` lives as a **top-level field** on `FactCheckState` (not nested). Phase 6 routing reads it directly: `state.get("reliability_signal", False)`.

### Claude's Discretion
- `EvidenceGraph` wrapper design (thin vs. rich): chose **thin wrapper** — premature to add `get_conflicts()` or `trusted_snippets()` before Phase 5 reveals what's needed. Phase 5 will use `.graph.in_edges(node, data=True)` for conflict detection.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Scope
- `.planning/REQUIREMENTS.md` §Evidence Graph Infrastructure (EVGRAPH-01–03) — exact networkx.DiGraph spec, edge types, tier attribute
- `.planning/REQUIREMENTS.md` §Source Tier Configuration (CONFIG-01–03) — env var names, defaults, Evidence.source_tier spec
- `.planning/REQUIREMENTS.md` §Tests (TEST-01, TEST-02) — what unit tests Phase 1 must enable

### Project Decisions
- `.planning/PROJECT.md` §Constraints — scope boundary (factcheck_agents/ and tests/ only, no new paid APIs)
- `.planning/PROJECT.md` §Key Decisions — evidence graph as plain Python structure rationale

### Existing Code to Modify
- `factcheck_agents/state.py` — existing `Evidence`, `FactCheckState` TypedDicts to extend
- `factcheck_agents/config.py` — existing `Settings` dataclass; follow the `field(default_factory=lambda: os.getenv(...))` pattern for new env vars
- `pyproject.toml` — `agents` extra where `networkx` must be added

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `factcheck_agents/config.py` `Settings` dataclass: all new env vars follow the `field(default_factory=lambda: os.getenv("VAR_NAME", "default"))` pattern already established.
- `factcheck_agents/state.py` `Evidence(TypedDict, total=False)`: adding `source_tier` is additive — `total=False` makes it optional for all existing callers.

### Established Patterns
- All settings from env vars (CFG-01 validated) — `FACTCHECK_TRUSTED_DOMAINS`, `FACTCHECK_FLAGGED_DOMAINS`, `FACTCHECK_RELIABILITY_THRESHOLD` follow the existing `FACTCHECK_*` namespace.
- TypedDict-only state (no Pydantic, no dataclass) — keep `graph_utils.py` as a plain Python class, not a Pydantic model.
- Graceful degrade everywhere — `evidence_graph: Optional[Any]` with `None` default means agents that run before the search agent don't crash.

### Integration Points
- `search_agent.py` (Phase 2) will call `classify_domain()` from `source_tier.py` and `EvidenceGraph.build_from_evidence()` from `graph_utils.py`.
- `verify_agent.py` (Phase 3) will read `state["evidence_graph"]` and write `state["reliability_signal"]`.
- `graph.py` (Phase 6) will read `state["reliability_signal"]` in `route_after_verify()`.

</code_context>

<specifics>
## Specific Ideas

- Default trusted domains (from REQUIREMENTS CONFIG-01): `vnexpress.net,thanhnien.vn,dantri.com.vn,tuoitre.vn`
- Default flagged domains (from REQUIREMENTS CONFIG-02): `kenh14.vn`
- Default reliability threshold (from REQUIREMENTS VERIFY-02): `0.5`
- Edge types for DiGraph: `"supports"`, `"contradicts"`, `"mentions"` (from EVGRAPH-01)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 1-State, Config & Evidence Graph Foundation*
*Context gathered: 2026-07-19*
