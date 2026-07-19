# Phase 5: Conclusion Agent (Binary Verdict + Vietnamese) - Context

**Gathered:** 2026-07-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend `factcheck_agents/agents/conclusion_agent.py` to: (1) detect cross-source conflicts via `state["evidence_graph"]` edge traversal, (2) apply the binary verdict rule and set `verdict_binary`/`verdict_label_vi` on the `Verdict` dict, (3) update `CONCLUSION_SYSTEM_PROMPT` to request Vietnamese rationale while keeping the same JSON response shape, and (4) update `_fallback_verdict` to return `verdict_binary`/`verdict_label_vi` with Vietnamese rationale strings. Also extend `Verdict` TypedDict in `state.py` with the two new fields.

No graph wiring changes (Phase 6), no output surface changes (Phase 7), no test files (Phase 8). Work confined to `state.py`, `conclusion_agent.py`, and `prompts.py`.

</domain>

<decisions>
## Implementation Decisions

### Conflict Detection Scope
- **D-01:** Only `contradicts` edges originating from a **trusted-tier** node to the statement node trigger a FAKE override. `flagged`-tier and `social`-tier `contradicts` edges are ignored by the rule. `"mentions"` edges (social nodes, Phase 4) are always neutral.
- **D-02:** If `state["evidence_graph"]` is `None` or missing, **skip the graph conflict check entirely** and fall through to the LLM/fallback path. Consistent with the graceful-degrade pattern: missing input = skip that step, not force a verdict.

### Binary Verdict Derivation (Two-Pass)
- **D-03:** **Agent code applies the binary mapping** — the LLM is not asked to emit `verdict_binary` directly. Flow: run LLM → get `label` (4-class: TRUE/FALSE/MISLEADING/UNVERIFIED) + Vietnamese `rationale` → agent code maps label to `verdict_binary` using the locked rule (SUPPORTED/TRUE/REAL→REAL, all others→FAKE) → if a trusted `contradicts` edge was found in the graph, override `verdict_binary=FAKE` regardless of LLM label. This is deterministic and testable (TEST-04).
- **D-04:** The `CONCLUSION_SYSTEM_PROMPT` **retains the same JSON response keys**: `label, confidence, rationale, citations, recommendation`. The only prompt changes are: add "Write `rationale` and `recommendation` in Vietnamese" and clarify label values are TRUE/FALSE/MISLEADING/UNVERIFIED. No new keys required from LLM.

### Fallback Rationale Language
- **D-05:** `_fallback_verdict` rationale and recommendation strings become **Vietnamese**. CONCL-05 requires all user-facing text in Vietnamese, including the no-LLM fallback path. This is a trivial literal string update.

### Evidence Tier Visibility to LLM
- **D-06:** `_format_evidence()` is updated to prepend **English tier labels** `[TRUSTED]`/`[FLAGGED]`/`[SOCIAL]`/`[UNKNOWN]` to each evidence item. Tier is read from `e.get("source_tier", "unknown")` on the flat `state["evidence"]` list — no graph traversal for formatting. English labels are acceptable here (technical context, not user-facing output per CONCL-05).

### Binary Mapping Table (locked from PROJECT.md + REQUIREMENTS.md)
- SUPPORTED / TRUE / REAL → `verdict_binary="REAL"`, `verdict_label_vi="Thật"`
- REFUTED / FALSE / FAKE / NEI / UNVERIFIED / MISLEADING → `verdict_binary="FAKE"`, `verdict_label_vi="Giả"`

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Scope
- `.planning/REQUIREMENTS.md` §Conclusion Agent (CONCL-01..CONCL-06) — exact conflict detection rule, binary mapping, Verdict field names, prompt language requirement, fallback contract
- `.planning/REQUIREMENTS.md` §Tests (TEST-04, TEST-05, TEST-06) — binary mapping test, Vietnamese label test, graceful degrade test
- `.planning/PROJECT.md` §Constraints — scope boundary (`factcheck_agents/` and `tests/` only); no new deps; additive fields only; all user-facing text in Vietnamese; internal field names stay English

### Locked Decisions from Prior Phases
- `.planning/phases/04-social-search-sub-node/04-CONTEXT.md` — D-02 (conclusion_agent reads `evidence_graph` directly, not flat evidence list for conflict detection), D-03 (social `"mentions"` edges are neutral — conflict detection reads only `"contradicts"`)
- `.planning/phases/03-verify-agent/03-CONTEXT.md` — D-01..D-05 (reliability_signal formula context); confirms COOLANT/PhoBERT have already run before conclusion agent
- `.planning/phases/01-state-config-evidence-graph-foundation/01-CONTEXT.md` — D-05 (node/edge structure: statement node, evidence nodes, edge types `supports`/`contradicts`/`mentions`), D-03 (EvidenceGraph 4-method API)

### Existing Code to Modify
- `factcheck_agents/agents/conclusion_agent.py` — primary file; add graph conflict detection, binary mapping, two-pass logic, tier-annotated `_format_evidence()`
- `factcheck_agents/state.py` `Verdict` TypedDict — add `verdict_binary: Literal["REAL", "FAKE"]` and `verdict_label_vi: Literal["Thật", "Giả"]` (both `total=False` for backward compat)
- `factcheck_agents/prompts.py` `CONCLUSION_SYSTEM_PROMPT` — add Vietnamese rationale instruction; no JSON key changes

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `conclusion_agent.py` `_fallback_verdict()`: update in-place — add binary mapping logic and Vietnamese strings; signature unchanged
- `conclusion_agent.py` `_format_evidence()`: update in-place — prepend `[TRUSTED]`/`[FLAGGED]`/`[SOCIAL]`/`[UNKNOWN]` from `e.get("source_tier", "unknown")` per item
- `conclusion_agent.py` `conclusion_agent()`: add graph conflict detection block (3–5 lines) before LLM call; add binary mapping + override after LLM response
- `factcheck_agents/graph_utils.py` `EvidenceGraph`: has `.graph` attribute (networkx DiGraph); use `.graph.edges(data=True)` to find `contradicts` edges from trusted nodes to statement node

### Established Patterns
- `total=False` TypedDict fields — already used throughout `Verdict`, `Evidence`, `ModelResult`; new fields follow the same pattern
- Graceful degrade: `None`-check before any graph operation; `try/except` wraps graph traversal same as model inference
- `lru_cache(maxsize=1)` singletons — not needed here (no new models in this phase)
- `label_map` dict pattern — already used in `_fallback_verdict` for label normalization; extend with binary mapping dict

### Integration Points
- **Reads from state:** `state["evidence_graph"]` (EvidenceGraph or None), `state["evidence"]` (List[Evidence]), `state["model_results"]` (List[ModelResult]), `state["statement"]`
- **Writes to state:** `{"verdict": Verdict(..., verdict_binary=..., verdict_label_vi=...)}` — same return shape, new fields added
- **Phase 6 graph.py:** no changes in Phase 5; `conclusion_agent` node name stays the same
- **Phase 7 output:** `cli.py`, `run_fact_check()`, MCP server will surface `verdict_binary` and `verdict_label_vi` from the Verdict dict written here

</code_context>

<specifics>
## Specific Ideas

- Graph traversal for conflict detection: iterate `evidence_graph.graph.edges(data=True)` looking for `edge_data.get("type") == "contradicts"` where the source node has `node_data.get("source_tier") == "trusted"` and the target is the statement node (node ID = `state["statement"]` or the statement text used as graph key in Phase 1).
- Vietnamese fallback rationale example: `"Không có mô hình nào khả dụng và không có LLM để đánh giá bằng chứng."` / `"Kết quả phỏng đoán dựa trên {model} ({label})."` — exact phrasing is at implementer's discretion; must be grammatically correct Vietnamese.
- `verdict_label_vi` mapping: `"Thật"` when `verdict_binary == "REAL"`, `"Giả"` when `"FAKE"` — set in one helper function to avoid duplication between LLM path and fallback path.
- Binary mapping dict: `_BINARY_MAP = {"TRUE": "REAL", "REAL": "REAL", "SUPPORTED": "REAL"}` — any key not in map → `"FAKE"`.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 5-Conclusion Agent (Binary Verdict + Vietnamese)*
*Context gathered: 2026-07-19*
