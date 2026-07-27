# Phase 7: Output Surface - Context

**Gathered:** 2026-07-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Surface `verdict_binary` and `verdict_label_vi` additively in three places: `cli.py` `_print_human()` (human-readable display), `__init__.py` `run_fact_check()` (Python API return dict), and `mcp_server.py` `fact_check` tool (MCP response dict). Also update README.md example CLI output and Python API snippet.

All changes are additive — no existing fields removed, no callers broken. `verdict_binary` and `verdict_label_vi` are already written to `Verdict` by `conclusion_agent` (Phase 5); this phase only surfaces them at the output layer.

No graph wiring (Phase 6), no test files (Phase 8), no changes to agents or state schema.

</domain>

<decisions>
## Implementation Decisions

### CLI Human Display (`_print_human`)
- **D-01:** Compact single-line format: `VERDICT: Thật  (TRUE, confidence 0.85)` — `verdict_label_vi` as primary label, 4-class `label` + confidence in parentheses on the same line.
- **D-02:** Fallback: if `verdict_label_vi` is absent from the verdict dict (pre-Phase-5 state or unexpected absence), fall back to the old format `VERDICT: {label}  (confidence {conf:.2f})` — preserve existing behavior for callers that haven't run Phase 5.
- **D-03:** `--json` output: **no code change needed** — the existing `printable = {k: v for k, v in result.items() if k != "messages"}` dump already includes `result["verdict"]` which contains `verdict_binary` and `verdict_label_vi`. OUTPUT-02 is auto-satisfied.

### `run_fact_check()` Top-Level Promotion
- **D-04:** After `graph.invoke(state)`, mutate the result dict in-place to promote the two new fields to top level:
  ```python
  result["verdict_binary"] = result.get("verdict", {}).get("verdict_binary")
  result["verdict_label_vi"] = result.get("verdict", {}).get("verdict_label_vi")
  return result
  ```
  This makes them accessible as `result["verdict_binary"]` without breaking existing callers that read `result["verdict"]["verdict_binary"]`.

### MCP `fact_check` Response
- **D-05:** Add `verdict_binary` and `verdict_label_vi` at the **top level** of the explicitly-constructed return dict, extracted from `result.get("verdict", {})`. They sit alongside `verdict`, `model_results`, `evidence`, `search_queries` — consistent with OUTPUT-04 "alongside existing verdict dict".

### README.md Updates
- **D-06:** Update two sections only: (1) terminal output example — show the new `VERDICT: Thật (TRUE, confidence ...)` line; (2) `run_fact_check()` Python snippet — show `verdict_binary` and `verdict_label_vi` in the example return dict. No MCP section changes, no architecture section changes.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Scope
- `.planning/REQUIREMENTS.md` §Output Surface (OUTPUT-01..OUTPUT-04) — exact field names, top-level requirement, no-breaking-change constraint
- `.planning/PROJECT.md` §Constraints — scope boundary (`factcheck_agents/` and `tests/` only); additive-only fields; all user-facing output in Vietnamese; internal field names stay English

### Prior Phase Decisions (fields being surfaced)
- `.planning/phases/05-conclusion-agent-binary-verdict-vietnamese/05-CONTEXT.md` — D-03 (binary mapping logic, `verdict_binary`/`verdict_label_vi` written by agent code not LLM), D-04 (Verdict TypedDict fields), D-05 (fallback verdict includes both fields)
- `.planning/phases/04-social-search-sub-node/04-CONTEXT.md` — D-02 (flat `evidence` list stays; `verdict` dict is source of truth for output)

### Existing Files to Modify
- `factcheck_agents/cli.py` — `_print_human()`: update VERDICT line format; `main()`: `--json` path needs no change
- `factcheck_agents/__init__.py` — `run_fact_check()`: post-process result dict to add top-level keys
- `factcheck_agents/mcp_server.py` — `fact_check()`: add `verdict_binary` and `verdict_label_vi` to return dict
- `factcheck_agents/README.md` (or root `README.md` if that's where CLI/API docs live) — update CLI output example and `run_fact_check()` snippet

### State Schema (read-only reference)
- `factcheck_agents/state.py` `Verdict` TypedDict — `verdict_binary: Literal["REAL", "FAKE"]`, `verdict_label_vi: Literal["Thật", "Giả"]` (both `total=False`); already there from Phase 5

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cli.py` `_print_human()` L17-35: minimal edit — change line 20 from `v.get('label', 'UNVERIFIED')` to `v.get('verdict_label_vi') or v.get('label', 'UNVERIFIED')` and adjust confidence placement. Add `label` as parenthetical when `verdict_label_vi` is present.
- `__init__.py` `run_fact_check()` L28-34: 3-line addition after `return graph.invoke(state)` is split into assign + mutate + return.
- `mcp_server.py` `fact_check()` L44-50: add two keys to the existing return dict literal.

### Established Patterns
- `v.get("key", default)` dict access pattern — used throughout `_print_human()`; follow the same defensiveness for `verdict_label_vi`
- Additive dict construction — `mcp_server.py` explicitly lists keys; add new keys without removing existing ones
- `total=False` TypedDict fields — all `Verdict` fields are optional; always use `.get()` when reading

### Integration Points
- **Reading from state:** `result["verdict"]["verdict_binary"]` and `result["verdict"]["verdict_label_vi"]` — both written by `conclusion_agent` (Phase 5); graceful degrade fallback also writes them
- **Phase 6 dependency:** Phase 7 depends on Phase 6 (LangGraph Wiring) being complete — the graph must route through `conclusion_agent` to populate `verdict`
- **`evaluate_agent` import in `mcp_server.py`:** `from .agents.evaluate_agent import _coolant, _phobert` on L31 — this will be updated in Phase 6 (not Phase 7 scope); leave it untouched here

</code_context>

<specifics>
## Specific Ideas

- `_print_human()` new VERDICT line: `f"VERDICT: {v.get('verdict_label_vi', v.get('label', 'UNVERIFIED'))}  ({v.get('label', '')+', ' if v.get('verdict_label_vi') else ''}confidence {v.get('confidence', 0):.2f})"` — or a cleaner two-step conditional that only shows the parenthetical `label` when `verdict_label_vi` is present.
- Simpler implementation: check `verdict_label_vi` first, branch into two format strings — avoids complex f-string inline logic.
- `run_fact_check()` mutation: assign `result` before returning, then two lines to set `result["verdict_binary"]` and `result["verdict_label_vi"]`.

</specifics>

<deferred>
## Deferred Ideas

- MCP `evaluate_statement` tool update (still references old `evaluate_agent._coolant` / `._phobert`) — Phase 6 scope (LangGraph Wiring renames evaluate → verify)
- MCP tools table / MCP tool description updates in README — deferred to a future docs pass; only the CLI output example and Python API snippet update in Phase 7

None other — discussion stayed within phase scope.

</deferred>

---

*Phase: 7-Output Surface*
*Context gathered: 2026-07-27*
