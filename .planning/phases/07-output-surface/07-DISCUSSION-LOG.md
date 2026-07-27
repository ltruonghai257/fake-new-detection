# Phase 7: Output Surface - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-27
**Phase:** 7-Output Surface
**Areas discussed:** CLI display format, Top-level field promotion, README.md scope

---

## CLI display format

### How should the VERDICT line look in _print_human()?

| Option | Description | Selected |
|--------|-------------|----------|
| Compact single-line | `VERDICT: Thật  (TRUE, confidence 0.85)` — Vietnamese primary, 4-class + confidence in parens. Minimal change. | ✓ |
| Split confidence to separate line | `VERDICT: Thật  (TRUE)\nConfidence: 0.85` — cleaner separation but adds a line. | |
| Uppercase Vietnamese label | `VERDICT: THẬT  (TRUE, confidence 0.85)` — allcaps to match existing style. | |

**User's choice:** Compact single-line
**Notes:** Minimize diff; keep all on one line. Vietnamese primary, parenthetical 4-class.

### Fallback when verdict_label_vi is missing?

| Option | Description | Selected |
|--------|-------------|----------|
| Fall back to 4-class label as primary | `VERDICT: UNVERIFIED  (confidence 0.20)` — old format preserved | ✓ |
| Always use v.get('verdict_label_vi') or v.get('label') | One .get() call, no explicit branch | |

**User's choice:** Fall back to 4-class label as primary
**Notes:** Defensive; keeps old behavior for any pre-Phase-5 state.

---

## Top-level field promotion

### How should run_fact_check() expose fields at top level?

| Option | Description | Selected |
|--------|-------------|----------|
| Mutate result dict after invoke | In-place: `result['verdict_binary'] = result.get('verdict', {}).get('verdict_binary')` | ✓ |
| Build a new wrapper dict | `return {**graph.invoke(state), 'verdict_binary': ..., ...}` — new dict, avoids mutation | |

**User's choice:** Mutate result dict in-place
**Notes:** Minimal code change.

### MCP fact_check() — where do new fields go?

| Option | Description | Selected |
|--------|-------------|----------|
| Top level alongside verdict dict | `{'verdict': {...}, 'verdict_binary': 'REAL', 'verdict_label_vi': 'Thật', ...}` | ✓ |
| Already inside verdict — no change needed | Fields accessible via `result['verdict']['verdict_binary']` | |

**User's choice:** Top level alongside verdict dict
**Notes:** Consistent with OUTPUT-04 wording "alongside existing verdict dict".

---

## README.md scope

| Option | Description | Selected |
|--------|-------------|----------|
| Example CLI output + run_fact_check() snippet only | Update terminal example + Python API return dict example | ✓ |
| CLI + Python API + MCP tool description | Also update MCP tools table/description | |
| Minimal — just CLI output example | Only terminal output line; defer Python API and MCP docs | |

**User's choice:** Example CLI output + run_fact_check() snippet only
**Notes:** No MCP section changes; no architecture changes.

---

## Claude's Discretion

None — user made all selections explicitly.

## Deferred Ideas

- MCP `evaluate_statement` tool still references old `evaluate_agent._coolant`/`._phobert` — Phase 6 scope
- MCP tools table / full README docs update — deferred to future docs pass
