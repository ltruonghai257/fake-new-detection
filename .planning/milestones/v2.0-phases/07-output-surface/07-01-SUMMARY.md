---
phase: 07-output-surface
plan: '01'
subsystem: api
tags: [cli, mcp, vietnamese, output-surface, additive]

requires:
  - phase: 05-conclusion-agent-binary-verdict-vietnamese
    provides: verdict_binary and verdict_label_vi fields in Verdict TypedDict
  - phase: 06-langgraph-wiring
    provides: graph routing through conclusion_agent to populate verdict
provides:
  - verdict_binary and verdict_label_vi surfaced at top level of run_fact_check() return dict
  - verdict_binary and verdict_label_vi in MCP fact_check tool response
  - verdict_label_vi as primary CLI label with 4-class label as parenthetical
  - Updated README.md CLI and Python API examples
affects: [08-tests]

tech-stack:
  added: []
  patterns: [additive-dict-promotion, defensive-get-with-fallback]

key-files:
  created: []
  modified:
    - factcheck_agents/cli.py
    - factcheck_agents/__init__.py
    - factcheck_agents/mcp_server.py
    - factcheck_agents/README.md

key-decisions:
  - "Two-branch conditional in _print_human() rather than complex f-string inline logic"
  - "Top-level promotion via post-invoke dict mutation in run_fact_check() — .get() throughout for total=False safety"
  - "OUTPUT-02 (--json) auto-satisfied — no code change needed; existing JSON dump includes result['verdict']"

patterns-established:
  - "Additive output surfacing: promote nested fields to top level without removing existing keys"

requirements-completed: [OUTPUT-01, OUTPUT-02, OUTPUT-03, OUTPUT-04]

duration: 5min
completed: 2026-07-27
---

# Phase 7: Output Surface Summary

**Vietnamese binary verdict labels surfaced across CLI, Python API, and MCP tool — all changes additive, no existing callers broken**

## Performance

- **Duration:** 5 min
- **Tasks:** 4
- **Files modified:** 4

## Accomplishments
- CLI `_print_human()` shows `verdict_label_vi` ("Thật"/"Giả") as primary label with 4-class label as parenthetical
- `run_fact_check()` return dict promotes `verdict_binary` and `verdict_label_vi` to top level
- MCP `fact_check()` response includes both new fields alongside all 5 existing keys
- README.md updated with new CLI output example and extended Python snippet

## Task Commits

1. **Task 1: _print_human() verdict_label_vi branch** - `feat(07-01)`
2. **Task 2: run_fact_check() top-level promotion** - `feat(07-01)`
3. **Task 3: mcp_server fact_check() new keys** - `feat(07-01)`
4. **Task 4: README.md examples** - `docs(07-01)`

## Files Created/Modified
- `factcheck_agents/cli.py` — `_print_human()` two-branch conditional for verdict_label_vi
- `factcheck_agents/__init__.py` — `run_fact_check()` post-invoke dict mutation
- `factcheck_agents/mcp_server.py` — `fact_check()` return dict gains two keys
- `factcheck_agents/README.md` — CLI output example + Python snippet updated

## Decisions Made
- Used two-branch `if/else` in `_print_human()` instead of complex inline f-string logic — cleaner, easier to maintain
- OUTPUT-02 (`--json`) required no code change — existing `printable` dict comprehension already includes `result["verdict"]` which contains both new fields

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All output surfaces now expose `verdict_binary` and `verdict_label_vi`
- Phase 8 (Tests) can write integration tests verifying these fields appear in CLI, Python API, and MCP responses
- No blockers

---
*Phase: 07-output-surface*
*Completed: 2026-07-27*
