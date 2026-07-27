# Phase 8: Tests - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-27
**Phase:** 08-tests
**Areas discussed:** Integration test strategy (TEST-06), Coverage scope, File placement

---

## Integration test strategy (TEST-06)

### How should the graceful-degrade test invoke the pipeline?

| Option | Description | Selected |
|--------|-------------|----------|
| Full graph.invoke() with mocked agents | Call build_graph().invoke() with all agents patched. Tests real LangGraph wiring + state threading. MemorySaver avoids sqlite CI dependency. | ✓ |
| Manual agent-chain in sequence | Call agents directly in sequence, merge state manually. Avoids LangGraph overhead but doesn't test graph.py wiring. | |
| Two tests: unit (agent-chain) + graph-level | Lightweight agent-chain test + graph smoke test. | |

**User's choice:** Full graph.invoke() with mocked agents

---

### How should agents be mocked inside the graph.invoke() test?

| Option | Description | Selected |
|--------|-------------|----------|
| Patch agent functions at module import level | @patch each agent to return a minimal state dict. Cleanest — agents never run real logic. | ✓ |
| Patch LLM and model checkpoints only | Let agents run real code, mock get_llm() and checkpoints. Tests actual graceful-degrade code path. | |

**User's choice:** Patch agent functions at module import level

---

### What should test_graceful_degrade assert?

| Option | Description | Selected |
|--------|-------------|----------|
| Fields present + correct types only | Assert verdict_binary in ('REAL','FAKE') and verdict_label_vi in ('Thật','Giả'). | |
| Fields present + no exception raised | Assert no exception raised and result['verdict'] is a dict. | ✓ |

**User's choice:** Fields present + no exception raised

---

## Coverage scope

### How broad should Phase 8 test coverage be?

| Option | Description | Selected |
|--------|-------------|----------|
| TEST-01..06 + graph routing + output surface | Also cover route_after_verify, build_graph() fallback, _print_human(), run_fact_check() fields, MCP response. | ✓ |
| TEST-01..06 only | Strictly the 6 requirements listed in ROADMAP. | |
| TEST-01..06 + graph routing only | Cover graph routing but skip CLI/API/MCP output surface. | |

**User's choice:** TEST-01..06 + graph routing + output surface

---

### How should route_after_verify be tested?

| Option | Description | Selected |
|--------|-------------|----------|
| Unit test the function directly | Import route_after_verify, call with signal=True and signal=False, assert returned string. | ✓ |
| Test via graph conditional edge behavior | Build graph, invoke twice, assert which nodes were called. | |

**User's choice:** Unit test the function directly

---

### How should cli._print_human() be tested?

| Option | Description | Selected |
|--------|-------------|----------|
| Capture stdout with capsys | Call _print_human(), capture stdout, assert 'Thật' or 'Giả' in output. | |
| You decide | Claude picks simplest approach covering D-01 and D-02. | ✓ |

**User's choice:** You decide (Claude to use capsys, cover new format + fallback)

---

## File placement

### Where should Phase 8's new test files live?

| Option | Description | Selected |
|--------|-------------|----------|
| tests/factcheck_agents/ (keep existing convention) | New files alongside existing 9 test files. | ✓ |
| tests/ root (match ROADMAP naming) | Top-level as ROADMAP suggests. Two conventions coexist. | |

**User's choice:** tests/factcheck_agents/

---

## Claude's Discretion

- Mock return value content for agent patches in TEST-06 (minimal state dicts)
- Whether graph wiring and output surface tests are one file or two (one-file-per-concern preferred)
- capsys approach for _print_human() testing

## Deferred Ideas

None — discussion stayed within phase scope.
