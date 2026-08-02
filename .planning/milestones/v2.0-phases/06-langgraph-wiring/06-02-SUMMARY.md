# Plan 06-02 Summary: Add checkpoint DB config and wire checkpointer

## Objective
Add `FACTCHECK_CHECKPOINT_DB` env var to config.py and wire a SqliteSaver/MemorySaver checkpointer into `build_graph()` so interrupted pipeline runs resume from the last completed node instead of restarting from search.

## Changes Made

### factcheck_agents/config.py
1. Added `checkpoint_db` field to the `Settings` dataclass:
   - Type: `str` (not `Path` - SqliteSaver accepts connection string)
   - Default: `.factcheck_checkpoints.db`
   - Env var: `FACTCHECK_CHECKPOINT_DB`
   - Placed in new `# ── LangGraph checkpoint ─────────────────────────────────────────────────` comment block

### .gitignore
1. Added `.factcheck_checkpoints.db` to prevent checkpoint DB from being committed

### factcheck_agents/graph.py
1. Added import: `from .config import settings`
2. Changed `build_graph()` signature to `build_graph(checkpointer=None)` for backward compatibility
3. Added default checkpointer resolution at start of `build_graph()`:
   - Try `SqliteSaver.from_conn_string(settings.checkpoint_db)` when langgraph[sqlite] extra is available
   - Fall back to `MemorySaver()` on ImportError (silent fallback)
4. Changed `return g.compile()` to `return g.compile(checkpointer=checkpointer)`
5. Added Phase 7 note comment about thread_id wiring for future work
6. `initial_state()` function unchanged (thread_id lives in invoke config, not state)

## Verification Results
All verification commands passed:
- `settings.checkpoint_db` returns `.factcheck_checkpoints.db` by default
- `FACTCHECK_CHECKPOINT_DB=/tmp/x.db` overrides the default
- `build_graph(checkpointer=MemorySaver())` accepts explicit checkpointer for tests
- `build_graph()` default checkpointer works (SqliteSaver or MemorySaver fallback)
- `initial_state()` unchanged
- Graph compiles without errors

## Requirements Satisfied
- GRAPH-04: `build_graph()` compiles with a LangGraph checkpointer attached
- Additional: SqliteSaver used when `langgraph[sqlite]` extra is available; MemorySaver is the silent fallback
- Additional: `build_graph(checkpointer=MemorySaver())` works for test callers passing explicit checkpointer
- Additional: Existing callers (`run_fact_check()`, cli.py, mcp_server.py) pass no args and are unaffected
- Additional: `g.invoke(state)` without a `config` dict does not crash when checkpointer is compiled in
- Additional: `FACTCHECK_CHECKPOINT_DB` env var controls the SqliteSaver file path; default is `.factcheck_checkpoints.db`

## Artifacts
- `factcheck_agents/config.py` - checkpoint_db field in Settings
- `factcheck_agents/graph.py` - build_graph(checkpointer=None) with checkpointer wiring
- `.gitignore` - .factcheck_checkpoints.db entry

## Commit
Commit hash: 272100f

## Status
✅ Complete
