# Phase 4: LangGraph → A2A Client Wiring - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-16
**Phase:** 4-LangGraph → A2A Client Wiring
**Areas discussed:** Async bridge, evidence_graph transport, Degrade markers, Timeout/retry, Graph test strategy

---

## Async Bridge

| Option | Description | Selected |
|--------|-------------|----------|
| Sync wrappers, keep invoke | a2a_client exposes sync functions; zero caller changes; demo streaming untouched until Phase 5 | ✓ |
| Async nodes + ainvoke | Graph nodes become async; entry points wrapped with asyncio.run; breaks demo streaming until Phase 5 | |
| Async core, sync shims | Async client functions + thin sync wrappers for graph nodes | |

**User's choice:** Sync wrappers, keep invoke
**Notes:** All entry points (run_fact_check, cli, mcp, demo streaming) use sync `.invoke()`. Keeping sync avoids touching any caller.

---

## evidence_graph Transport

| Option | Description | Selected |
|--------|-------------|----------|
| Rebuild locally in client | Client wrapper reconstructs EvidenceGraph from returned evidence list | ✓ |
| Drop from wire payload | Handlers omit evidence_graph; client reconstructs locally | |
| Serialize it structurally | Dump/restore EvidenceGraph as adjacency dict across the wire | |
| Defensive consumer only | Guard conclusion_agent to treat non-EvidenceGraph as no-conflict | |

**User's choice:** "You decide all" — Claude selected rebuild locally
**Notes:** `conclusion_agent` calls `evidence_graph.graph.nodes(...)` unguarded (line 179). A repr-string from serialization would raise. Rebuilding locally in the client wrapper is the cleanest fix — no handler or serialization changes.

---

## Degrade Markers

| Option | Description | Selected |
|--------|-------------|----------|
| Shared degrade helper | One helper returns per-agent degrade diff; nodes catch and return it | |
| Per-node hand-rolled diffs | Each node writes its own degrade return | |
| Add availability field | Shared helper + state field recording which agents degraded | |

**User's choice:** Shared degrade helper — refined to `@degrade_on_unavailable` decorator

**Follow-up — Partial debate:**
| Option | Description | Selected |
|--------|-------------|----------|
| Abort, route to judge | Either advocate down → exit_reason='agent_unavailable', judge scores without arguments | |
| Partial debate | One-sided turns if only one advocate is available; judge gets partial argument scores | ✓ |

**User's choice:** Partial debate
**Notes:** Available advocate still argues; unavailable side turns are skipped. If both down, exit_reason='agent_unavailable' and judge runs without argument quality signal.

---

## Timeout / Retry

| Option | Description | Selected |
|--------|-------------|----------|
| 120s base, 30s fast | Base 120s (LLM agents), fast 30s (agreement_gate, social_loop); A2A_CLIENT_TIMEOUT override | |
| 90s uniform | Single 90s timeout for all 10 agents | |
| You decide on per-agent | Claude picks reasonable values per agent type | ✓ |

**User's choice:** "You decide" — Claude defined per-agent values
**Notes:** LLM agents (search, real_source, fake_source, advocates, judge, conclusion) = 120s; agreement_gate = 30s; social_loop = 30s; evaluate_agent = 60s. Config: `A2A_CLIENT_TIMEOUT` global override + `A2A_CLIENT_TIMEOUT_<NAME>` per-agent. No automatic retry.

---

## Graph Test Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Update patch targets | Edit 2 test files to patch a2a_client.* methods instead of graph.* imports | ✓ |
| Keep aliases in graph.py | Retain module-level aliases so existing patches still resolve | |
| Defer graph tests to Phase 5 | Skip graph-level tests in Phase 4; add session-scoped fixtures in Phase 5 (A2A-07b) | |

**User's choice:** Update patch targets
**Notes:** `test_graceful_degrade.py` and `test_debate_pipeline_integration.py` patches change from `factcheck_agents.graph.<agent>` to `factcheck_agents.a2a_client.<agent>`. Minimal edits (~10 lines each).

---

## Claude's Discretion

- **evidence_graph rebuild method:** Local client rebuild (EvidenceGraph.build_from_evidence) — no handler or serialization changes
- **Per-agent timeout values:** Based on agent type — LLM agents 120s, calculation-only 30s, model-inference 60s
- **messages tuple→list normalization:** Client normalizes on return; lists unpack identically to tuples
- **evaluate_agent wrapper:** Exists for API completeness (port 9002) even though no graph node calls it

## Deferred Ideas

None — discussion stayed within phase scope.
