---
phase: '04'
slug: langgraph-a2a-client-wiring
status: verified
threats_open: 0
asvs_level: 2
created: 2026-08-17
---

# Phase 04 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| B1 User → CLI/graph | Local user input to `initial_state()` | `statement`, `image_path`, `language` (cli.py:68-71) |
| B2 LangGraph client ↔ A2A servers (9001-9010) | Plaintext HTTP over loopback, no auth | Full serialized `FactCheckState` (client→server); state diff dict (server→client) — a2a_client.py:72-110, a2a_server.py:176-193 |
| B3 Agent servers ↔ internet | Tavily / Google CSE / Google FactCheck / OpenAI / crawled article pages+images | Queries, evidence text, API keys (POST bodies and GET query strings) — web_search.py:22,67; fake_source_agent.py:90; llm.py:23; helpers.py:43,118 |
| B4 State ↔ checkpointer | SqliteSaver on disk (`.factcheck_checkpoints.db`) | Full state incl. `EvidenceGraph` via `_asdict()`/`graph_data` — graph.py:69-75, graph_utils.py:10-29 |
| B5 State → local filesystem | logs/debates/*, logs/expert/*.json, logs/agent_*.log | Debate turns, verdict JSON, server logs — graph.py:125, expert_agent.py:252-266 |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-01 | Tampering | a2a_client wrappers / graph state merge | accept | Accepted risk A-01 — unvalidated server diff merged into graph state | closed |
| D-04 | DoS | a2a_client response parsing | accept | Accepted risk A-02 — malformed server responses crash pipeline (fail-fast) | closed |
| P-01 | Injection | search/conclusion/expert prompts | accept | Accepted risk A-03 — prompt injection via untrusted statement/web evidence | closed |
| T-02 | Tampering/Elevation | conclusion_agent EvidenceGraph repr rebuild | mitigate | Repr string discarded; graph rebuilt from evidence list only (conclusion_agent.py:149-152) | closed |
| T-03 | Tampering | EvidenceGraph checkpoint round-trip | mitigate | `graph_data` is plain dicts/tuples → networkx `add_node(**attrs)`/`add_edge`; no dangerous sink (graph_utils.py:10-29) | closed |
| S-01 | Spoofing | A2A endpoints (no auth) | mitigate | All 10 servers bind `127.0.0.1`; client pins `localhost`; port-conflict = hard abort (a2a_server.py:280; a2a_client.py:83; scripts/start_agents.sh:42-45) | closed |
| S-02 | Spoofing | Client target selection | mitigate | Host hardcoded `http://localhost:{port}`; ports from trusted operator env config (a2a_client.py:83; config.py:129-158, 224-237) | closed |
| I-01 | Information disclosure | Secrets in state/logs | mitigate | Keys never enter state (state.py:60-150 has no key fields); passed only to HTTP clients inside swallowing try/except; zero logger calls in agents/; client logs agent-name/port/cause only (web_search.py:22,38,67,76; fake_source_agent.py:85-143; llm.py:23; a2a_client.py:103-107,122-127) | closed |
| I-02 | Information disclosure | News text / LLM outputs in logs | mitigate | No content logging (agents have no loggers; a2a_client logs metadata only); expert verdict JSON on disk is by-design (verdicts+citations, no raw news text); state plaintext on loopback only (a2a_client.py:122-127; expert_agent.py:252-266) | closed |
| R-01 | Repudiation | Audit trail | mitigate | Degrade warnings per-agent with port+cause, WORKING warnings, debate degrade warnings, server-side `logger.exception`, debate transcripts persisted (a2a_client.py:103-107,122-127; graph.py:144,164; a2a_server.py:189; graph.py:125) | closed |
| D-01 | DoS | Per-agent timeouts | mitigate | `httpx.Client(timeout=...)` with per-agent defaults (LLM 120s / evaluate 60s / calc 30s), global override; no retry (fail-fast) (a2a_client.py:70-89, 44-49; config.py:163-201) | closed |
| D-02 | DoS | TASK_STATE_WORKING | mitigate | Single-shot POST; WORKING → warning + `{}`, no poll loop (a2a_client.py:102-108) | closed |
| D-03 | DoS | Debate loop / degrade paths | mitigate | `max_debate_rounds=10` (config.py:98-100); both-down break (graph.py:176-178); degrade exits after round 1; `social_loop_fired=True` prevents re-entry (graph.py:134,176-182; a2a_client.py:168-172; social_loop_agent.py:61-66) | closed |
| E-01 | Elevation | Deserialized payloads / LLM output | mitigate | No eval/exec/pickle/yaml/subprocess sinks (only torch `.eval()`); `parse_json` uses `json.loads` only; server-side `_extract_state` requires dict-shaped payload, failures contained as failed tasks (llm.py:27-41; a2a_server.py:205-221, 188-193) | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| A-01 | T-01 | Server diffs originate from first-party A2A agents on loopback; no external attacker can inject state. Residual: local compromise can corrupt pipeline state. Validation deferred. | User | 2026-08-17 |
| A-02 | D-04 | Malformed responses only from first-party servers; failure mode is fail-fast crash, not silent corruption. Residual: availability impact if servers misbehave. | User | 2026-08-17 |
| A-03 | P-01 | News statement and web evidence are inherently untrusted; verdict steering via crafted evidence is possible. Partial label canonicalization exists (state.py:40-57) but is not an injection defense. Mitigation deferred to a future phase. | User | 2026-08-17 |

*Accepted risks do not resurface in future audit runs.*

---

## Unregistered Flags

Informational only — new attack surface observed during retroactive audit, no threat mapping at plan time (phase authored before formal threat modelling):

1. **No request-size/body limits + unbounded `InMemoryTaskStore`** (a2a_server.py:247): any local process can POST huge payloads / flood tasks → memory DoS of agent servers (loopback-only, unauthenticated).
2. **No server-side execution timeout**: `agent_fn` runs in `asyncio.to_thread` (a2a_server.py:230) with no cancellation on client disconnect — orphaned LLM/web calls continue after client timeout (resource/quota burn).
3. **`A2A_CLIENT_TIMEOUT` not cast at load** (config.py:163-165): annotation `Optional[int]` but value stays `str`; garbage env value → `ValueError` at first call instead of fallback; `0` silently busts httpx timeout semantics (operator misconfig).
4. **D-05 partial debate is dead code in production**: wrappers swallow `AgentUnavailableError` and return the degrade diff, so `debate_node`'s `except AgentUnavailableError` branches (graph.py:141,161) never fire; loop exits round 1 with `exit_reason="llm_error"` instead of `"agent_unavailable"` — mislabeled audit trail, one-sided debates never actually run.
5. **CLI `thread_id` from `hash(statement)`** (cli.py:70): randomized per process (PYTHONHASHSEED) → new checkpoint thread per run → SqliteSaver DB growth; run identity non-reproducible.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-08-17 | 14 | 14 | 0 | gsd-security-auditor (retroactive-STRIDE) via /gsd-secure-phase 04 |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-08-17
