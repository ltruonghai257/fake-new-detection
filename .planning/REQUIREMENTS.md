# Requirements: factcheck_agents v3.1

**Defined:** 2026-08-13 (v3.1 — A2A Protocol Integration; v3.0 shipped 2026-08-03)
**Core Value:** A user submits a Vietnamese claim and gets back a weighted, debate-tested verdict — with live streaming debate transcript and full audit log — even when model checkpoints are missing.

## v3.1 Requirements

### A2A SDK & Agent Handlers (Phase 1)

-   [ ] **A2A-01**: `a2a-sdk[http-server,fastapi]` (≥ 1.1.0) added to `factcheck_agents/` dependencies; each of the 10 agent modules (`search_agent`, `evaluate_agent`, `real_source_agent`, `fake_source_agent`, `social_loop_agent`, `agreement_gate`, `real_advocate`, `fake_advocate`, `judge_agent`, `conclusion_agent`) gains a `TaskHandler` class that accepts an A2A `Task` object and returns an A2A `TaskResult`
-   [ ] **A2A-02**: Each agent's uvicorn app exposes `GET /.well-known/agent.json` containing its A2A Agent Card: `name`, `description`, `version`, `skills` list (one skill per agent), and `url` matching its assigned port; all fields conform to the A2A Agent Card JSON schema

### Agent Launch Scripts (Phase 1)

-   [ ] **A2A-03**: `scripts/start_agents.sh` starts all 10 uvicorn processes in the background, one per port (search=9001, evaluate=9002, real_source=9003, fake_source=9004, social_loop=9005, agreement_gate=9006, real_advocate=9007, fake_advocate=9008, judge=9009, conclusion=9010); writes a PID file (`scripts/.agent_pids`) for clean shutdown
-   [ ] **A2A-03b**: `scripts/stop_agents.sh` reads the PID file and sends `SIGTERM` to each process; handles missing PIDs gracefully; exits 0

### A2A Client Module (Phase 2)

-   [ ] **A2A-04**: `factcheck_agents/a2a_client.py` implements typed wrapper functions (one per agent) that construct an A2A `Task` message, call the appropriate local port via `httpx`, and deserialize the `TaskResult` back to the existing Python types used by the LangGraph state (`Evidence`, `VerifyResult`, `DebateTurn`, etc.); timeouts and connection errors propagate as `AgentUnavailableError`

### LangGraph Graph Refactor (Phase 2)

-   [ ] **A2A-05**: `factcheck_agents/graph.py` — all 10 node functions updated to call `a2a_client.*` functions instead of importing agent functions directly; conditional routing edges (`social_loop_router`, `agreement_router`, `nei_gate`) remain unchanged; `build_graph()` and `build_debate_graph()` signatures and return types unchanged
-   [ ] **A2A-05b**: `AgentUnavailableError` caught at each node and mapped to the existing graceful-degrade path (same behaviour as a missing model checkpoint in v3.0)

### Demo App SSE Bridge Update (Phase 3)

-   [ ] **A2A-06**: `demo_app/backend/streaming.py` updated to call A2A agent HTTP endpoints via `a2a_client`; the SSE event schema (`stage_start`, `turn_start`, `chunk`, `turn_end`, `verdict`, `heartbeat`) is **unchanged** — the React frontend requires no changes
-   [ ] **A2A-06b**: If an A2A agent is unreachable when the demo app calls it, the SSE stream emits a `stage_error` event with a Vietnamese-language error message and closes gracefully (no 500)

### Test Updates (Phase 3)

-   [ ] **A2A-07**: Each of the 10 agent `TaskHandler`s is unit-tested by spinning up the uvicorn server in-process (via `pytest-anyio` or `asyncio` loop), sending a real A2A `Task`, and asserting the `TaskResult` schema; existing Python-level unit tests (non-HTTP) are retained for regression
-   [ ] **A2A-07b**: Existing graph integration tests (2 sample Vietnamese claims end-to-end) are updated to start agent servers before the test session (session-scoped fixture) and tear down after; total test run still passes in < 60 s on developer hardware

### Backward Compatibility (Phase 3)

-   [ ] **A2A-08**: `factcheck_agents/cli.py` — no changes; `run_fact_check()` in `__init__.py` — no signature or return-type changes; `mcp_server.py` — no changes; external callers of v3.0 remain unaffected

---

## v3.0 Requirements (Shipped 2026-08-03)

### Evidence Retrieval (Phase 1)

-   [ ] **EVRET-01**: `real_source_agent` searches credible Vietnamese outlets only (vnexpress.net, tuoitre.vn, thanhnien.vn, ttxvn.gov.vn, vtv.vn, dantri.com.vn); results written to `state["evidence_real"]` as `List[Evidence]`
-   [ ] **EVRET-02**: `fake_source_agent` searches tingia.gov.vn and Google Fact Check Tools API (`claims.search`, `languageCode=vi`); Google Fact Check API is stubbed (returns empty list) when `GOOGLE_FACTCHECK_API_KEY` is unset; real calls gated on key being provided
-   [ ] **EVRET-03**: Outputs stay as two separate typed lists (`evidence_real`, `evidence_fake`) on `FactCheckState` — never merged into a boolean flag or single list
-   [ ] **EVRET-04**: If both `evidence_real` and `evidence_fake` are empty after retrieval, gate immediately to NEI verdict with `confidence=0.0` and skip model inference + debate

### Evidence Reranking (Phase 1)

-   [ ] **RERANK-01**: Before truncating evidence to PhoBERT's 256-token budget, rerank all snippets from `evidence_real` ∪ `evidence_fake` by relevance to the claim using BM25 + embedding rerank (or cross-encoder if already available in repo); select the top-k snippets that maximize recall within the token budget
-   [ ] **RERANK-02**: Unit test asserts recall@k on a small labeled sample (≥ 5 claim–snippet pairs) stays within an acceptable range of the target; target set to actual achievable value after measuring on real data (documented in test as a comment, not hardcoded as 93%)

### Conditional Social-Search Loop (Phase 1)

-   [ ] **SOCLOOP-01**: A conditional edge between evidence retrieval and model verification fires a one-shot supplemental social search (TikTok FactCheckVN + previously flagged pages via Tavily/Google CSE) when: `len(evidence_real) + len(evidence_fake) < FACTCHECK_SOCIAL_LOOP_MIN_COUNT` (default 3) **AND** `evidence_credibility_score < FACTCHECK_SOCIAL_LOOP_MIN_CREDIBILITY` (default 0.4); both thresholds are configurable env vars
-   [ ] **SOCLOOP-02**: `social_loop_fired: bool` field on `FactCheckState` is set to `True` when the loop runs; the routing function checks this flag before the evidence credibility check — if `True`, skip directly to model verification regardless of evidence weakness
-   [ ] **SOCLOOP-03**: Unit test asserts that when `social_loop_fired=True` on input state, the social loop node is never reached a second time (routing short-circuits to `verify`)

### Agreement Gate (Phase 1)

-   [ ] **AGREE-01**: After `verify_agent` runs PhoBERT + COOLANT, `agreement_gate` computes an agreement score: `0.30 × phobert_confidence + 0.30 × coolant_confidence + 0.40 × evidence_credibility`; unavailable model → treat confidence as 0.0 and normalize over available signals; NEI from any model → force `agreement_score = 0.0`
-   [ ] **AGREE-02**: Evidence-credibility component computed as `0.40 × tier_score + 0.30 × count_score + 0.30 × consistency_score`; never collapses to 0 (floor at 0.1); all three sub-components logged to `logs/verdicts/<request_id>.json`
-   [ ] **AGREE-03**: If `agreement_score ≥ FACTCHECK_AGREEMENT_THRESHOLD` (default 0.8), skip debate and route directly to judge; log `debate_exit_reason = "skipped_high_agreement"` to state

### Debate Loop (Phase 1)

-   [ ] **DEBATE-01**: `real_advocate` LLM agent argues for "Real" verdict, citing only `evidence_real`; `fake_advocate` LLM agent argues for "Fake" verdict, citing only `evidence_fake`; each round, the active advocate explicitly rebuts the opponent's previous argument (must quote + counter)
-   [ ] **DEBATE-02**: Debate loop is implemented inside a single LangGraph node (`debate_node`); max rounds hard-capped by `FACTCHECK_MAX_DEBATE_ROUNDS` (default 2) Python `for` loop — never delegated to the LLM
-   [ ] **DEBATE-03**: Every debate turn (agent name, round number, ISO timestamp, full argument text) is printed to stdout **and** appended atomically to `logs/debates/<request_id>.jsonl` (one JSON object per line, `ensure_ascii=False`); the skip case from AGREE-03 also writes a single `{"debate_skipped": true, ...}` line to the same file

### Weighted Judge (Phase 1)

-   [ ] **JUDGE-01**: After debate (or skip), `judge_agent` scores each debate argument on three 1–5 dimensions: Factuality (claims grounded in cited evidence), Rebuttal Engagement (directly addresses opponent's points), Evidence Grounding (citations from appropriate tier); scored via a single structured-output LLM call per turn
-   [ ] **JUDGE-02**: Final verdict combines PhoBERT confidence (30%) + COOLANT confidence (30%) + evidence-credibility (40%); when debate was skipped, judge runs directly from model results + evidence without argument scores; judge confidence capped at 0.7 when debate was skipped (no argument quality signal)
-   [ ] **JUDGE-03**: Final output: `{verdict: "Real"|"Fake"|"NEI", confidence: float, explanation: str, weight_breakdown: {phobert: float, coolant: float, evidence: float, argument_scores: {...}}}`; full breakdown written atomically to `logs/verdicts/<request_id>.json`

### Demo Web App (Phase 2)

-   [ ] **DEMO-01**: New `demo_app/` directory at project root containing FastAPI backend (`demo_app/backend/`) and React/Vite/TypeScript frontend (`demo_app/frontend/`); backend imports `factcheck_agents` directly (no subprocess)
-   [ ] **DEMO-02**: `POST /api/analyze` accepts `{statement: str, image_path?: str}`; response is a streaming SSE endpoint; SSE event types: `stage_start`, `turn_start`, `chunk`, `turn_end`, `verdict`, `heartbeat` (every 5s); client disconnect detected and loop aborted
-   [ ] **DEMO-03**: Debate stage streams turn-by-turn live — character-level chunking, alternating chat bubbles (real_advocate / fake_advocate); final verdict card shows label, confidence gauge, 30/30/40 weight breakdown bar, and download buttons for `logs/debates/<id>.jsonl` and `logs/verdicts/<id>.json`
-   [ ] **DEMO-04**: All UI copy in Vietnamese; app is local-only (no auth, no public deployment, CORS allows only `localhost:5173`)

## v4+ Requirements (deferred)

### Extended Debate

-   **DEBATE-EXT-01**: Adaptive termination via Wald-SPRT convergence detection (instead of fixed max_rounds)
-   **DEBATE-EXT-02**: Multi-round evidence exhaustion check (stop if no new evidence cited)

### Cross-Platform Evidence

-   **SOC-01**: Paid X/Twitter or Meta Graph API integration — explicit confirmation required before adding
-   **SOC-02**: Real-time streaming social media monitoring

### Multilingual

-   **LANG-01**: Auto-detect statement language and match debate + verdict language

## Out of Scope

| Feature                                                   | Reason                                                     |
| --------------------------------------------------------- | ---------------------------------------------------------- |
| Modifying PhoBERT/COOLANT weights, architecture, training | Model-level changes out of scope; only consumption changes |
| Removing existing source-tier system                      | Extended for evidence-credibility; not replaced            |
| Public deployment of demo app                             | Thesis demo only; no hosting, no auth                      |
| Paid X/Twitter or Meta Graph API                          | Stop condition — requires explicit confirmation            |
| WebSockets for demo streaming                             | SSE is sufficient; WebSockets add complexity for no gain   |
| Bayesian belief updating in agreement gate                | Over-engineering; no calibration data                      |
| User participation in debate                              | Breaks automated verification flow                         |

## Traceability

### v3.0 (Shipped)

| Requirement | Phase   | Status  |
| ----------- | ------- | ------- |
| EVRET-01    | Phase 1 | Shipped |
| EVRET-02    | Phase 1 | Shipped |
| EVRET-03    | Phase 1 | Shipped |
| EVRET-04    | Phase 1 | Shipped |
| RERANK-01   | Phase 1 | Shipped |
| RERANK-02   | Phase 1 | Shipped |
| SOCLOOP-01  | Phase 1 | Shipped |
| SOCLOOP-02  | Phase 1 | Shipped |
| SOCLOOP-03  | Phase 1 | Shipped |
| AGREE-01    | Phase 1 | Shipped |
| AGREE-02    | Phase 1 | Shipped |
| AGREE-03    | Phase 1 | Shipped |
| DEBATE-01   | Phase 1 | Shipped |
| DEBATE-02   | Phase 1 | Shipped |
| DEBATE-03   | Phase 1 | Shipped |
| JUDGE-01    | Phase 1 | Shipped |
| JUDGE-02    | Phase 1 | Shipped |
| JUDGE-03    | Phase 1 | Shipped |
| DEMO-01     | Phase 2 | Shipped |
| DEMO-02     | Phase 2 | Shipped |
| DEMO-03     | Phase 2 | Shipped |
| DEMO-04     | Phase 2 | Shipped |

### v3.1 (Active)

| Requirement | Phase   | Status  |
| ----------- | ------- | ------- |
| A2A-01      | Phase 1 | Pending |
| A2A-02      | Phase 1 | Pending |
| A2A-03      | Phase 1 | Pending |
| A2A-03b     | Phase 1 | Pending |
| A2A-04      | Phase 2 | Pending |
| A2A-05      | Phase 2 | Pending |
| A2A-05b     | Phase 2 | Pending |
| A2A-06      | Phase 3 | Pending |
| A2A-06b     | Phase 3 | Pending |
| A2A-07      | Phase 3 | Pending |
| A2A-07b     | Phase 3 | Pending |
| A2A-08      | Phase 3 | Pending |

**Coverage:**

-   v3.1 requirements: 12 total
-   Mapped to phases: 12
-   Unmapped: 0 ✓

---

_Requirements defined: 2026-08-02_
_Last updated: 2026-08-02 — initial M2 definition including REQ 6 (RERANK) and REQ 7 (SOCLOOP)_
