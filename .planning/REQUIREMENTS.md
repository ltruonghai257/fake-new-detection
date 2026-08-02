# Requirements: factcheck_agents v3.0

**Defined:** 2026-08-02
**Core Value:** A user submits a Vietnamese claim and gets back a weighted, debate-tested verdict — with live streaming debate transcript and full audit log — even when model checkpoints are missing.

## v3.0 Requirements

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

| Requirement | Phase   | Status  |
| ----------- | ------- | ------- |
| EVRET-01    | Phase 1 | Pending |
| EVRET-02    | Phase 1 | Pending |
| EVRET-03    | Phase 1 | Pending |
| EVRET-04    | Phase 1 | Pending |
| RERANK-01   | Phase 1 | Pending |
| RERANK-02   | Phase 1 | Pending |
| SOCLOOP-01  | Phase 1 | Pending |
| SOCLOOP-02  | Phase 1 | Pending |
| SOCLOOP-03  | Phase 1 | Pending |
| AGREE-01    | Phase 1 | Pending |
| AGREE-02    | Phase 1 | Pending |
| AGREE-03    | Phase 1 | Pending |
| DEBATE-01   | Phase 1 | Pending |
| DEBATE-02   | Phase 1 | Pending |
| DEBATE-03   | Phase 1 | Pending |
| JUDGE-01    | Phase 1 | Pending |
| JUDGE-02    | Phase 1 | Pending |
| JUDGE-03    | Phase 1 | Pending |
| DEMO-01     | Phase 2 | Pending |
| DEMO-02     | Phase 2 | Pending |
| DEMO-03     | Phase 2 | Pending |
| DEMO-04     | Phase 2 | Pending |

**Coverage:**

-   v3.0 requirements: 22 total
-   Mapped to phases: 22
-   Unmapped: 0 ✓

---

_Requirements defined: 2026-08-02_
_Last updated: 2026-08-02 — initial M2 definition including REQ 6 (RERANK) and REQ 7 (SOCLOOP)_
