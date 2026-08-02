# Phase 1: Debate Pipeline - Context

**Gathered:** 2026-08-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the complete debate-based verification pipeline: dual-source evidence retrieval (real_source_agent + fake_source_agent), BM25+PhoBERT reranker, conditional social loop (social_loop_agent), agreement gate, bounded advocate debate (debate_node), and weighted judge agent. Wire all components into `build_debate_graph()` while keeping `build_graph()` for M1 backward compat. Deliver unit + integration tests covering RERANK-02, SOCLOOP-03, and end-to-end on 2 Vietnamese claims.

**Not in scope:** Demo web app (Phase 2), adaptive convergence detection (DEBATE-EXT-01 deferred), Twitter/Facebook social search, Bayesian belief updating.

</domain>

<decisions>
## Implementation Decisions

### Reranker Backend (RERANK-01)

- **D-01:** BM25 (`rank_bm25`, new dep to add to agents extra in `pyproject.toml`) + PhoBERT CLS-token embeddings. Reuse the PhoBERT model already loaded in `verify_agent` via `lru_cache` — zero new model deps.
- **D-02:** Combine scores with a linear blend: `final_score = 0.5 × bm25_norm + 0.5 × embed_cos_sim`. Both scores normalized to [0,1] before combining.
- **D-03:** Top-k selection uses **greedy fill by token count**: iterate ranked list in order, add each snippet until adding the next would exceed PhoBERT's 256-token budget. No fixed-k shortcut.
- **D-04:** When PhoBERT model is unavailable (cold start / checkpoint missing), reranker falls back to **BM25-only** (skip embedding step). Reranker never raises an exception or blocks the pipeline.
- **D-05:** Reranker writes its top-k output back to the **existing `evidence` field** on `FactCheckState`. `verify_agent` is unchanged — still reads from `evidence`. No new field needed.

### consistency_score Definition (AGREE-02)

- **D-06:** `consistency_score` = mean cosine similarity of top-k evidence embeddings to the claim embedding, both using PhoBERT CLS pooling. This is computed as a side-effect of the reranker run — no extra inference round.
- **D-07:** When PhoBERT embeddings are unavailable (BM25-fallback mode), `consistency_score = 0.1` (AGREE-02 floor). Never inflate credibility when semantic scoring is unavailable.

### Social Loop Targets and State Field (SOCLOOP-01/02)

- **D-08:** `social_loop_agent` targets `["tiktok.com"] + settings.flagged_domains.split(",")` as `include_domains`. This is a separate node from the M1 `social_search_agent` (which targets twitter.com + facebook.com and stays in `build_graph()` only).
- **D-09:** Social loop results are stored in a **new `evidence_social: List[Evidence]` field** on `FactCheckState` (`total=False`). Results are never forced into `evidence_real` or `evidence_fake` — each source_tier classification is preserved via `classify_domain()`.
- **D-10:** Reranker processes `evidence_real ∪ evidence_fake ∪ evidence_social` as one **unified pool**. Social evidence is neutral — it is ranked purely by relevance to the claim, not by its source category.

### M2 Graph Topology (build_debate_graph)

- **D-11:** M2 **replaces** the M1 `search_agent` with a parallel fan-out of `real_source_agent` and `fake_source_agent`. `build_debate_graph()` does not include `search_agent`. M1 `build_graph()` retains `search_agent` unchanged.
- **D-12:** `real_source_agent` and `fake_source_agent` run as a **LangGraph parallel fan-out** from START — both execute concurrently and merge before the reranker node.
- **D-13:** `verify_agent` (PhoBERT + COOLANT) remains unchanged in M2. It reads from `evidence` (written by the reranker) and produces `model_results` + `reliability_signal` + model confidences needed by the agreement gate.
- **D-14:** The M1 `social_search_agent` (twitter/facebook) is **removed from `build_debate_graph()`**. M2 uses the new `social_loop_agent` (tiktok + flagged) triggered conditionally.
- **D-15:** Advocate debate stays in `debate_node` with a fixed `max_debate_rounds` cap (default 2, per DEBATE-02). Adaptive convergence detection (DEBATE-EXT-01) is deferred to v4+.

### Claude's Discretion

- Linear blend alpha (D-02): α=0.5 chosen — adjust if RERANK-02 recall@k benchmark shows BM25 dominates.
- BM25 normalization: normalize by max BM25 score in the batch (so all scores are relative, not absolute).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope & Requirements
- `.planning/ROADMAP.md` §"Phase 1: Debate Pipeline" — goals, 5-plan wave breakdown, success criteria
- `.planning/REQUIREMENTS.md` §"Evidence Retrieval", "Evidence Reranking", "Conditional Social-Search Loop", "Agreement Gate", "Debate Loop", "Weighted Judge" — all 18 Phase 1 requirements

### Existing Source Code (read before implementing any new file)
- `factcheck_agents/state.py` — `FactCheckState`, `Evidence`, `ModelResult`, `Verdict` TypedDicts to extend
- `factcheck_agents/config.py` — `Settings` dataclass pattern; new M2 env vars must follow the same `os.getenv(...)` field pattern
- `factcheck_agents/graph.py` — `build_graph()` topology to keep; `route_after_verify` as pattern for new routing functions
- `factcheck_agents/agents/verify_agent.py` — `ThreadPoolExecutor` concurrent pattern + `lru_cache` model singletons; reuse for parallel source agents
- `factcheck_agents/source_tier.py` — `classify_domain(url)` pure function; used by source agents and social_loop_agent for tier tagging
- `factcheck_agents/agents/search_agent.py` — query generation pattern + `_fetch_evidence_image()` import; real/fake source agents follow the same web_search call pattern
- `factcheck_agents/helpers.py` — `_fetch_evidence_image()` shared utility; new source agents should import from here
- `factcheck_agents/agents/conclusion_agent.py` — existing conclusion agent; judge_agent replaces it in M2 graph but must remain for M1 compat
- `factcheck_agents/agents/__init__.py` — export pattern; all new M2 agents must be exported here

### New Dependency
- `rank_bm25>=0.2.2` — add to `pyproject.toml` agents extra (BM25 implementation for reranker)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `verify_agent.ThreadPoolExecutor(max_workers=2)` pattern → reuse for fan-out of `real_source_agent` + `fake_source_agent` in a single parent node, or use LangGraph's native fan-out
- `lru_cache(maxsize=1)` model singletons (`_phobert()`, `_coolant()`) → PhoBERT is already loaded; reranker can call `_phobert()` to get embeddings without reloading
- `search_agent.web_search(q, max_results=N, include_domains=[...])` pattern → exact pattern for real/fake/social source agents
- `classify_domain(url)` → already handles trusted/flagged/social/unknown; all new source agents must call this on each result
- `config.py` env-var pattern → add `agreement_threshold`, `max_debate_rounds`, `google_factcheck_api_key`, `social_loop_min_count`, `social_loop_min_credibility`

### Established Patterns
- `total=False` on all TypedDicts → all new M2 state fields (`evidence_real`, `evidence_fake`, `evidence_social`, `social_loop_fired`, `request_id`, `agreement_score`, `debate_turns`, `debate_exit_reason`, `weight_breakdown`) must use `total=False`
- Graceful degrade: every model/tool failure returns a fallback result, never raises → apply to reranker (BM25 fallback), source agents (empty list fallback), social loop (skip if search fails)
- Atomic file writes: use `open(path, 'a')` + `json.dumps(..., ensure_ascii=False) + '\n'` for JSONL debate logs; use `open(path, 'w')` + `json.dump(...)` for JSON verdict logs; both in `logs/` directory
- `request_id`: UUID4 generated at `initial_state()` call time; stored on `FactCheckState`; used as filename stem for `logs/debates/<request_id>.jsonl` and `logs/verdicts/<request_id>.json`

### Integration Points
- Reranker output → writes to `state["evidence"]` → consumed by `verify_agent` (unchanged)
- `verify_agent` outputs → `model_results` + `reliability_signal` → read by `agreement_gate` for confidence scores
- `agreement_gate` → conditional routing: if `agreement_score >= threshold` → skip debate, route to `judge_agent`; else route to `debate_node`
- `debate_node` → writes `debate_turns: List[dict]` and `debate_exit_reason: str` to state
- `judge_agent` → writes `verdict: Verdict` to state (replaces M1 `conclusion_agent` in M2 graph)
- `build_debate_graph()` M2 topology: `START → [real_source | fake_source] (fan-out) → reranker → [social_loop?] → verify → agreement_gate → [debate_node?] → judge → END`

</code_context>

<specifics>
## Specific Ideas

- Real advocate cites only `evidence_real`; fake advocate cites only `evidence_fake` (DEBATE-01). Each round, the active advocate must quote + counter the opponent's previous argument.
- Google Fact Check API (`claims.search?languageCode=vi`) is stubbed when `GOOGLE_FACTCHECK_API_KEY` is unset — `fake_source_agent` returns only tingia.gov.vn results in that case (EVRET-02).
- `logs/debates/` and `logs/verdicts/` directories must be created if they don't exist (use `Path.mkdir(parents=True, exist_ok=True)`).
- EVRET-04 gate: if both `evidence_real` and `evidence_fake` are empty after retrieval, short-circuit to NEI verdict immediately (skip reranker, social loop, verify, debate). Add as a conditional edge after the fan-out merge.

</specifics>

<deferred>
## Deferred Ideas

- **DEBATE-EXT-01** (adaptive termination via Wald-SPRT) — deferred to v4+; debate_node uses fixed max_rounds per DEBATE-02
- **Twitter/Facebook social search in M2** — M1 `social_search_agent` kept in `build_graph()` only; not included in M2 `build_debate_graph()`
- **Bayesian belief updating in agreement gate** — out of scope per REQUIREMENTS.md

</deferred>

---

*Phase: 1-debate-pipeline*
*Context gathered: 2026-08-03*
