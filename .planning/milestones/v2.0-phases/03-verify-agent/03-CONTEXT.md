# Phase 3: Verify Agent - Context

**Gathered:** 2026-07-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Create `factcheck_agents/agents/verify_agent.py` that replaces `evaluate_agent` — runs PhoBERT ViFactCheck and COOLANT concurrently via `ThreadPoolExecutor`, computes a weighted `reliability_signal: bool`, and writes both `reliability_signal` and `model_results` to state. Update `factcheck_agents/models/phobert_checker.py` `build_evidence_text()` to tier-sort evidence before building the context string. No social search, no verdict logic, no `graph.py` changes (those are Phase 6).

</domain>

<decisions>
## Implementation Decisions

### reliability_signal Fusion Logic
- **D-01:** Use **weighted confidence magnitudes** (not label-converted scores). Weights: 0.6 × PhoBERT confidence + 0.4 × COOLANT confidence. Label direction is ignored for signal computation — high confidence from either model is treated as reliable evidence worth pursuing with social search.
- **D-02:** When one model is unavailable (COOLANT has no image, or either checkpoint missing), **normalize remaining weight to 1.0** — the available model carries full weight. Example: COOLANT unavailable → `reliability_signal = phobert_conf ≥ threshold`.
- **D-03:** **NEI label always → False**, regardless of confidence. If the only/best available model returns NEI, `reliability_signal = False`. NEI means "can't determine" — not reliable enough to warrant social search.
- **D-04:** Both models unavailable → `reliability_signal = False`.
- **D-05:** Signal formula: `weighted_score = sum(w_i × conf_i for available models) / sum(w_i for available models)` where `w_phobert=0.6, w_coolant=0.4`. Then `reliability_signal = (weighted_score >= settings.reliability_threshold) AND (no available model returned NEI)`.

### build_evidence_text() Tier Ordering
- **D-06:** Add tier-sort **inside `build_evidence_text()`** — sort by `source_tier` key before iterating. Tier order: `{"trusted": 0, "flagged": 1, "social": 2, "unknown": 3}`. Items missing `source_tier` default to `"unknown"`. No signature change — still `build_evidence_text(evidence: List[dict], max_chars: int = 2000) -> str`.
- **D-07:** After sorting, truncate to `max_chars` as before (simple fill-and-cut). No per-tier budget. Trusted snippets naturally appear first and get priority due to sort order.

### COOLANT Image Resolution
- **D-08:** `verify_agent` replicates `evaluate_agent`'s image fallback — if `state.get("image_path")` is None, scan evidence list for `e.get("image_path")` and use first match. Preserves v1.0 behavior where search-agent-fetched news images can drive COOLANT.

### Claude's Discretion
- `ThreadPoolExecutor` usage: create per-call executor with `max_workers=2`; no persistent executor needed since model singletons (`lru_cache`) are already shared across calls.
- Exception isolation: if one model raises inside `ThreadPoolExecutor`, catch and convert to `unavailable` ModelResult — never propagate. Consistent with existing graceful-degrade pattern in `evaluate_agent`.
- `evaluate_agent.py` lifecycle: leave intact through Phase 3 — Phase 6 wires `verify_agent` into `graph.py`. No deprecation or deletion in this phase.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Scope
- `.planning/REQUIREMENTS.md` §Verify Agent (VERIFY-01, VERIFY-02, VERIFY-03) — exact concurrency, signal formula, and state-write requirements
- `.planning/REQUIREMENTS.md` §Tests (TEST-03, TEST-06 partial) — what tests Phase 3 must enable (signal computation, graceful degrade with no checkpoints)

### Phase 1 Context (locked decisions this phase depends on)
- `.planning/phases/01-state-config-evidence-graph-foundation/01-CONTEXT.md` — D-09 (`reliability_signal: Optional[bool]` on `FactCheckState`), D-07 (`source_tier` Literal types), D-10 (`reliability_signal` is a top-level state field)

### Phase 2 Context (evidence structure this phase consumes)
- `.planning/phases/02-search-evidence-agent/02-CONTEXT.md` — D-05 (evidence already tier-sorted in `state["evidence"]`), D-10 (backward compat: both `state["evidence"]` and `state["evidence_graph"]` are written)

### Existing Code to Modify / Create
- `factcheck_agents/agents/verify_agent.py` — new file; mirrors `evaluate_agent.py` structure with `ThreadPoolExecutor` concurrency and `reliability_signal` computation
- `factcheck_agents/models/phobert_checker.py` `build_evidence_text()` — add tier-sort (D-06, D-07); `evaluate_agent.py` — leave intact
- `factcheck_agents/agents/evaluate_agent.py` — read-only reference; do NOT modify (Phase 6 swaps it)
- `factcheck_agents/config.py` `settings.reliability_threshold` — already present (Phase 1); read via `settings.reliability_threshold`

### Project Constraints
- `.planning/PROJECT.md` §Constraints — scope boundary (factcheck_agents/ and tests/ only); graceful degrade mandatory; no new paid APIs
- `.planning/PROJECT.md` §Key Decisions — lazy model loading with `lru_cache`; `unavailable` result (not raise) for missing checkpoints

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `factcheck_agents/agents/evaluate_agent.py` `_phobert()` / `_coolant()` `lru_cache` singletons: reuse exactly the same pattern in `verify_agent.py` — models are already thread-safe for read-only inference.
- `factcheck_agents/agents/evaluate_agent.py` image fallback loop (~lines 34–39): copy verbatim into `verify_agent.py` for evidence image resolution (D-08).
- `factcheck_agents/models/phobert_checker.py` `build_evidence_text()`: extend in-place — add 3-line sort before the existing loop (D-06, D-07).
- `factcheck_agents/config.py` `settings.reliability_threshold`: already available at `settings.reliability_threshold` (float, default 0.5).

### Established Patterns
- `lru_cache(maxsize=1)` on model-getter functions — keeps model instances as process-level singletons; `verify_agent.py` must follow the same pattern.
- `ModelResult` TypedDict: `available: bool`, `label: str`, `confidence: float`, `note: str` — already defined in `state.py`; weighted signal computation reads `available`, `label`, and `confidence` fields.
- Graceful degrade: any unexpected exception in model inference → return `ModelResult(available=False, note=str(exc))` — do NOT raise. Applies inside `ThreadPoolExecutor` futures too.

### Integration Points
- `verify_agent(state: FactCheckState) -> dict` returns `{"reliability_signal": bool, "model_results": list[ModelResult], "messages": [...]}` — same shape as `evaluate_agent` return minus `reliability_signal`.
- `state["evidence"]` (flat list, `List[Evidence]`) → passed to `build_evidence_text()` in `phobert_checker.py`.
- `state["image_path"]` → primary image source; evidence fallback if None (D-08).
- Phase 6 `graph.py` will read `state["reliability_signal"]` in `route_after_verify()` — no changes in this phase.

</code_context>

<specifics>
## Specific Ideas

- Tier sort key in `build_evidence_text()`: `_TIER_ORDER = {"trusted": 0, "flagged": 1, "social": 2, "unknown": 3}`. Sort: `sorted(evidence, key=lambda e: _TIER_ORDER.get(e.get("source_tier", "unknown"), 3))`.
- Signal weights as module constants (not config env vars — they're internal algorithm detail, not user-tunable): `_PHOBERT_WEIGHT = 0.6`, `_COOLANT_WEIGHT = 0.4`.
- NEI check: `label in ("NEI",)` — PhoBERT only; COOLANT uses REAL/FAKE labels (no NEI case to handle).
- User's explicit intent: "baseline first, this milestone I expect it run well" — keep implementations minimal and focused on correctness over cleverness.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 3-Verify Agent*
*Context gathered: 2026-07-19*
