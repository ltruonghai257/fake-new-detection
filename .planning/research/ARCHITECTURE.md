# ARCHITECTURE.md — M2: Debate-Based Verification Pipeline and Demo App

## New Components

| Component           | File                          | Output                                                     |
| ------------------- | ----------------------------- | ---------------------------------------------------------- |
| `real_source_agent` | `agents/real_source_agent.py` | `state["evidence_real"]`                                   |
| `fake_source_agent` | `agents/fake_source_agent.py` | `state["evidence_fake"]`                                   |
| `evidence_reranker` | `reranker.py`                 | reranked evidence list (BM25 + embedding)                  |
| `social_loop_node`  | `agents/social_loop_agent.py` | one-shot weak-evidence social search                       |
| `agreement_gate`    | `agents/agreement_gate.py`    | `state["agreement_score"]`                                 |
| `debate_node`       | `agents/debate_node.py`       | `state["debate_turns"]` (encapsulates real/fake advocates) |
| `judge_agent`       | `agents/judge_agent.py`       | `state["verdict"]`, `state["weight_breakdown"]`            |
| `demo_app/`         | `demo_app/`                   | FastAPI backend + React/Vite/TS frontend                   |

## Modified Components

| Component            | Change                                                                                                                                 |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `state.py`           | Add: `evidence_real`, `evidence_fake`, `agreement_score`, `debate_turns`, `request_id`, `weight_breakdown`, `social_loop_fired`        |
| `config.py`          | Add: `FACTCHECK_AGREEMENT_THRESHOLD` (0.8), `FACTCHECK_MAX_DEBATE_ROUNDS` (2), `GOOGLE_FACTCHECK_API_KEY` (""), social loop thresholds |
| `graph.py`           | Replace `search → verify → … → conclusion` with full M2 topology; keep `evaluate_agent` export for backward compat                     |
| `agents/__init__.py` | Export new agents; keep `conclusion_agent` export for backward compat                                                                  |

**NOT modified in Phase 1:** `cli.py`, `mcp_server.py`, `__init__.py` (Phase 2 scope), `verify_agent.py`, `phobert_checker.py`, training code

## State Shape Changes

```python
class FactCheckState(TypedDict, total=False):
    # === Existing (unchanged) ===
    statement: str
    image_path: Optional[str]
    search_queries: List[str]
    evidence: List[Evidence]          # kept for backward compat
    model_results: List[ModelResult]
    evidence_graph: Optional[Any]
    reliability_signal: Optional[bool]
    verdict: Verdict
    messages: Annotated[list, add_messages]

    # === New M2 fields ===
    request_id: str                   # uuid4 prefix for log filenames
    evidence_real: List[Evidence]     # from real_source_agent
    evidence_fake: List[Evidence]     # from fake_source_agent
    social_loop_fired: bool           # hard cap: social loop never runs twice
    agreement_score: Optional[float]  # from agreement_gate (0.0–1.0)
    debate_turns: List[dict]          # [{round, agent, text, scores}]
    debate_exit_reason: Optional[str] # "max_rounds" | "echo_chamber" | "timeout" | "skipped"
    weight_breakdown: dict            # {"phobert": 0.3, "coolant": 0.3, "evidence": 0.4, "scores": {...}}
```

All new fields use `total=False` — no existing tests break. `initial_state()` sets sane defaults.

## Graph Topology

```
START
  ↓
real_source_agent ──────────────────────────────────────────────────┐
fake_source_agent ──────────────────────────────────────────────────┤
  ↓                                                                  │
[reranker runs inside evidence agents before writing to state]       │
  ↓                                                                  │
route_after_social_loop ←─────────────────────────────────────────  │
  ├── (evidence weak AND social_loop_fired=False) → social_loop_node │
  └── (otherwise)                                                    │
        ↓                                                            │
      verify_agent  [PhoBERT + COOLANT, existing]                   │
        ↓                                                            │
      agreement_gate                                                 │
        ├── (agreement_score ≥ threshold) → judge_agent             │
        └── (agreement_score < threshold) → debate_node             │
                                              ↓                      │
                                          judge_agent                │
                                              ↓                      │
                                            END  ←───────────────── ┘
```

**Conditional edges:**

-   `route_after_social_loop(state)`: reads `social_loop_fired`, `len(evidence_real)`, `len(evidence_fake)`, credibility scores → `"social_loop"` or `"verify"`
-   `route_after_agreement(state)`: reads `agreement_score` → `"debate"` or `"judge"`
-   `social_loop_node` sets `social_loop_fired=True` before returning (prevents second fire)

**Debate loop is a single node** — not graph-level recursion. `debate_node` internally calls real_advocate and fake_advocate LLM calls in a Python `for` loop up to `max_debate_rounds`. This avoids LangGraph cycle complexity and checkpointing issues.

## Build Order

### Phase 1 — Debate Pipeline (REQs 1–4, 6, 7)

**Wave 1 (independent):**

1. `state.py` — add new fields, update `initial_state()`
2. `config.py` — add new env vars with defaults

**Wave 2 (depends on Wave 1):** 3. `reranker.py` — BM25 + embedding reranker (RERANK-01) 4. `real_source_agent.py` — trusted domain search 5. `fake_source_agent.py` — tingia.gov.vn + Google Fact Check stub

**Wave 3 (depends on Wave 2):** 6. `social_loop_agent.py` — one-shot weak-evidence social search (SOCLOOP-01) 7. `agreement_gate.py` — agreement score + evidence-credibility computation 8. Unit tests: reranker recall@k, social loop fire-once guard

**Wave 4 (depends on Wave 3):** 9. `debate_node.py` — bounded advocate debate with JSONL logging 10. `judge_agent.py` — weighted verdict + verdict JSON logging 11. Update `graph.py` — full M2 topology

**Wave 5 (depends on Wave 4):** 12. Integration tests: full pipeline on 2 sample claims

### Phase 2 — Demo App (REQ 5)

13. `demo_app/backend/main.py` — FastAPI + SSE endpoint
14. `demo_app/frontend/` — React/Vite/TS
15. CORS, heartbeat, StrictMode safety

## Demo App Integration

**Backend ↔ Pipeline:**

-   Direct Python import of `run_fact_check()` from `factcheck_agents`
-   FastAPI background task runs pipeline; yields SSE events via `asyncio.Queue`
-   No subprocess; no network hop between demo and pipeline

**SSE API contract:**

```
POST  /api/analyze         (JSON body: {statement, image_path?})
GET   /api/analyze/stream  (EventSource, query param: request_id)

SSE events:
  stage_start  {"stage": "retrieval"|"verification"|"debate"|"judgment"}
  turn_start   {"agent": "real_advocate"|"fake_advocate", "round": N}
  chunk        {"content": "...text fragment..."}
  turn_end     {"scores": {"factuality": N, "engagement": N, "grounding": N}}
  verdict      {"verdict": "Real"|"Fake"|"NEI", "confidence": 0.0-1.0, "weight_breakdown": {...}}
  heartbeat    {}  (every 5s)
```

**Frontend structure:**

```
demo_app/
  backend/
    main.py          FastAPI app
    streaming.py     SSE generator + asyncio bridge
  frontend/
    src/
      App.tsx
      components/
        DebateTranscript.tsx   alternating advocate bubbles
        VerdictCard.tsx        label + confidence + weight bar
        EvidencePanel.tsx      tier badges
    package.json
    vite.config.ts
    tsconfig.json
```

## Test Breakage Risks

| Test File                  | Risk                                                                | Mitigation                                                                 |
| -------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `test_graph_wiring.py`     | New nodes change topology                                           | Update `route_after_verify` tests; add `route_after_agreement` tests       |
| `test_conclusion_agent.py` | `conclusion_agent` still exists; no breakage if we don't replace it | Keep `conclusion_agent`; `judge_agent` is new additive component           |
| `test_search_agent.py`     | `search_agent` unchanged                                            | No breakage; new source agents are additive                                |
| All 83 existing tests      | New `total=False` state fields                                      | Safe — TypedDict optional fields don't require values in existing fixtures |

**Key decision:** `conclusion_agent` stays in the codebase; `judge_agent` is a new additive node. `graph.py` continues to expose both `build_graph()` (M1 topology for backward compat) and a new `build_debate_graph()` for M2. This avoids breaking any callers that use the old graph.

---

_Research completed: 2026-08-02_
