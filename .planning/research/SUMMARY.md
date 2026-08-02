# Research Summary — M2: Debate-Based Verification Pipeline and Demo App

## TL;DR for Implementation

M2 replaces the single-pass verdict (which silently overrides model predictions with evidence presence) with a structured debate pipeline: dual-source retrieval → evidence reranking → optional social loop → parallel model verification → agreement gate → advocate debate → weighted judge. A demo web app streams the debate live via SSE.

---

## Stack (from STACK.md)

**New Python deps (add to `pyproject.toml`):**

-   `fastapi>=0.110.0` + `sse-starlette>=2.1.0` + `uvicorn[standard]>=0.29.0` — demo app backend
-   Reranker: check repo first for `sentence-transformers` or `rank_bm25`; if absent, add `rank_bm25>=0.2.2` (BM25) and optionally `sentence-transformers>=2.7.0` (embedding rerank)
-   `google-api-python-client` — **already in requirements.txt**; use for Fact Check Tools API stub

**New frontend deps (npm in `demo_app/frontend/`):**

-   `react@^18.3.0`, `typescript@^5.4.0`, `vite@^5.2.0`, `@vitejs/plugin-react@^4.3.0`, `tailwindcss@^3.4.0`
-   SSE client: **native `EventSource` API** — no npm package needed

**JSONL logging:** stdlib `json` + `pathlib` — no new dep. Use `ensure_ascii=False` for Vietnamese.

**Do NOT add:** WebSockets, Redis, Celery, `sse-starlette` alternatives, React EventSource packages.

---

## Features (from FEATURES.md)

### Agreement Gate

-   Score: `0.30 * phobert_confidence + 0.30 * coolant_confidence + 0.40 * evidence_credibility`
-   Thresholds: ≥ 0.75 skip debate, < 0.75 enter debate (configurable via `FACTCHECK_AGREEMENT_THRESHOLD`)
-   Unavailable model → treat confidence as 0.0, normalize over available signals only
-   NEI from any model → force `agreement_score = 0.0`

### Evidence-Credibility Formula

```
credibility = 0.40 * tier_score + 0.30 * count_score + 0.30 * consistency_score
tier_score:        trusted=1.0 | flagged=0.5 | unknown=0.3  (weighted avg over all evidence)
count_score:       min(1.0, log2(1 + trusted_count) / log2(6))
consistency_score: aligned_evidence / max(1, total_evidence)  (+0.2 diversity bonus if ≥ 3 domains)
floor:             max(0.1, computed)  — never collapses to 0
```

### Debate Loop

-   `real_advocate` cites `evidence_real` (trusted tier); `fake_advocate` cites `evidence_fake`
-   Each turn scored on: Factuality (1-5), Rebuttal Engagement (1-5), Evidence Grounding (1-5)
-   Termination: `max_rounds` hard cap | echo chamber (similarity > 0.9 for 2 rounds) | quality degradation
-   **Debate loop stays inside a single LangGraph node** — not graph-level recursion

### Streaming UI

-   SSE preferred over WebSockets; 8-char/20ms flush for live feel
-   Event types: `stage_start`, `turn_start`, `chunk`, `turn_end`, `verdict`, `heartbeat`
-   UI: alternating chat bubbles (blue=real / red=fake), quality score badges, 30/30/40 weight bar

---

## Architecture (from ARCHITECTURE.md)

### Graph Topology

```
START → real_source → fake_source
          ↓ [reranker runs inside agents]
        route_after_social_loop
          ├── (weak evidence, loop not fired) → social_loop_node ─┐
          └── (otherwise) ──────────────────────────────────────  ↓
                                                          verify_agent
                                                               ↓
                                                        agreement_gate
                                                    ├── (≥ threshold) → judge
                                                    └── (< threshold) → debate_node → judge
                                                                              ↓
                                                                            END
```

### Key Decisions

| Decision                                                  | Rationale                                                |
| --------------------------------------------------------- | -------------------------------------------------------- |
| Debate loop inside single node                            | Avoids LangGraph cycle complexity + checkpointing issues |
| `conclusion_agent` kept, `judge_agent` is additive        | No test breakage; old graph still works                  |
| `build_debate_graph()` new function, keep `build_graph()` | Backward compat for M1 callers                           |
| `social_loop_fired: bool` on state                        | Hard cap without graph-level cycle                       |
| Demo app = direct import, not subprocess                  | Zero network overhead; simpler SSE integration           |

### New State Fields (all `total=False`)

`evidence_real`, `evidence_fake`, `social_loop_fired`, `request_id`, `agreement_score`, `debate_turns`, `debate_exit_reason`, `weight_breakdown`

### Build Order (Phase 1 — Debate Pipeline)

1. `state.py` + `config.py` (Wave 1)
2. `reranker.py`, `real_source_agent.py`, `fake_source_agent.py` (Wave 2)
3. `social_loop_agent.py`, `agreement_gate.py` + unit tests (Wave 3)
4. `debate_node.py`, `judge_agent.py`, `graph.py` update (Wave 4)
5. Integration tests on 2 sample claims (Wave 5)

---

## Pitfalls (from PITFALLS.md)

### P0 — Must fix before M2 ships

| Risk                                        | Prevention                                                                                            |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Agreement gate ÷0 when model unavailable    | Normalize over available signals only; if < 2, force `agreement_score=0.0`                            |
| Existing 83 tests break on new state fields | All new fields `total=False`; update `initial_state()` defaults; update `conftest.py` fixtures        |
| Debate rounds unbounded (cost runaway)      | Hard-cap via Python `for` loop, never trust LLM to terminate                                          |
| JSONL corruption on crash                   | `f.flush()` + `os.fsync(f.fileno())` after each line; atomic `.tmp` + `os.replace()` for verdict JSON |
| Vietnamese text mangled in logs             | `json.dumps(..., ensure_ascii=False)` everywhere                                                      |

### P1 — Address during Phase 1 execution

| Risk                                            | Prevention                                                                          |
| ----------------------------------------------- | ----------------------------------------------------------------------------------- |
| Judge overconfidence (confidence always > 0.95) | Calibrate: multiply by 0.8 if > 0.9; log `raw_confidence` + `calibrated_confidence` |
| SSE React StrictMode double-mount               | `useEffect` cleanup: `return () => es.close()`                                      |
| Evidence-credibility binary collapse            | Floor at 0.1; log tier/count/consistency components separately                      |
| Social loop fires twice                         | `social_loop_fired: bool` on state; test asserts second execution skipped           |

### P2 — Awareness items

-   Advocate hallucinating evidence: validate cited URLs against `evidence_real`/`evidence_fake`
-   Echo chamber early exit: detect similarity > 0.9 across 2 consecutive rounds
-   CORS for local dev: FastAPI `CORSMiddleware` with `allow_origins=["http://localhost:5173"]`
-   macOS log dir permissions: `Path("logs/debates").mkdir(parents=True, exist_ok=True, mode=0o755)`

---

## Requirements Coverage Check

| REQ-ID        | Phase   | Key Design Choices                                                     |
| ------------- | ------- | ---------------------------------------------------------------------- |
| EVRET-01..04  | Phase 1 | `real_source_agent.py`, `fake_source_agent.py`, NEI gate in graph      |
| RERANK-01     | Phase 1 | `reranker.py` with BM25 + embedding; recall@k test                     |
| SOCLOOP-01    | Phase 1 | `social_loop_agent.py`; `social_loop_fired` state flag; fire-once test |
| AGREE-01..02  | Phase 1 | `agreement_gate.py`; 0.30/0.30/0.40 formula; threshold env var         |
| DEBATE-01..02 | Phase 1 | `debate_node.py`; JSONL logging; `logs/debates/<id>.jsonl`             |
| JUDGE-01..03  | Phase 1 | `judge_agent.py`; 1-5 scoring; `logs/verdicts/<id>.json`               |
| DEMO-01..02   | Phase 2 | `demo_app/backend/` + `demo_app/frontend/`; SSE streaming              |

---

_Synthesized: 2026-08-02 from STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md_
