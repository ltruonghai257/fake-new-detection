# Phase 2: Demo App - Context

**Gathered:** 2026-08-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Ship a local-only thesis defense web app: FastAPI SSE backend (`demo_app/backend/`) + React/Vite/TypeScript frontend (`demo_app/frontend/`). The app accepts a Vietnamese statement + optional image, runs the M2 debate pipeline (`build_debate_graph()`), streams the debate live turn-by-turn via SSE, and displays a Vietnamese verdict card with 30/30/40 weight breakdown.

**Not in scope:** Auth, public deployment, WebSockets, per-turn LLM token streaming, adaptive termination (DEBATE-EXT-01), Twitter/Facebook social search, any changes to `factcheck_agents/` package internals.

</domain>

<decisions>
## Implementation Decisions

### Streaming Bridge (SSE ↔ LangGraph)

- **D-01:** Debate text delivery uses **re-chunk buffered text** — `debate_node.py` is NOT modified. Each full advocate turn (returned by synchronous `llm.invoke()`) is buffered, then re-emitted as ~8-character SSE `chunk` events every 20 ms.
- **D-02:** The pipeline runs in a background `ThreadPoolExecutor` thread. An **`asyncio.Queue`** bridges the sync thread to the async SSE generator: the thread posts event dicts via `loop.call_soon_threadsafe(queue.put_nowait, event)`, and the async SSE generator reads from the queue with `await queue.get()`.
- **D-03:** `streaming.py` imports `build_debate_graph` and `initial_state` **directly** from `factcheck_agents.graph` — no new function added to `factcheck_agents/__init__.py`. The demo app is self-contained.

### Image Input

- **D-04:** The submission form exposes **both** a file upload (`<input type="file">`) and a URL text field. When both are provided, the **URL takes priority** (URL field value is used as `image_path`). When only file is provided, the backend saves it to a temp path and passes that as `image_path`. When neither is provided, `image_path=None` (COOLANT gracefully degrades).

### Evidence Panel

- **D-05:** The evidence panel is **revealed with the verdict** — hidden during debate streaming, shown all at once when the `verdict` SSE event arrives.
- **D-06:** Evidence items are organized in **two tabs**: "Nguồn ủng hộ" (`evidence_real`) and "Nguồn phản bác" (`evidence_fake`). Each item within a tab carries **tier badges** (trusted=green, flagged=orange, social=blue, unknown=gray).

### Score Badge Timing

- **D-07:** Argument quality score badges (from `judge_agent`) appear on debate turn chat bubbles **all at once when the `verdict` SSE event arrives** — not during streaming. Badges are hidden while debate is streaming.
- **D-08:** Each bubble shows **three separate dimension badges** (one per judge scoring dimension, e.g., "Bằng chứng: 4", "Lập luận: 3", "Phản bác: 5"). No aggregate score badge.

### Stage Progress Display

- **D-09:** A **horizontal step indicator** is shown while the pipeline runs, driven by `stage_start` SSE events. Active step is highlighted; completed steps are checked.
- **D-10:** Vietnamese stage labels (used both as `stage_start` event `name` values and as UI step labels):
  - `"evidence_retrieval"` → "Tìm bằng chứng"
  - `"reranking"` → "Xếp hạng bằng chứng"
  - `"verification"` → "Kiểm định mô hình"
  - `"debate"` → "Tranh luận"
  - `"verdict"` → "Phán quyết"

### Claude's Discretion

- **Pipeline entry point** (D-03): Direct import in `streaming.py` — no `__init__.py` changes. Keeps the demo app self-contained.
- **Error state UX**: Show a red **error card** with Vietnamese message "Đã xảy ra lỗi. Vui lòng thử lại." and a "Thử lại" retry button. No partial results shown on error.
- **Heartbeat display**: Heartbeat SSE events are consumed client-side to keep the `EventSource` alive but are NOT shown in the UI.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope & Requirements
- `.planning/ROADMAP.md` §"Phase 2: Demo App" — goals, 2-plan breakdown (02-01 backend, 02-02 frontend), success criteria (5 bullets)
- `.planning/REQUIREMENTS.md` §DEMO-01, DEMO-02, DEMO-03, DEMO-04 — 4 requirements for this phase

### Debate Pipeline (Phase 1, already implemented)
- `factcheck_agents/graph.py` — `build_debate_graph()` function (M2 entry point); `initial_state()` signature
- `factcheck_agents/agents/debate_node.py` — Synchronous LLM debate loop; `llm.invoke()` blocking calls; JSONL logging pattern; `debate_turns` list structure
- `factcheck_agents/agents/judge_agent.py` — 3-dimension scoring (1-5 each); `weight_breakdown` dict; `logs/verdicts/<request_id>.json` write
- `factcheck_agents/state.py` — `FactCheckState` fields: `evidence_real`, `evidence_fake`, `evidence_social`, `debate_turns`, `agreement_score`, `weight_breakdown`, `request_id`, `verdict`

### Existing Infrastructure
- `factcheck_agents/config.py` — `Settings` dataclass; env var pattern
- `factcheck_agents/__init__.py` — Current `run_fact_check()` using M1 graph (DO NOT MODIFY for Phase 2)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `build_debate_graph(checkpointer=None)` in `graph.py` — direct import; pass `checkpointer=None` for demo (no persistence needed per run)
- `initial_state(statement, image_path, language)` in `graph.py` — constructs the initial `FactCheckState`
- `logs/debates/<request_id>.jsonl` and `logs/verdicts/<request_id>.json` — written by `debate_node` and `judge_agent`; served as file downloads from FastAPI
- `state["evidence_real"]` / `state["evidence_fake"]` — `List[Evidence]` with `source_tier`, `title`, `url`, `snippet` fields; mapped to tab panels
- `state["debate_turns"]` — `List[dict]` with `agent`, `round`, `text`, `timestamp` keys; each turn maps to a chat bubble
- `state["weight_breakdown"]` — dict from `judge_agent`; contains PhoBERT/COOLANT/evidence weights and final score

### Established Patterns
- FastAPI SSE: use `StreamingResponse` with `media_type="text/event-stream"` and a generator yielding `data: {json}\n\n` strings
- React StrictMode-safe `EventSource`: create in `useEffect`, close in cleanup callback (prevents double-connection in dev)
- Vietnamese UI copy is required for all user-visible text (DEMO-04)
- CORS allows only `http://localhost:5173` (Vite dev server default port)

### Integration Points
- `streaming.py` → calls `build_debate_graph().invoke(state)` in thread; intercepts stage transitions via the `asyncio.Queue` event bridge
- `stage_start` events are emitted **by streaming.py logic** (not by graph nodes) — streaming.py wraps graph execution and posts stage events when it detects state transitions (or via direct injection points)
- Re-chunking: after `debate_node` completes (detected via state diff or post-processing), `streaming.py` reads `state["debate_turns"]` and re-emits each turn's text as sequential `chunk` events
- `verdict` SSE event carries the full `state["verdict"]` dict + `weight_breakdown` + scored `debate_turns` (with dimension badges)

</code_context>

<specifics>
## Specific Ideas

- Chat bubbles: blue for `real_advocate`, red (or orange) for `fake_advocate` — specified in ROADMAP.md success criteria
- Verdict card confidence gauge: can be radial arc or horizontal bar (no preference specified — Claude decides)
- Download buttons for `logs/debates/<id>.jsonl` and `logs/verdicts/<id>.json` must be **working** (DEMO-03) — FastAPI needs file-serve endpoints for these
- Tailwind CSS + Vite + TypeScript scaffold for the frontend (ROADMAP.md plan 02-02 specifies this)
- Character-level re-chunking rate: ~8 chars / 20 ms (from ROADMAP.md success criteria)

</specifics>

<deferred>
## Deferred Ideas

- **Real LLM token streaming** — `llm.stream()` + `debate_node` modification. Deferred; re-chunk approach sufficient for demo.
- **WebSocket transport** — SSE is sufficient per REQUIREMENTS.md Out of Scope.
- **Auth, public deployment** — explicitly out of scope (REQUIREMENTS.md Out of Scope).
- **Social search in demo** — M2 graph includes `social_loop_agent`; no separate demo-specific social UI needed.

</deferred>

---

*Phase: 2-demo-app*
*Context gathered: 2026-08-03*
