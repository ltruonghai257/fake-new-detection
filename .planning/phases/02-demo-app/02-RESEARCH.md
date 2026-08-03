# Phase 2: Demo App — Research

**Produced:** 2026-08-03  
**Purpose:** Concrete code-level answers for the planner. All patterns verified against the existing codebase.

---

## 1. User Constraints (from 02-CONTEXT.md — planner must honor verbatim)

### Phase Boundary
- Ship a local-only thesis defense web app: FastAPI SSE backend (`demo_app/backend/`) + React/Vite/TypeScript frontend (`demo_app/frontend/`)
- Input: Vietnamese statement + optional image (file upload OR URL)
- Output: Live SSE streaming of debate turns + Vietnamese verdict card with 30/30/40 weight breakdown
- **Out of scope:** Auth, public deployment, WebSockets, per-turn LLM token streaming, adaptive termination, Twitter/Facebook social search, any changes to `factcheck_agents/` package internals

### Locked Decisions (D-01 through D-10)

**D-01 (Re-chunking):** Debate text delivery uses re-chunked buffered text — `debate_node.py` is NOT modified. Each full advocate turn (returned by synchronous `llm.invoke()`) is buffered, then re-emitted as ~8-character SSE `chunk` events every 20 ms.

**D-02 (Threading Bridge):** The pipeline runs in a background `ThreadPoolExecutor` thread. An `asyncio.Queue` bridges the sync thread to the async SSE generator: the thread posts event dicts via `loop.call_soon_threadsafe(queue.put_nowait, event)`, and the async SSE generator reads from the queue with `await queue.get()`.

**D-03 (Direct Import):** `streaming.py` imports `build_debate_graph` and `initial_state` directly from `factcheck_agents.graph` — no new function added to `factcheck_agents/__init__.py`.

**D-04 (Image Input Priority):** Both file upload (`<input type="file">`) and URL text field. URL takes priority when both provided. Backend saves uploaded file to temp path. `image_path=None` when neither provided.

**D-05 (Evidence Panel Timing):** Evidence panel hidden during debate streaming, shown all at once when the `verdict` SSE event arrives.

**D-06 (Evidence Panel Structure):** Two tabs: "Nguồn ủng hộ" (`evidence_real`) and "Nguồn phản bác" (`evidence_fake`). Tier badges: trusted=green, flagged=orange, social=blue, unknown=gray.

**D-07 (Score Badge Timing):** Score badges appear on debate bubbles all at once when `verdict` SSE event arrives — NOT during streaming.

**D-08 (Score Badge Structure):** Three separate dimension badges per bubble: "Bằng chứng: N", "Lập luận: N", "Phản bác: N" (1-5 scale from `argument_scores`).

**D-09 (Stage Indicator):** Horizontal step indicator driven by `stage_start` SSE events. Active step highlighted; completed steps checked.

**D-10 (Vietnamese Stage Labels):**
- `"evidence_retrieval"` → "Tìm bằng chứng"
- `"reranking"` → "Xếp hạng bằng chứng"
- `"verification"` → "Kiểm định mô hình"
- `"debate"` → "Tranh luận"
- `"verdict"` → "Phán quyết"

### Claude's Discretion
- **Error state UX:** Red error card "Đã xảy ra lỗi. Vui lòng thử lại." + "Thử lại" button. No partial results on error.
- **Heartbeat display:** Consume heartbeat events client-side to keep EventSource alive; NOT shown in UI.
- **Confidence gauge:** Horizontal bar (0-100%). [ASSUMED]

---

## 2. Standard Stack & Dependencies

### Backend — New Python Packages
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
```
[ASSUMED — these are mainstream FastAPI packages as of 2025; versions are conservative lower bounds]

These are added to `demo_app/backend/requirements.txt`. They do NOT go into `pyproject.toml` extras (demo app is self-contained, not part of the `factcheck_agents` package).

**Already available in project environment:**
- `python-dotenv` (in main deps)
- `requests`, `httpx`, `orjson` (in main deps)
- All `factcheck_agents` dependencies (installed via `uv sync --extra agents`)

### Frontend — npm Packages
```json
{
  "react": "^18.3.1",
  "react-dom": "^18.3.1",
  "@vitejs/plugin-react": "^4.3.1",
  "vite": "^5.3.1",
  "tailwindcss": "^3.4.4",
  "postcss": "^8.4.39",
  "autoprefixer": "^10.4.19",
  "typescript": "^5.5.3",
  "@types/react": "^18.3.3",
  "@types/react-dom": "^18.3.0"
}
```
[ASSUMED — latest stable as of July 2025; React 18 required for StrictMode-safe EventSource]

**No additional runtime state management libraries needed** — React `useState`/`useEffect` sufficient.

---

## 3. Architecture Patterns

### Pattern A: Two-Endpoint Design (POST + GET SSE)

Native `EventSource` only supports GET requests, so the architecture uses two endpoints:

1. **`POST /api/analyze`** — Accepts form data (`statement`, optional `image_url`, optional `image_file`). Saves analysis parameters to in-memory dict. Returns `{request_id: string}`.
2. **`GET /api/stream/{request_id}`** — EventSource connects here. Pops from in-memory dict, returns `StreamingResponse(sse_stream(...), media_type="text/event-stream")`.

**Client flow:**
```
POST /api/analyze → {request_id}
  ↓
GET /api/stream/{request_id}  (EventSource)
  ↓ SSE events flow
stage_start × 5 → turn_start/chunk/turn_end × N → verdict
```

### Pattern B: asyncio.Queue + ThreadPoolExecutor Bridge (D-02)

```python
async def sse_stream(request_id: str, statement: str, image_path: str | None) -> AsyncGenerator[str, None]:
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    done = asyncio.Event()

    def _post(evt: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, evt)

    def run_graph() -> None:
        """Sync function, runs in ThreadPoolExecutor."""
        try:
            from factcheck_agents.graph import build_debate_graph, initial_state  # D-03
            graph = build_debate_graph(checkpointer=None)
            state = initial_state(statement, image_path, language="vi")
            state["request_id"] = request_id   # override UUID so log files match

            accumulated: dict = {}
            emitted_stages: set[str] = set()

            for chunk in graph.stream(state):
                node_name, node_output = next(iter(chunk.items()))
                accumulated.update(node_output)

                stage = NODE_STAGE_MAP.get(node_name)
                if stage and stage not in emitted_stages:
                    emitted_stages.add(stage)
                    _post({"type": "stage_start", "name": stage})

                if node_name == "debate":
                    for turn in node_output.get("debate_turns", []):
                        _post({"type": "turn_start", "agent": turn["agent"], "round": turn["round"]})
                        import time
                        for text_chunk in rechunk(turn.get("text", ""), 8):
                            time.sleep(0.02)
                            _post({"type": "chunk", "text": text_chunk})
                        _post({"type": "turn_end", "agent": turn["agent"], "round": turn["round"]})

            _post({
                "type": "verdict",
                "verdict": accumulated.get("verdict"),
                "weight_breakdown": accumulated.get("weight_breakdown"),
                "evidence_real": accumulated.get("evidence_real", []),
                "evidence_fake": accumulated.get("evidence_fake", []),
                "debate_turns": accumulated.get("debate_turns", []),
            })
            _post({"type": "_done"})
        except Exception as exc:
            _post({"type": "error", "error": str(exc)})
            _post({"type": "_done"})

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(run_graph)

    async def _heartbeat() -> None:
        while not done.is_set():
            await asyncio.sleep(5)
            if not done.is_set():
                queue.put_nowait({"type": "heartbeat"})

    hb_task = asyncio.create_task(_heartbeat())

    try:
        while True:
            evt = await queue.get()
            if evt.get("type") == "_done":
                break
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        done.set()   # signal thread to abort on next _post check
    finally:
        done.set()
        hb_task.cancel()
        executor.shutdown(wait=False)
```

### Pattern C: LangGraph Node → Stage Mapping

```python
NODE_STAGE_MAP: dict[str, str] = {
    "real_source": "evidence_retrieval",
    "fake_source":  "evidence_retrieval",   # dedup via emitted_stages set
    "reranker":     "reranking",
    "social_loop":  "reranking",            # same stage, dedup
    "verify":       "verification",
    "debate":       "debate",
    "judge":        "verdict",
    # nei_gate and agreement_gate return {} — no stage emitted
}
```

**Why use `graph.stream()` instead of `graph.invoke()`:** `stream()` yields `{node_name: node_output}` as each node completes, enabling stage detection without modifying `factcheck_agents`. `invoke()` would require post-processing only, making stage ordering harder.

### Pattern D: React StrictMode-Safe EventSource

```typescript
useEffect(() => {
    if (!requestId) return;
    const es = new EventSource(`http://localhost:8000/api/stream/${requestId}`);

    es.addEventListener('stage_start', (e) => { /* ... */ });
    es.addEventListener('turn_start', (e) => { /* ... */ });
    es.addEventListener('chunk', (e) => { /* append text */ });
    es.addEventListener('turn_end', (e) => { /* finalize turn */ });
    es.addEventListener('verdict', (e) => { /* reveal evidence + badges */ });

    es.onerror = () => { setError(true); es.close(); };

    return () => es.close();  // StrictMode cleanup — prevents double-connection
}, [requestId]);
```

### Pattern E: File Upload + URL Priority (D-04)

```python
@app.post("/api/analyze")
async def analyze(
    statement: Annotated[str, Form()],
    image_url: Annotated[str | None, Form()] = None,
    image_file: Annotated[UploadFile | None, File()] = None,
):
    image_path: str | None = image_url or None     # URL wins
    if image_file and not image_path:
        suffix = Path(image_file.filename or "").suffix or ".tmp"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await image_file.read()); tmp.close()
        image_path = tmp.name
    ...
```

---

## 4. Don't Hand-Roll

| Problem | Use instead |
|---------|-------------|
| SSE response | FastAPI `StreamingResponse(media_type="text/event-stream")` |
| File upload | FastAPI `UploadFile` + `Form()` |
| Threading | `concurrent.futures.ThreadPoolExecutor` |
| Async queue | `asyncio.Queue` |
| CORS | `fastapi.middleware.cors.CORSMiddleware` |
| Vietnamese JSON | `json.dumps(obj, ensure_ascii=False)` |
| Frontend SSE | Native `EventSource` (no library) |
| Frontend HTTP | Native `fetch()` |
| Frontend styling | Tailwind CSS utility classes |

---

## 5. SSE Event Protocol (exact JSON shapes)

All events use the wire format: `data: <json>\n\n`

```
stage_start   → {"type":"stage_start","name":"evidence_retrieval"}
turn_start    → {"type":"turn_start","agent":"real_advocate","round":0}
chunk         → {"type":"chunk","text":"Theo các "}
turn_end      → {"type":"turn_end","agent":"real_advocate","round":0}
heartbeat     → {"type":"heartbeat"}
verdict       → {"type":"verdict","verdict":{...},"weight_breakdown":{...},"evidence_real":[...],"evidence_fake":[...],"debate_turns":[...]}
error         → {"type":"error","error":"<message>"}
```

**`verdict` event detail:**
```json
{
  "type": "verdict",
  "verdict": {
    "label": "TRUE|FALSE|MISLEADING|UNVERIFIED",
    "verdict_binary": "REAL|FAKE",
    "verdict_label_vi": "Thật|Giả",
    "confidence": 0.85,
    "rationale": "Vietnamese string",
    "citations": ["url1"],
    "recommendation": "Vietnamese string"
  },
  "weight_breakdown": {
    "phobert": 0.3,
    "coolant": 0.3,
    "evidence": 0.4,
    "argument_scores": [
      {"agent":"real_advocate","round":0,"factuality":4,"rebuttal_engagement":3,"evidence_grounding":5}
    ]
  },
  "evidence_real": [{"title":"","url":"","snippet":"","source_tier":"trusted"}],
  "evidence_fake": [{"title":"","url":"","snippet":"","source_tier":"flagged"}],
  "debate_turns": [{"agent":"real_advocate","round":0,"text":"...","timestamp":"..."}]
}
```

---

## 6. State Field Mapping (FactCheckState → UI)

| UI Component | State Field | Notes |
|---|---|---|
| StageIndicator | `stage_start.name` | From SSE event |
| DebateTranscript bubbles | `debate_turns[i].agent/text/round` | blue=real_advocate, red=fake_advocate |
| Score badges (D-08) | `weight_breakdown.argument_scores[i]` | factuality, rebuttal_engagement, evidence_grounding |
| VerdictCard label | `verdict.verdict_label_vi` | "Thật" or "Giả" |
| VerdictCard confidence | `verdict.confidence` | 0.0-1.0 → percentage |
| Weight breakdown bar | `weight_breakdown.phobert/coolant/evidence` | 30/30/40 |
| Download links | `request_id` | `/api/download/debate/{id}` |
| EvidencePanel tab 1 | `evidence_real[]` | "Nguồn ủng hộ" |
| EvidencePanel tab 2 | `evidence_fake[]` | "Nguồn phản bác" |
| Tier badge colors | `evidence[i].source_tier` | trusted=green, flagged=orange, social=blue, unknown=gray |

---

## 7. Integration Points

### streaming.py integration chain
```
POST /api/analyze
  → store {statement, image_path} in _pending[request_id]
  → return {request_id}

GET /api/stream/{request_id}
  → pop from _pending
  → StreamingResponse(sse_stream(request_id, statement, image_path))

sse_stream()
  → ThreadPoolExecutor.submit(run_graph)
  → run_graph():
      build_debate_graph(checkpointer=None)   # D-03
      state = initial_state(statement, image_path, language="vi")
      state["request_id"] = request_id        # ensures log files match client request_id
      for chunk in graph.stream(state):
          detect stage → emit stage_start
          detect "debate" node → rechunk turns, emit turn_start/chunk/turn_end
      emit verdict (from accumulated state)
```

### Download endpoint integration
```
judge_agent writes → logs/verdicts/{request_id}.json
debate_node writes → logs/debates/{request_id}.jsonl
GET /api/download/debate/{request_id} → FileResponse("logs/debates/{id}.jsonl")
GET /api/download/verdict/{request_id} → FileResponse("logs/verdicts/{id}.json")
```
**Important:** Log files are written by `debate_node` and `judge_agent` using `state["request_id"]`. Since we override `state["request_id"] = request_id` after `initial_state()`, the log filenames will match the client's `request_id`.

---

## 8. Common Pitfalls

| Pitfall | Root cause | Fix |
|---|---|---|
| Event loop blocked | `graph.stream()` is sync | ThreadPoolExecutor (D-02) |
| Thread-unsafe queue.put | queue not thread-safe from different threads | `loop.call_soon_threadsafe(queue.put_nowait, ...)` (D-02) |
| Double EventSource in StrictMode | React StrictMode mounts twice | `return () => es.close()` in useEffect |
| Vietnamese JSON garbled | `ensure_ascii=True` default | `json.dumps(obj, ensure_ascii=False)` |
| Temp file leak | NamedTemporaryFile not cleaned up | try/finally cleanup after stream generator exits |
| Log file not found for download | request_id mismatch | Override `state["request_id"]` before `graph.stream()` |
| CORS preflight fails for Form+File | Browser sends OPTIONS with custom headers | `allow_headers=["*"]` in CORSMiddleware |
| Heartbeat prevents disconnect detection | Heartbeat loop not cancelled on client disconnect | `done.set()` in finally block |

---

## 9. Project Constraints (from .windsurfrules)

- Always prefix shell commands with `rtk` to minimize token consumption
- Use `uv run` for Python commands (not `python` or `python3`)
- No auth, no public deployment (DEMO-04 / CONTEXT deferred)

---

*Phase: 2-demo-app*  
*Research produced: 2026-08-03*

## RESEARCH COMPLETE
