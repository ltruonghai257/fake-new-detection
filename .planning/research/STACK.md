# STACK.md — Stack Additions for M2: Debate-Based Verification Pipeline and Demo App

## Backend Additions

### Google Fact Check Tools API
- **Package**: `google-api-python-client` (already in requirements.txt: `>=2.120.0`)
- **Why**: Already installed for Google Drive upload; includes `factchecktools_v1alpha1` API
- **Integration**: Use existing `google-api-python-client` with `build('factchecktools', 'v1alpha1', developerKey=...)`
- **Stub/Mock without API key**: Use `unittest.mock.patch` on the `claims().search()` method in tests; raise `ValueError` when key absent to gate real calls
- **No new package needed** — leverage existing Google client library

### FastAPI + SSE Streaming
- **Package**: `fastapi>=0.110.0` (PyPI, stable as of mid-2026)
- **SSE helper**: `sse-starlette>=2.1.0` — provides `EventSourceResponse`; FastAPI itself does not bundle SSE in older versions
- **ASGI Server**: `uvicorn[standard]>=0.29.0`
- **Integration**:
  ```python
  from sse_starlette.sse import EventSourceResponse

  @app.get("/analyze/stream")
  async def stream_debate(request: Request) -> EventSourceResponse:
      async def generator():
          async for event in run_debate_pipeline(state):
              if await request.is_disconnected():
                  break
              yield {"event": event["type"], "data": json.dumps(event)}
      return EventSourceResponse(generator())
  ```
- **Why sse-starlette over raw Starlette StreamingResponse**: Handles SSE framing (retry, id, keep-alive pings) correctly; disconnect detection built-in

### Python JSONL Logging
- **Recommended**: stdlib `json` + `pathlib.Path` — no new package
- **Implementation pattern**:
  ```python
  def append_jsonl(path: Path, entry: dict) -> None:
      path.parent.mkdir(parents=True, exist_ok=True)
      with open(path, "a", encoding="utf-8") as f:
          f.write(json.dumps(entry, ensure_ascii=False) + "\n")
  ```
- **Use `ensure_ascii=False`** — Vietnamese text must not be escaped to `\uXXXX`

### LangGraph Streaming (Alongside SSE)
- **Package**: `langgraph` — already in pyproject.toml (upgrade may be needed for streaming APIs)
- **API**: `graph.stream(input, config, stream_mode="values")` — yields state snapshots after each node
- **Integration with SSE**: LangGraph yields state dicts; SSE layer translates to typed events
- **No new package** — upgrade existing langgraph if needed for stream_mode support

## Frontend Stack

### Core Framework
- **React**: `react@^18.3.0` (npm) — React 19 is out but ecosystem compatibility still maturing; 18.3 is the safe stable choice
- **TypeScript**: `typescript@^5.4.0`
- **Vite**: `vite@^5.2.0`
- **React Plugin**: `@vitejs/plugin-react@^4.3.0`

### SSE Client
- **Recommended**: Native browser `EventSource` API (no package needed)
- **Why**: Built-in, zero dependencies, sufficient for unidirectional streaming from same origin
- **CORS note**: For local dev (different ports), add CORS middleware to FastAPI:
  ```python
  from fastapi.middleware.cors import CORSMiddleware
  app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"])
  ```
- **React StrictMode risk**: Double-mount will open two SSE connections. Fix: cleanup `useEffect`:
  ```typescript
  useEffect(() => {
    const es = new EventSource('/api/analyze/stream');
    return () => es.close(); // cleanup on unmount
  }, []);
  ```

### UI Styling
- **Tailwind CSS**: `tailwindcss@^3.4.0` + `@tailwindcss/typography` — minimal setup, no heavy component lib needed for thesis demo

## Logging

### Backend JSONL Logging
- **Approach**: stdlib `json` + custom `append_jsonl()` helper
- **Log locations**:
  - `logs/debates/<request_id>.jsonl` — one line per debate turn
  - `logs/verdicts/<request_id>.json` — single JSON object with full weight breakdown
- **Crash safety**: Write to a temp file then `os.replace()` for verdicts (atomic); debates use append-only so partial writes are recoverable
- **macOS permissions**: Create `logs/` at startup, check writable; raise `RuntimeError` if not

## Packages NOT to Add

| Package | Reason to Skip |
|---------|---------------|
| `rapidlog`, `JSON-LOGGER` | Stdlib sufficient for JSONL; adds dependency for no gain |
| `react-eventsource`, `@react-nano/use-event-source` | Native EventSource API is sufficient |
| `websockets` | SSE is one-way (server→client); WebSocket is overkill for debate streaming |
| `celery` / `rq` | No background job queue needed; debate runs synchronously in SSE response |
| `redis` | No shared state between requests; in-process is fine for local-only demo |
| `orjson` for demo_app | Already in factcheck_agents but demo_app stdlib `json` is fine |

## Summary Table

| Feature | Package | Version | Status |
|---------|---------|---------|--------|
| Google Fact Check API | `google-api-python-client` | `>=2.120.0` | **Existing** |
| FastAPI | `fastapi` | `>=0.110.0` | **New** |
| SSE helper | `sse-starlette` | `>=2.1.0` | **New** |
| ASGI Server | `uvicorn[standard]` | `>=0.29.0` | **New** |
| JSONL Logging | stdlib `json` | — | **No new dep** |
| React | `react` | `^18.3.0` | **New (frontend)** |
| TypeScript | `typescript` | `^5.4.0` | **New (frontend)** |
| Vite | `vite` | `^5.2.0` | **New (frontend)** |
| React Plugin | `@vitejs/plugin-react` | `^4.3.0` | **New (frontend)** |
| Tailwind CSS | `tailwindcss` | `^3.4.0` | **New (frontend)** |
| SSE Client | Native `EventSource` | — | **No new dep** |

## Integration Points with Existing Code

- **config.py**: Add `GOOGLE_FACTCHECK_API_KEY` (empty string default → stub mode)
- **state.py**: Add `evidence_real`, `evidence_fake`, `debate_turns`, `agreement_score`, `weight_breakdown` fields
- **graph.py**: New debate-related nodes; debate loop stays inside a single node (not graph-level loop)
- **demo_app/backend/**: Thin FastAPI wrapper around `run_fact_check()`; streams debate events via SSE
- **demo_app/frontend/**: Vite + React app; connects to backend SSE endpoint

---
*Research completed: 2026-08-02*
