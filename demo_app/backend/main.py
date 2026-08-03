"""FastAPI demo backend for the debate-based fact-checking app.

Endpoints:
  POST /api/analyze           — accept statement + optional image; return {request_id}
  GET  /api/stream/{id}       — SSE stream for a pending analysis (EventSource target)
  GET  /api/download/debate/{id}  — download debate JSONL log (DEMO-03)
  GET  /api/download/verdict/{id} — download verdict JSON log (DEMO-03)

CORS: allows only http://localhost:5173 (Vite dev server) per DEMO-04.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import Annotated

# Project root on path so factcheck_agents is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# backend/ on path so `streaming` is importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from streaming import sse_stream

app = FastAPI(title="Fact-Check Demo API")

# CORS — allow any localhost port for local dev (DEMO-04)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: request_id → {statement, image_path}
_pending: dict[str, dict] = {}


async def _download_image(url: str) -> str | None:
    """Download image URL to a temp file; return local path or None on failure.

    COOLANT.predict() requires a local file path — it calls Path(image_path).exists().
    """
    try:
        suffix = Path(url.split("?")[0]).suffix or ".jpg"
        if not suffix.startswith("."):
            suffix = ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, urllib.request.urlretrieve, url, tmp.name)
        return tmp.name
    except Exception:
        return None


@app.post("/api/analyze")
async def analyze(
    statement: Annotated[str, Form()],
    image_url: Annotated[str | None, Form()] = None,
    image_file: Annotated[UploadFile | None, File()] = None,
) -> dict:
    """Accept statement + optional image (D-04: URL takes priority over file)."""
    image_path: str | None = None

    # D-04: URL takes priority — download to temp file so COOLANT gets a local path
    if image_url and image_url.strip():
        image_path = await _download_image(image_url.strip())

    # D-04: save uploaded file if no URL (or URL download failed)
    if image_file and not image_path:
        suffix = Path(image_file.filename or "").suffix or ".tmp"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await image_file.read())
        tmp.close()
        image_path = tmp.name

    request_id = str(uuid.uuid4())
    _pending[request_id] = {"statement": statement, "image_path": image_path}
    return {"request_id": request_id}


@app.get("/api/stream/{request_id}")
async def stream(request_id: str) -> StreamingResponse:
    """SSE endpoint — EventSource connects here (DEMO-02)."""
    params = _pending.pop(request_id, None)
    if params is None:
        raise HTTPException(
            status_code=404, detail="request_id not found or already consumed"
        )

    return StreamingResponse(
        sse_stream(request_id, params["statement"], params.get("image_path")),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Cache-Control": "no-cache",
        },
    )


@app.get("/api/download/debate/{request_id}")
async def download_debate(request_id: str) -> FileResponse:
    """Download debate JSONL log written by debate_node (DEMO-03)."""
    path = Path("logs") / "debates" / f"{request_id}.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Debate log not found")
    return FileResponse(str(path), filename=f"debate_{request_id}.jsonl")


@app.get("/api/download/verdict/{request_id}")
async def download_verdict(request_id: str) -> FileResponse:
    """Download verdict JSON log written by judge_agent (DEMO-03)."""
    path = Path("logs") / "verdicts" / f"{request_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Verdict log not found")
    return FileResponse(str(path), filename=f"verdict_{request_id}.json")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
