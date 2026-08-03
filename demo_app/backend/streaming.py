"""SSE streaming bridge for debate pipeline.

Decision mapping:
  D-01 — rechunk() emits ~8 chars every 20 ms per debate turn
  D-02 — ThreadPoolExecutor thread + asyncio.Queue + loop.call_soon_threadsafe
  D-03 — direct import of build_debate_graph, initial_state from factcheck_agents.graph
  D-09 — stage_start events emitted per NODE_STAGE_MAP
  D-10 — stage names match Vietnamese label keys
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

# Ensure project root is importable when running from demo_app/backend
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# NODE_STAGE_MAP: maps LangGraph node names to D-10 stage name values (D-09)
NODE_STAGE_MAP: dict[str, str] = {
    "real_source": "evidence_retrieval",
    "fake_source":  "evidence_retrieval",  # dedup via emitted_stages set
    "reranker":     "reranking",
    "social_loop":  "reranking",           # same stage, dedup
    "verify":       "verification",
    "debate":       "debate",
    "judge":        "verdict",
    # nei_gate and agreement_gate return {} — no stage emitted
}


def rechunk(text: str, chunk_size: int = 8) -> list[str]:
    """Split text into fixed-size chunks for re-emission (D-01)."""
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


async def sse_stream(
    request_id: str,
    statement: str,
    image_path: str | None,
) -> AsyncGenerator[str, None]:
    """Async generator yielding SSE-formatted event strings.

    Architecture (D-02):
      1. asyncio.Queue bridges the sync thread to this async generator
      2. run_graph() runs in ThreadPoolExecutor, posts events via
         loop.call_soon_threadsafe(queue.put_nowait, event)
      3. This generator awaits queue.get() and yields formatted SSE strings
      4. Heartbeat task fires every 5 s from async side
      5. done event signals shutdown to heartbeat and thread
    """
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    done = asyncio.Event()

    def _post(evt: dict) -> None:
        """Thread-safe event posting (D-02)."""
        loop.call_soon_threadsafe(queue.put_nowait, evt)

    def run_graph() -> None:
        """Synchronous pipeline execution in background thread (D-02, D-03)."""
        try:
            from factcheck_agents.graph import build_debate_graph, initial_state  # D-03

            graph = build_debate_graph(checkpointer=None)  # no persistence needed for demo
            state = initial_state(statement, image_path, language="vi")
            state["request_id"] = request_id  # override so log files match client's request_id

            accumulated: dict = {}
            emitted_stages: set[str] = set()

            # Pass request_id as thread_id so each SSE request gets isolated checkpoint state
            stream_config = {"configurable": {"thread_id": request_id}}

            for chunk in graph.stream(state, config=stream_config):
                if done.is_set():
                    break
                node_name, node_output = next(iter(chunk.items()))
                accumulated.update(node_output)

                # Emit stage_start for node transitions (D-09, D-10)
                stage = NODE_STAGE_MAP.get(node_name)
                if stage and stage not in emitted_stages:
                    emitted_stages.add(stage)
                    _post({"type": "stage_start", "name": stage})

                # Debate turn re-chunking (D-01)
                if node_name == "debate":
                    for turn in node_output.get("debate_turns", []):
                        if done.is_set():
                            break
                        _post({
                            "type": "turn_start",
                            "agent": turn.get("agent", ""),
                            "round": turn.get("round", 0),
                        })
                        for text_chunk in rechunk(turn.get("text", ""), 8):
                            if done.is_set():
                                break
                            time.sleep(0.02)  # ~20 ms per chunk (D-01)
                            _post({"type": "chunk", "text": text_chunk})
                        _post({
                            "type": "turn_end",
                            "agent": turn.get("agent", ""),
                            "round": turn.get("round", 0),
                        })

            # Emit final verdict with all accumulated state (D-05 payload)
            _post({
                "type": "verdict",
                "verdict": accumulated.get("verdict"),
                "weight_breakdown": accumulated.get("weight_breakdown"),
                "evidence_real": accumulated.get("evidence_real", []),
                "evidence_fake": accumulated.get("evidence_fake", []),
                "debate_turns": accumulated.get("debate_turns", []),
            })
        except Exception as exc:
            _post({"type": "error", "error": str(exc)})
        finally:
            _post({"type": "_done"})

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(run_graph)

    async def _heartbeat() -> None:
        """Emit heartbeat every 5 s to keep EventSource alive (DEMO-02)."""
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
        # Client disconnected — signal thread to abort (DEMO-02)
        done.set()
    finally:
        done.set()
        hb_task.cancel()
        executor.shutdown(wait=False)
        # Clean up temp file if from upload (D-04)
        if image_path and image_path.startswith("/tmp"):
            Path(image_path).unlink(missing_ok=True)
