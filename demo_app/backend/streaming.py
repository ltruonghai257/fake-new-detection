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
    "fake_source": "evidence_retrieval",  # dedup via emitted_stages set
    "reranker": "reranking",
    "social_loop": "reranking",  # same stage, dedup
    "verify": "verification",
    "debate": "debate",
    "judge": "verdict",
    # nei_gate and agreement_gate return {} — no stage emitted
}


def rechunk(text: str, chunk_size: int = 8) -> list[str]:
    """Split text into fixed-size chunks for re-emission (D-01)."""
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _summarize_node(node_name: str, node_output: dict, accumulated: dict) -> str:
    """Build a Vietnamese human-readable summary of what a node produced."""
    if node_name in ("real_source", "fake_source"):
        ev_real = accumulated.get("evidence_real") or []
        ev_fake = accumulated.get("evidence_fake") or []
        parts = []
        if ev_real:
            trusted = sum(1 for e in ev_real if e.get("source_tier") == "trusted")
            parts.append(f"{len(ev_real)} nguồn ủng hộ ({trusted} tin cậy)")
        if ev_fake:
            parts.append(f"{len(ev_fake)} nguồn phản bác")
        if not parts:
            return "Chưa tìm thấy bằng chứng phù hợp."
        return "Tìm thấy: " + ", ".join(parts) + "."

    if node_name == "reranker":
        ev_real = accumulated.get("evidence_real") or []
        ev_fake = accumulated.get("evidence_fake") or []
        return f"Xếp hạng xong: {len(ev_real)} nguồn ủng hộ, {len(ev_fake)} nguồn phản bác."

    if node_name == "verify":
        model_results = accumulated.get("model_results") or []
        parts = []
        for r in model_results:
            name = r.get("model", "?")
            if not r.get("available"):
                parts.append(f"{name}: không khả dụng")
            else:
                label = r.get("label", "?")
                conf = r.get("confidence", 0.0)
                parts.append(f"{name}: {label} ({conf*100:.0f}%)")
        return "Kết quả mô hình: " + " | ".join(parts) + "."

    if node_name == "debate":
        turns = node_output.get("debate_turns") or []
        exit_reason = node_output.get("debate_exit_reason", "")
        converged = node_output.get("debate_converged", False)
        agreed = node_output.get("debate_agreed_verdict")
        reason_vi = {
            "no_llm": "không có LLM",
            "llm_error": "lỗi LLM",
            "max_rounds": "đạt số vòng tối đa",
            "converged": "hai bên đồng thuận",
        }.get(exit_reason, exit_reason)
        if not turns:
            return f"Tranh luận bị bỏ qua ({reason_vi})."
        if converged and agreed:
            return f"Tranh luận {len(turns)} lượt → đồng thuận: {agreed}."
        return f"Tranh luận {len(turns)} lượt ({reason_vi})."

    if node_name == "judge":
        verdict = accumulated.get("verdict") or {}
        label = verdict.get("verdict_label_vi", verdict.get("label", "?"))
        conf = verdict.get("confidence", 0.0)
        return f"Phán quyết: {label} ({conf*100:.0f}% tin cậy)."

    return ""


async def sse_stream(
    request_id: str,
    statement: str,
    image_path: str | None,
    use_phobert: bool = True,
    use_coolant: bool = True,
    use_evidence: bool = True,
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

            graph = build_debate_graph(
                checkpointer=None
            )  # no persistence needed for demo
            state = initial_state(statement, image_path, language="vi")
            state["use_phobert"] = use_phobert
            state["use_coolant"] = use_coolant
            state["use_evidence"] = use_evidence
            state["request_id"] = (
                request_id  # override so log files match client's request_id
            )

            accumulated: dict = {}
            emitted_stages: set[str] = set()

            # Pass request_id as thread_id so each SSE request gets isolated checkpoint state
            stream_config = {"configurable": {"thread_id": request_id}}

            for chunk in graph.stream(state, config=stream_config):
                if done.is_set():
                    break
                node_name, node_output = next(iter(chunk.items()))
                if node_output is None:
                    continue  # LangGraph 1.2.x emits None for nodes returning {} (no-op updates)
                accumulated.update(node_output)

                # Emit stage_start for node transitions (D-09, D-10)
                stage = NODE_STAGE_MAP.get(node_name)
                if stage and stage not in emitted_stages:
                    emitted_stages.add(stage)
                    _post({"type": "stage_start", "name": stage})

                # Emit stage_log: human-readable summary of what this node produced
                log_msg = _summarize_node(node_name, node_output, accumulated)
                if log_msg:
                    _post(
                        {
                            "type": "stage_log",
                            "stage": stage or node_name,
                            "message": log_msg,
                        }
                    )

                # Debate turn re-chunking (D-01)
                if node_name == "debate":
                    for turn in node_output.get("debate_turns", []):
                        if done.is_set():
                            break
                        _post(
                            {
                                "type": "turn_start",
                                "agent": turn.get("agent", ""),
                                "round": turn.get("round", 0),
                                "verdict": turn.get("verdict"),
                                "confidence": turn.get("confidence"),
                            }
                        )
                        # argument field (new structured format); fallback to text for compat
                        text = turn.get("argument") or turn.get("text", "")
                        for text_chunk in rechunk(text, 8):
                            if done.is_set():
                                break
                            time.sleep(0.02)  # ~20 ms per chunk (D-01)
                            _post({"type": "chunk", "text": text_chunk})
                        _post(
                            {
                                "type": "turn_end",
                                "agent": turn.get("agent", ""),
                                "round": turn.get("round", 0),
                                "verdict": turn.get("verdict"),
                                "confidence": turn.get("confidence"),
                                "concession": turn.get("concession"),
                            }
                        )
                    # Emit convergence event after all turns
                    if node_output.get("debate_converged"):
                        _post(
                            {
                                "type": "debate_converged",
                                "agreed_verdict": node_output.get("debate_agreed_verdict"),
                            }
                        )

            # Emit model results to UI (for debate context display)
            if accumulated.get("model_results"):
                model_results = accumulated.get("model_results")
                formatted_results = []
                for r in model_results:
                    if r.get("available"):
                        formatted_results.append(
                            {
                                "model": r.get("model"),
                                "label": r.get("label"),
                                "confidence": r.get("confidence"),
                                "probabilities": r.get("probabilities"),
                            }
                        )
                if formatted_results:
                    _post(
                        {
                            "type": "model_results",
                            "results": formatted_results,
                        }
                    )

            # Extract per-model labels and probabilities for UI display
            model_results = accumulated.get("model_results") or []
            phobert_result = next(
                (
                    r
                    for r in model_results
                    if r.get("model") == "phobert_vifactcheck" and r.get("available")
                ),
                None,
            )
            coolant_result = next(
                (
                    r
                    for r in model_results
                    if r.get("model") == "coolant" and r.get("available")
                ),
                None,
            )

            # Compute evidence breakdown components for UI transparency
            ev_real = accumulated.get("evidence_real") or []
            ev_fake = accumulated.get("evidence_fake") or []
            trusted_count = sum(1 for e in ev_real if e.get("source_tier") == "trusted")
            tier_score = trusted_count / len(ev_real) if ev_real else 0.0
            count_score = min(1.0, (len(ev_real) + len(ev_fake)) / 5)
            consistency_score = max(0.1, accumulated.get("consistency_score", 0.1))

            weight_breakdown = dict(accumulated.get("weight_breakdown") or {})
            weight_breakdown["phobert_label"] = (
                phobert_result.get("label") if phobert_result else None
            )
            weight_breakdown["phobert_probabilities"] = (
                phobert_result.get("probabilities") if phobert_result else None
            )
            weight_breakdown["phobert_evidence_text"] = (
                phobert_result.get("evidence_text") if phobert_result else None
            )
            weight_breakdown["phobert_workflow_steps"] = (
                phobert_result.get("workflow_steps") if phobert_result else None
            )
            weight_breakdown["coolant_label"] = (
                coolant_result.get("label") if coolant_result else None
            )
            weight_breakdown["coolant_probabilities"] = (
                coolant_result.get("probabilities") if coolant_result else None
            )
            weight_breakdown["coolant_workflow_steps"] = (
                coolant_result.get("workflow_steps") if coolant_result else None
            )
            weight_breakdown["evidence_breakdown"] = {
                "tier_score": round(tier_score, 4),
                "count_score": round(count_score, 4),
                "consistency_score": round(consistency_score, 4),
                "trusted_count": trusted_count,
                "total_real": len(ev_real),
                "total_fake": len(ev_fake),
                "total_evidence": len(ev_real) + len(ev_fake),
            }
            weight_breakdown["evidence_workflow_steps"] = accumulated.get(
                "evidence_workflow_steps", []
            )

            # Emit final verdict with all accumulated state (D-05 payload)
            _post(
                {
                    "type": "verdict",
                    "verdict": accumulated.get("verdict"),
                    "weight_breakdown": weight_breakdown,
                    "evidence_real": accumulated.get("evidence_real", []),
                    "evidence_fake": accumulated.get("evidence_fake", []),
                    "debate_turns": accumulated.get("debate_turns", []),
                }
            )
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
            evt_type = evt.get("type", "message")
            yield f"event: {evt_type}\ndata: {json.dumps(evt, ensure_ascii=False)}\n\n"
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
