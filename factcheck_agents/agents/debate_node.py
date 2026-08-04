"""Debate node: bounded for-loop debate between real_advocate and fake_advocate.

Each advocate cites only their assigned evidence tier. Debate turns are logged
to logs/debates/<request_id>.jsonl. If no LLM is configured, the node returns
empty turns with exit_reason="no_llm" (DEBATE-02).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from ..config import settings
from ..state import Evidence, FactCheckState
from .llm import get_llm


REAL_ADVOCATE_PROMPT = (
    "Bạn là luật sư bào chữa THẬT. Nhiệm vụ của bạn là lập luận rằng tuyên bố là ĐÚNG/THẬT "
    "chỉ dựa trên bằng chứng ở phần [TRUSTED]. Không được trích dẫn bằng chứng từ [FLAGGED] "
    "hoặc nguồn khác. Trình bày ngắn gọn nhưng đầy đủ. "
    "Nếu không có bằng chứng hỗ trợ, hãy nói rõ điều đó."
)

FAKE_ADVOCATE_PROMPT = (
    "Bạn là luật sư phản biện GIẢ. Nhiệm vụ của bạn là lập luận rằng tuyên bố là SAI/GIẢ "
    "chỉ dựa trên bằng chứng ở phần [FLAGGED] hoặc phần kiểm chứng. "
    "Không được trích dẫn bằng chứng từ [TRUSTED]. Trình bày ngắn gọn nhưng đầy đủ. "
    "Nếu không có bằng chứng phản bác, hãy nói rõ điều đó."
)


def _format_evidence(evidence: List[Evidence]) -> str:
    """Format evidence list with tier tags for advocate prompts."""
    if not evidence:
        return "(no supporting evidence available)"
    lines = []
    for i, e in enumerate(evidence, 1):
        tier = e.get("source_tier", "unknown").upper()
        tag = f"[{tier}]"
        lines.append(
            f"{tag} [{i}] {e.get('title')} — {e.get('url')}\n    {e.get('snippet')}"
        )
    return "\n".join(lines)


def _append_turn(request_id: str, turn: dict) -> None:
    """Append a debate turn to logs/debates/<request_id>.jsonl.

    Wrapped in try/except to never block the pipeline (DEBATE-03).
    """
    try:
        log_dir = Path("logs/debates")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{request_id}.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
    except Exception:
        pass


def debate_node(state: FactCheckState) -> dict:
    """Run bounded debate between real_advocate and fake_advocate.

    Returns:
        dict with keys:
            - debate_turns: list of turn dicts
            - debate_exit_reason: str ("no_llm", "llm_error", or "max_rounds")
            - messages: list of message tuples
    """
    from ..state import FactCheckState

    # No-LLM guard (DEBATE-02)
    llm = get_llm()
    if llm is None:
        return {
            "debate_turns": [],
            "debate_exit_reason": "no_llm",
            "messages": [("assistant", "[Debate] skipped — no LLM configured")],
        }

    # Dir creation (DEBATE-03)
    Path("logs/debates").mkdir(parents=True, exist_ok=True)

    statement = state["statement"]
    evidence_real = state.get("evidence_real") or []
    evidence_fake = state.get("evidence_fake") or []
    request_id = state.get("request_id", "unknown")

    turns = []
    exit_reason = None

    # Bounded for-loop debate (DEBATE-02, D-15)
    for round_num in range(settings.max_debate_rounds):
        # Build real_advocate prompt (DEBATE-01)
        real_evidence_text = _format_evidence(evidence_real)
        real_user = f"CLAIM:\n{statement}\n\nEVIDENCE:\n{real_evidence_text}\n"
        if round_num > 0 and turns:
            # Include last fake_advocate turn for context
            last_fake = [t for t in turns if t["agent"] == "fake_advocate"][-1]
            real_user += f"\nOPPOSING ARGUMENT:\n{last_fake['text']}\n"

        # Call LLM for real_advocate
        try:
            _real_prompt = settings.real_advocate_prompt or REAL_ADVOCATE_PROMPT
            real_resp = llm.invoke([("system", _real_prompt), ("user", real_user)])
            real_text = str(getattr(real_resp, "content", ""))
        except Exception as exc:
            error_turn = {
                "agent": "real_advocate",
                "round": round_num,
                "timestamp": datetime.utcnow().isoformat(),
                "text": "",
                "error": str(exc),
            }
            turns.append(error_turn)
            print(error_turn)
            _append_turn(request_id, error_turn)
            exit_reason = "llm_error"
            break

        real_turn = {
            "agent": "real_advocate",
            "round": round_num,
            "timestamp": datetime.utcnow().isoformat(),
            "text": real_text,
        }
        turns.append(real_turn)
        print(real_turn)
        _append_turn(request_id, real_turn)

        # Build fake_advocate prompt (DEBATE-01)
        fake_evidence_text = _format_evidence(evidence_fake)
        fake_user = f"CLAIM:\n{statement}\n\nEVIDENCE:\n{fake_evidence_text}\n"
        # Include last real_advocate turn for context
        fake_user += f"\nOPPOSING ARGUMENT:\n{real_text}\n"

        # Call LLM for fake_advocate
        try:
            _fake_prompt = settings.fake_advocate_prompt or FAKE_ADVOCATE_PROMPT
            fake_resp = llm.invoke([("system", _fake_prompt), ("user", fake_user)])
            fake_text = str(getattr(fake_resp, "content", ""))
        except Exception as exc:
            error_turn = {
                "agent": "fake_advocate",
                "round": round_num,
                "timestamp": datetime.utcnow().isoformat(),
                "text": "",
                "error": str(exc),
            }
            turns.append(error_turn)
            print(error_turn)
            _append_turn(request_id, error_turn)
            exit_reason = "llm_error"
            break

        fake_turn = {
            "agent": "fake_advocate",
            "round": round_num,
            "timestamp": datetime.utcnow().isoformat(),
            "text": fake_text,
        }
        turns.append(fake_turn)
        print(fake_turn)
        _append_turn(request_id, fake_turn)

    return {
        "debate_turns": turns,
        "debate_exit_reason": exit_reason or "max_rounds",
        "messages": [
            (
                "assistant",
                f"[Debate] {len(turns)} turns ({exit_reason or 'max_rounds'})",
            )
        ],
    }
