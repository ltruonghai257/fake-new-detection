"""Debate node: convergence-driven debate between real_advocate and fake_advocate.

Each advocate outputs structured JSON with verdict, confidence, argument, concession.
Full turn history is passed each round. Debate exits when both advocates agree on the
same verdict (REAL/FAKE), or when max_debate_rounds is hit as a safety cap.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..config import settings
from ..state import Evidence, FactCheckState
from .llm import get_llm


REAL_ADVOCATE_PROMPT = (
    "Bạn là luật sư bào chữa THẬT. Nhiệm vụ của bạn là lập luận rằng tuyên bố là ĐÚNG/THẬT "
    "chỉ dựa trên bằng chứng ở phần [TRUSTED]. Không được trích dẫn bằng chứng từ [FLAGGED].\n\n"
    "Phản hồi PHẢI là JSON hợp lệ với các trường sau:\n"
    "{\n"
    '  "verdict": "REAL",\n'
    '  "confidence": 0.0-1.0,\n'
    '  "argument": "lập luận của bạn",\n'
    '  "concession": "điểm nào bạn nhượng bộ cho phía đối lập (nếu có, hoặc null)"\n'
    "}\n"
    "Chỉ trả về JSON, không có text nào khác."
)

FAKE_ADVOCATE_PROMPT = (
    "Bạn là luật sư phản biện GIẢ. Nhiệm vụ của bạn là lập luận rằng tuyên bố là SAI/GIẢ "
    "chỉ dựa trên bằng chứng ở phần [FLAGGED] hoặc phần kiểm chứng. "
    "Không được trích dẫn bằng chứng từ [TRUSTED].\n\n"
    "Phản hồi PHẢI là JSON hợp lệ với các trường sau:\n"
    "{\n"
    '  "verdict": "FAKE",\n'
    '  "confidence": 0.0-1.0,\n'
    '  "argument": "lập luận của bạn",\n'
    '  "concession": "điểm nào bạn nhượng bộ cho phía đối lập (nếu có, hoặc null)"\n'
    "}\n"
    "Chỉ trả về JSON, không có text nào khác."
)


def _format_evidence(evidence: List[Evidence]) -> str:
    """Format evidence list with tier tags for advocate prompts."""
    if not evidence:
        return "(no supporting evidence available)"
    lines = []
    for i, e in enumerate(evidence, 1):
        tier = e.get("source_tier", "unknown").upper()
        lines.append(
            f"[{tier}] [{i}] {e.get('title')} — {e.get('url')}\n    {e.get('snippet')}"
        )
    return "\n".join(lines)


def _format_model_results(results: List[dict]) -> str:
    """Format full model outputs including per-class probabilities."""
    if not results:
        return "(no model predictions available)"
    lines = []
    for r in results:
        if not r.get("available"):
            lines.append(f"- {r.get('model', 'unknown').upper()}: unavailable ({r.get('note', '')})")
            continue
        model = r.get("model", "unknown").upper()
        label = r.get("label", "N/A")
        confidence = r.get("confidence", 0.0)
        probs = r.get("probabilities") or {}
        prob_str = ", ".join(f"{k}: {v:.1%}" for k, v in sorted(probs.items(), key=lambda x: -x[1]))
        lines.append(f"- {model}: {label} (confidence={confidence:.1%})")
        if prob_str:
            lines.append(f"  Probabilities: {prob_str}")
    return "\n".join(lines) if lines else "(no model predictions available)"


def _format_history(turns: List[dict]) -> str:
    """Format full debate history for context."""
    if not turns:
        return "(no prior turns)"
    lines = []
    for t in turns:
        agent = t.get("agent", "unknown")
        round_num = t.get("round", 0)
        verdict = t.get("verdict", "?")
        argument = t.get("argument", t.get("text", ""))[:300]
        concession = t.get("concession")
        line = f"[Round {round_num}] {agent}: verdict={verdict}, argument={argument}"
        if concession:
            line += f", concession={concession}"
        lines.append(line)
    return "\n".join(lines)


def _parse_advocate_json(content: str) -> Optional[dict]:
    """Parse advocate JSON response; return None on failure."""
    try:
        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return None


def _append_turn(request_id: str, turn: dict) -> None:
    """Append a debate turn to logs/debates/<request_id>.jsonl (DEBATE-03)."""
    try:
        log_dir = Path("logs/debates")
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"{request_id}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
    except Exception:
        pass


def debate_node(state: FactCheckState) -> dict:
    """Run convergence-driven debate between real_advocate and fake_advocate.

    Exits when both advocates agree on the same verdict (REAL/FAKE), or when
    max_debate_rounds is hit. Full history passed each round.

    Returns:
        dict with keys:
            - debate_turns: list of structured turn dicts
            - debate_exit_reason: "converged" | "max_rounds" | "no_llm" | "llm_error"
            - debate_converged: bool
            - debate_agreed_verdict: "REAL" | "FAKE" | None
            - messages: list of message tuples
    """
    llm = get_llm()
    if llm is None:
        return {
            "debate_turns": [],
            "debate_exit_reason": "no_llm",
            "debate_converged": False,
            "debate_agreed_verdict": None,
            "messages": [("assistant", "[Debate] skipped — no LLM configured")],
        }

    Path("logs/debates").mkdir(parents=True, exist_ok=True)

    statement = state["statement"]
    evidence_real = state.get("evidence_real") or []
    evidence_fake = state.get("evidence_fake") or []
    model_results = state.get("model_results") or []
    request_id = state.get("request_id", "unknown")

    model_output_text = _format_model_results(model_results)
    real_evidence_text = _format_evidence(evidence_real)
    fake_evidence_text = _format_evidence(evidence_fake)

    turns: List[dict] = []
    exit_reason: Optional[str] = None
    converged = False
    agreed_verdict: Optional[str] = None

    for round_num in range(settings.max_debate_rounds):
        history_text = _format_history(turns)

        # ── real_advocate ────────────────────────────────────────────────────
        real_user = (
            f"CLAIM:\n{statement}\n\n"
            f"MODEL PREDICTIONS (PhoBERT + COOLANT):\n{model_output_text}\n\n"
            f"[TRUSTED] EVIDENCE:\n{real_evidence_text}\n\n"
            f"DEBATE HISTORY:\n{history_text}\n"
        )
        try:
            _real_prompt = settings.real_advocate_prompt or REAL_ADVOCATE_PROMPT
            real_resp = llm.invoke([("system", _real_prompt), ("user", real_user)])
            real_content = str(getattr(real_resp, "content", ""))
            real_data = _parse_advocate_json(real_content) or {}
        except Exception as exc:
            error_turn = {
                "agent": "real_advocate",
                "round": round_num,
                "timestamp": datetime.utcnow().isoformat(),
                "verdict": None,
                "confidence": 0.0,
                "argument": "",
                "concession": None,
                "error": str(exc),
            }
            turns.append(error_turn)
            _append_turn(request_id, error_turn)
            exit_reason = "llm_error"
            break

        real_turn = {
            "agent": "real_advocate",
            "round": round_num,
            "timestamp": datetime.utcnow().isoformat(),
            "verdict": real_data.get("verdict", "REAL"),
            "confidence": float(real_data.get("confidence", 0.5)),
            "argument": real_data.get("argument", real_content[:500]),
            "concession": real_data.get("concession"),
        }
        turns.append(real_turn)
        _append_turn(request_id, real_turn)

        # ── fake_advocate ────────────────────────────────────────────────────
        history_text = _format_history(turns)
        fake_user = (
            f"CLAIM:\n{statement}\n\n"
            f"MODEL PREDICTIONS (PhoBERT + COOLANT):\n{model_output_text}\n\n"
            f"[FLAGGED] EVIDENCE:\n{fake_evidence_text}\n\n"
            f"DEBATE HISTORY:\n{history_text}\n"
        )
        try:
            _fake_prompt = settings.fake_advocate_prompt or FAKE_ADVOCATE_PROMPT
            fake_resp = llm.invoke([("system", _fake_prompt), ("user", fake_user)])
            fake_content = str(getattr(fake_resp, "content", ""))
            fake_data = _parse_advocate_json(fake_content) or {}
        except Exception as exc:
            error_turn = {
                "agent": "fake_advocate",
                "round": round_num,
                "timestamp": datetime.utcnow().isoformat(),
                "verdict": None,
                "confidence": 0.0,
                "argument": "",
                "concession": None,
                "error": str(exc),
            }
            turns.append(error_turn)
            _append_turn(request_id, error_turn)
            exit_reason = "llm_error"
            break

        fake_turn = {
            "agent": "fake_advocate",
            "round": round_num,
            "timestamp": datetime.utcnow().isoformat(),
            "verdict": fake_data.get("verdict", "FAKE"),
            "confidence": float(fake_data.get("confidence", 0.5)),
            "argument": fake_data.get("argument", fake_content[:500]),
            "concession": fake_data.get("concession"),
        }
        turns.append(fake_turn)
        _append_turn(request_id, fake_turn)

        # ── convergence check ────────────────────────────────────────────────
        real_v = str(real_turn["verdict"]).upper()
        fake_v = str(fake_turn["verdict"]).upper()
        if real_v == fake_v and real_v in {"REAL", "FAKE"}:
            converged = True
            agreed_verdict = real_v
            exit_reason = "converged"
            break

    return {
        "debate_turns": turns,
        "debate_exit_reason": exit_reason or "max_rounds",
        "debate_converged": converged,
        "debate_agreed_verdict": agreed_verdict,
        "messages": [
            (
                "assistant",
                f"[Debate] {len(turns)} turns ({exit_reason or 'max_rounds'})"
                + (f" → agreed={agreed_verdict}" if converged else ""),
            )
        ],
    }
