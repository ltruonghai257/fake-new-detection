"""Judge agent: scores debate turns on 1-5 dimensions and produces weight breakdown.

Does NOT produce a final verdict — that's done by expert_agent.
This node only scores the debate and computes component signals for downstream.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

from ..config import settings
from ..state import (
    FAKE_MODEL_LABELS,
    REAL_MODEL_LABELS,
    Evidence,
    FactCheckState,
    ModelResult,
)
from .llm import get_llm, parse_json

_BINARY_REAL_LABELS = {"SUPPORTED", "REAL", "TRUE"}
_BINARY_NEI_LABELS = {"NEI", "UNVERIFIED"}
# FAKE: REFUTED, FAKE, FALSE, MISLEADING — explicit falsehoods


def _format_models(model_results: List[ModelResult]) -> str:
    lines = []
    for m in model_results:
        if m.get("available"):
            lines.append(
                f"- {m['model']}: {m.get('label')} (conf={m.get('confidence')}, probs={m.get('probabilities')})"
            )
        else:
            lines.append(f"- {m['model']}: unavailable ({m.get('note')})")
    return "\n".join(lines)


def _format_evidence(evidence: List[Evidence]) -> str:
    if not evidence:
        return "(no web evidence retrieved)"
    lines = []
    for i, e in enumerate(evidence, 1):
        lines.append(f"[{i}] {e.get('title')} — {e.get('url')}\n    {e.get('snippet')}")
    return "\n".join(lines)


def _format_debate_turns(debate_turns: List[dict]) -> str:
    if not debate_turns:
        return "(no debate turns)"
    lines = []
    for turn in debate_turns:
        agent = turn.get("agent", "unknown")
        round_num = turn.get("round", 0)
        error = turn.get("error")
        if error:
            lines.append(f"- {agent} (round {round_num}): ERROR: {error}")
            continue
        verdict = turn.get("verdict", "?")
        confidence = turn.get("confidence", 0.0)
        argument = turn.get("argument", turn.get("text", ""))[:300]
        concession = turn.get("concession")
        line = f"- {agent} (round {round_num}): verdict={verdict} conf={confidence:.0%}, argument={argument}"
        if concession:
            line += f", concession={concession}"
        lines.append(line)
    return "\n".join(lines)


_JUDGE_SYSTEM_PROMPT_DEFAULT = (
    "Bạn là GIÁM KHẢO tranh luận, trung lập, trong phiên xác minh tin tức tiếng Việt. "
    "Bạn KHÔNG phán quyết claim — chỉ chấm chất lượng màn tranh luận. "
    "Chỉ đánh giá trên dữ liệu được cung cấp, không dùng kiến thức ngoài.\n\n"
    "ĐẦU VÀO (trong tin nhắn user):\n"
    "- CLAIM cần xác minh\n"
    "- TÍN HIỆU MODEL: PhoBERT đo độ đúng của claim theo văn bản; COOLANT đo mức nhất quán "
    "giữa claim và ảnh ('FAKE' = ảnh và claim không khớp, chưa kết luận tin giả). "
    "Luận điểm tham chiếu, không phải kết luận sự thật.\n"
    "- BẰNG CHỨNG web kèm tier; BIÊN BẢN TRANH LUẬN hai phe; trạng thái hội tụ (nếu có)\n\n"
    "NHIỆM VỤ:\n"
    "1. Chấm TỪNG lượt tranh luận, số nguyên 1-5, trên ba tiêu chí:\n"
    "   - factuality: khẳng định trong lượt có đúng theo bằng chứng không\n"
    "   - rebuttal_engagement: có phản bác trực tiếp lập luận gần nhất của đối thủ không\n"
    "   - evidence_grounding: có bám dữ liệu đã cho không — trừ mạnh nếu bịa số liệu/nguồn\n"
    "2. Xác định bên thắng theo điểm: 'real_advocate' | 'fake_advocate' | 'tie'.\n"
    "3. Viết explanation bằng tiếng Việt gồm: model_summary (từng model cho tín hiệu gì — "
    "hai model đo hai khía cạnh khác nhau, kết quả khác nhau không hẳn là mâu thuẫn), "
    "debate_winner (kèm lý do), evidence_summary, confidence_breakdown "
    "(bốn trọng số thực, cộng = 1.0).\n\n"
    "Nếu debate_converged=true: agreed_verdict là tiên nghiệm tham chiếu, không thay thế "
    "việc chấm chất lượng lập luận thực tế.\n\n"
    "ĐẦU RA — DUY NHẤT một object JSON hợp lệ, không markdown, không văn bản trước/sau:\n"
    "{\n"
    '  "turn_scores": [\n'
    '    {"agent": "real_advocate", "round": 0, "factuality": 4, "rebuttal_engagement": 3, "evidence_grounding": 5}\n'
    "  ],\n"
    '  "explanation": {\n'
    '    "model_summary": "từng model cho tín hiệu gì, kèm xác suất.",\n'
    '    "debate_winner": "real_advocate | fake_advocate | tie",\n'
    '    "evidence_summary": "tóm tắt bằng chứng then chốt.",\n'
    '    "confidence_breakdown": {"phobert": 0.3, "coolant": 0.3, "evidence": 0.2, "debate": 0.2}\n'
    "  }\n"
    "}"
)

JUDGE_SYSTEM_PROMPT = settings.judge_prompt or _JUDGE_SYSTEM_PROMPT_DEFAULT


def _write_verdict_log(request_id: str, data: dict) -> None:
    """Write verdict log to logs/verdicts/<request_id>.json.

    Wrapped in try/except to never block the pipeline (JUDGE-03).
    """
    try:
        log_dir = Path("logs/verdicts")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{request_id}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def judge_agent(state: FactCheckState) -> dict:
    """Score debate turns and compute weight breakdown for downstream expert.

    Does NOT produce a final verdict. Returns turn_scores, debate_winner,
    and computed weights only. The expert_agent produces the final verdict.
    """
    statement = state["statement"]
    model_results = state.get("model_results", []) or []
    evidence_real = state.get("evidence_real") or []
    evidence_fake = state.get("evidence_fake") or []
    debate_turns = state.get("debate_turns") or []
    debate_converged = state.get("debate_converged", False)
    debate_agreed_verdict = state.get("debate_agreed_verdict")
    request_id = state.get("request_id", "unknown")

    model_detail: dict = {}
    for m in model_results:
        if m.get("available"):
            model_detail[m.get("model", "unknown")] = {
                "label": m.get("label"),
                "confidence": m.get("confidence"),
                "probabilities": m.get("probabilities"),
            }

    # No debate → no scores, minimal breakdown
    if not debate_turns:
        return {
            "weight_breakdown": {
                "phobert": 0.0,
                "coolant": 0.0,
                "evidence": 0.0,
                "debate": 0.0,
                "phobert_conf": 0.0,
                "coolant_conf": 0.0,
                "debate_conf": 0.0,
                "debate_direction": 0.0,
                "argument_scores": [],
                "debate_winner": "none",
                "model_signal": 0.0,
            },
            "messages": [("assistant", "[Judge] No debate — skipping scoring")],
        }

    Path("logs/verdicts").mkdir(parents=True, exist_ok=True)

    llm = get_llm()
    if llm is None:
        return {
            "weight_breakdown": {
                "phobert": 0.0,
                "coolant": 0.0,
                "evidence": 0.0,
                "debate": 0.0,
                "phobert_conf": 0.0,
                "coolant_conf": 0.0,
                "debate_conf": 0.0,
                "debate_direction": 0.0,
                "argument_scores": [],
                "debate_winner": "none (no LLM)",
                "model_signal": 0.0,
            },
            "messages": [("assistant", "[Judge] No LLM — skipping debate scoring")],
        }

    convergence_note = ""
    if debate_converged and debate_agreed_verdict:
        if len(model_detail) >= 2:
            convergence_note = (
                f"\nDEBATE CONVERGENCE: Both advocates agreed on verdict={debate_agreed_verdict}. "
                "Treat this as a strong prior.\n"
            )
        else:
            convergence_note = (
                f"\nDEBATE CONVERGENCE: Both advocates agreed on verdict={debate_agreed_verdict}. "
                "CAUTION: only one model was available, so both sides may have anchored on the "
                "same single signal — do NOT treat this convergence as independent agreement.\n"
            )
    if len(model_detail) == 1:
        convergence_note += (
            "\nSINGLE-MODEL MODE: chỉ có một model khả dụng — kết quả nó là tín hiệu yếu, "
            "không phải prior mạnh. Ưu tiên bằng chứng khi chấm điểm evidence_grounding.\n"
        )
    user = (
        f"CLAIM:\n{statement}\n\n"
        f"MODEL PREDICTIONS (các model khả dụng, kèm xác suất):\n{_format_models(model_results)}\n\n"
        f"EVIDENCE (TRUSTED):\n{_format_evidence(evidence_real)}\n\n"
        f"EVIDENCE (FLAGGED/FACT-CHECK):\n{_format_evidence(evidence_fake)}\n\n"
        f"DEBATE TRANSCRIPT:\n{_format_debate_turns(debate_turns)}\n"
        f"{convergence_note}"
    )

    try:
        resp = llm.invoke([("system", JUDGE_SYSTEM_PROMPT), ("user", user)])
        data = parse_json(getattr(resp, "content", "") or "") or {}
    except Exception:
        return {
            "weight_breakdown": {
                "phobert": 0.0,
                "coolant": 0.0,
                "evidence": 0.0,
                "debate": 0.0,
                "phobert_conf": 0.0,
                "coolant_conf": 0.0,
                "debate_conf": 0.0,
                "debate_direction": 0.0,
                "argument_scores": [],
                "debate_winner": "unknown (LLM error)",
                "model_signal": 0.0,
            },
            "messages": [("assistant", "[Judge] LLM error — skipping debate scoring")],
        }

    turn_scores = data.get("turn_scores", [])
    explanation_data = data.get("explanation", {})
    debate_winner = explanation_data.get("debate_winner", "unknown")

    # Compute model signal direction
    ph_conf = 0.0
    co_conf = 0.0
    ph_avail = False
    co_avail = False
    for m in model_results:
        if m.get("model") == "phobert_vifactcheck" and m.get("available"):
            ph_conf = m.get("confidence", 0.0)
            ph_avail = True
        elif m.get("model") == "coolant" and m.get("available"):
            co_conf = m.get("confidence", 0.0)
            co_avail = True

    model_signal = 0.0
    if ph_avail:
        ph_label = str(
            model_detail.get("phobert_vifactcheck", {}).get("label", "")
        ).upper()
        if ph_label in REAL_MODEL_LABELS:
            model_signal += ph_conf
        elif ph_label in FAKE_MODEL_LABELS:
            model_signal -= ph_conf
    if co_avail:
        co_label = str(model_detail.get("coolant", {}).get("label", "")).upper()
        if co_label in REAL_MODEL_LABELS:
            model_signal += co_conf
        elif co_label in FAKE_MODEL_LABELS:
            model_signal -= co_conf
    model_signal = max(-1.0, min(1.0, model_signal))

    # Debate signal
    debate_direction = 0.0
    debate_conf = 0.0
    if debate_turns:
        if debate_winner == "real_advocate":
            debate_direction = 1.0
        elif debate_winner == "fake_advocate":
            debate_direction = -1.0
        debate_conf = 0.7 if debate_converged else 0.4

    _jw_ph = float(os.getenv("FACTCHECK_JUDGE_PHOBERT_WEIGHT", "0.35"))
    _jw_co = float(os.getenv("FACTCHECK_JUDGE_COOLANT_WEIGHT", "0.35"))
    _jw_ev = float(os.getenv("FACTCHECK_JUDGE_EVIDENCE_WEIGHT", "0.15"))
    _jw_db = float(os.getenv("FACTCHECK_JUDGE_DEBATE_WEIGHT", "0.15"))
    weight_breakdown = {
        "phobert": _jw_ph if ph_avail else 0.0,
        "coolant": _jw_co if co_avail else 0.0,
        "evidence": _jw_ev,
        "debate": _jw_db,
        "phobert_conf": ph_conf,
        "coolant_conf": co_conf,
        "debate_conf": debate_conf,
        "debate_direction": debate_direction,
        "argument_scores": turn_scores,
        "debate_winner": debate_winner,
        "model_signal": model_signal,
        "phobert_label": model_detail.get("phobert_vifactcheck", {}).get("label"),
        "phobert_probabilities": model_detail.get("phobert_vifactcheck", {}).get(
            "probabilities"
        ),
        "coolant_label": model_detail.get("coolant", {}).get("label"),
        "coolant_probabilities": model_detail.get("coolant", {}).get("probabilities"),
    }

    _write_verdict_log(
        request_id, {"weight_breakdown": weight_breakdown, "turn_scores": turn_scores}
    )

    return {
        "weight_breakdown": weight_breakdown,
        "messages": [
            (
                "assistant",
                f"[Judge] {len(turn_scores)} turns scored, winner={debate_winner}",
            )
        ],
    }


# ── A2A service wrapper ─────────────────────────────────────────────────────
from ..a2a_server import AgentCardConfig, BaseTaskHandler, run_server


class JudgeAgentHandler(BaseTaskHandler):
    """A2A TaskHandler exposing :func:`judge_agent` over HTTP (port 9009)."""

    agent_card_config = AgentCardConfig(
        name="judge_agent",
        description="Scores debate turns on 1-5 dimensions; computes weight breakdown",
        version="1.0",
        skills=[
            {
                "id": "judging",
                "name": "Debate Judging",
                "description": "Score turns and compute weighted verdict breakdown",
            }
        ],
        port=settings.a2a_port_judge,
    )

    async def agent_fn(self, state: FactCheckState) -> dict:
        return judge_agent(state)


if __name__ == "__main__":
    run_server(JudgeAgentHandler(), JudgeAgentHandler.agent_card_config)
