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


JUDGE_SYSTEM_PROMPT = (
    "Bạn là GIÁM KHẢO tranh luận, trung lập, trong một phiên xác minh tin tức tiếng Việt. "
    "Bạn KHÔNG ra phán quyết cuối về claim — nhiệm vụ của bạn là chấm điểm màn tranh luận. "
    "Chỉ đánh giá dựa trên dữ liệu được cung cấp, TUYỆT ĐỐI không dùng kiến thức ngoài.\n\n"
    "Bạn được cung cấp:\n"
    "- Claim cần xác minh (tiếng Việt)\n"
    "- Kết quả dự đoán của PhoBERT và COOLANT kèm phân phối xác suất đầy đủ theo từng lớp\n"
    "- Bằng chứng web từ nguồn tin cậy và nguồn bị gắn cờ (mỗi nguồn có tier)\n"
    "- Biên bản tranh luận giữa luật sư phe REAL và luật sư phe FAKE\n"
    "- Việc tranh luận có hội tụ hay không và verdict được đồng thuận (nếu có)\n\n"
    "NHIỆM VỤ:\n"
    "1. Chấm điểm TỪNG lượt tranh luận trên ba tiêu chí (số nguyên 1-5):\n"
    "   - factuality: các khẳng định trong lượt đó có đúng sự thật theo bằng chứng không?\n"
    "   - rebuttal_engagement: có phản bác trực tiếp lập luận gần nhất của đối thủ không?\n"
    "   - evidence_grounding: có bám vào kết quả model/bằng chứng đã cho không (không bịa)?\n"
    "   Trừ điểm mạnh mọi lượt trích số liệu không có trong kết quả model, hoặc bịa nguồn.\n"
    "2. Xác định bên thắng dựa trên điểm số: 'real_advocate', 'fake_advocate', hoặc 'tie'.\n"
    "3. Viết explanation gồm các mục:\n"
    "   - model_summary: PhoBERT và COOLANT nói gì, kèm phân phối xác suất, có thống nhất không.\n"
    "   - debate_winner: bên nào thắng và tại sao (dựa trên điểm số).\n"
    "   - evidence_summary: tóm tắt bằng chứng then chốt.\n"
    "   - confidence_breakdown: mức đóng góp của PhoBERT, COOLANT, bằng chứng và tranh luận "
    "(bốn trọng số là số thực, PHẢI cộng lại bằng 1.0).\n\n"
    "Nếu debate_converged=true, coi agreed_verdict là tiên nghiệm mạnh khi cân nhắc, "
    "nhưng việc chấm điểm phải phản ánh chất lượng lập luận thực tế của từng lượt.\n\n"
    "QUY TẮC:\n"
    "- Chỉ dùng dữ liệu được cung cấp; không bịa số liệu hay nguồn.\n"
    "- Viết model_summary và evidence_summary bằng tiếng Việt.\n\n"
    "CHỈ trả về DUY NHẤT một object JSON hợp lệ, không markdown, không văn bản trước/sau:\n"
    "{\n"
    '  "turn_scores": [\n'
    '    {"agent": "real_advocate", "round": 0, "factuality": 4, "rebuttal_engagement": 3, "evidence_grounding": 5}\n'
    "  ],\n"
    '  "explanation": {\n'
    '    "model_summary": "PhoBERT/COOLANT nói gì, kèm phân phối xác suất.",\n'
    '    "debate_winner": "real_advocate | fake_advocate | tie",\n'
    '    "evidence_summary": "Tóm tắt bằng chứng then chốt.",\n'
    '    "confidence_breakdown": {"phobert": 0.3, "coolant": 0.3, "evidence": 0.2, "debate": 0.2}\n'
    "  }\n"
    "}\n"
    "Không thêm giải thích ngoài JSON."
)


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
        convergence_note = (
            f"\nDEBATE CONVERGENCE: Both advocates agreed on verdict={debate_agreed_verdict}. "
            "Treat this as a strong prior.\n"
        )
    user = (
        f"CLAIM:\n{statement}\n\n"
        f"MODEL PREDICTIONS (PhoBERT + COOLANT with probabilities):\n{_format_models(model_results)}\n\n"
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
