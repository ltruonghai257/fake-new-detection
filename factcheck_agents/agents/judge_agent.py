"""Judge agent: scores debate turns on 1-5 dimensions and produces weight breakdown.

Does NOT produce a final verdict — that's done by expert_agent.
This node only scores the debate and computes component signals for downstream.
"""

from __future__ import annotations

import json
import logging
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
from .debate_utils import _assess_source_unanimity
from .llm import get_llm, parse_json

logger = logging.getLogger(__name__)

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
    "- TÍN HIỆU MODEL: PhoBERT chỉ là tham khảo YẾU (độ tin cậy thấp) — đừng dùng nó làm trụ cột; ưu tiên BẰNG CHỨNG và COOLANT. PhoBERT đo độ đúng của claim theo văn bản; COOLANT đo mức nhất quán "
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


JUDGE_MODERATOR_PROMPT = (
    "Bạn là GIÁM KHẢO điều phối một cuộc tranh luận tự do REAL vs FAKE về một claim. "
    "Không có thứ tự lượt cố định. Sau mỗi phát biểu, chọn ai nói tiếp:\n"
    '- "real" hoặc "fake": bên có điểm cần phản bác/làm rõ nhất; một bên CÓ THỂ nói liên tiếp '
    "hoặc xen ngang nếu bên kia chưa trả lời được luận điểm quan trọng.\n"
    '- "conclude": CHỈ khi một bên thắng rõ, hai bên đồng thuận, hoặc tranh luận đã lặp, '
    "không còn ý mới.\n"
    'Trả về DUY NHẤT JSON: {"next": "real|fake|conclude", "reason": "một câu tiếng Việt"}'
)


def judge_pick_next(state: FactCheckState, turns: list) -> tuple[str | None, str]:
    """Moderator: pick next speaker ('real'|'fake') or 'conclude'. None = no decision."""
    llm = get_llm()
    if llm is None or not turns:
        return None, ""
    user = (
        f"CLAIM:\n{state.get('statement', '')}\n\n"
        f"TRANH LUẬN ĐẾN NAY:\n{_format_debate_turns(turns)}\n"
    )
    try:
        resp = llm.invoke([("system", JUDGE_MODERATOR_PROMPT), ("user", user)])
        data = parse_json(getattr(resp, "content", "") or "") or {}
        nxt = str(data.get("next", "")).lower()
        return (nxt if nxt in {"real", "fake", "conclude"} else None), str(data.get("reason", ""))
    except Exception as exc:
        logger.error("[Judge] moderator failed, alternating: %s", exc)
        return None, ""


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

    Path("logs/verdicts").mkdir(parents=True, exist_ok=True)

    llm = get_llm()
    llm_failed = None  # reason string when the judge LLM cannot score the debate
    if llm is None:
        llm_failed = "không có LLM"
    elif not debate_turns or all(t.get("error") for t in debate_turns):
        llm_failed = "tranh luận không diễn ra"

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
    if any(t.get("error") for t in debate_turns) and not llm_failed:
        convergence_note += (
            "\nTRANH LUẬN BỊ NGẮT GIỮA CHỪNG (lỗi LLM/hạn mức). Chỉ chấm các lượt hợp lệ; "
            "nếu chưa đủ để phân thắng bại thì debate_winner=tie và dựa vào bằng chứng + model.\n"
        )
    user = (
        f"CLAIM:\n{statement}\n\n"
        f"MODEL PREDICTIONS (các model khả dụng, kèm xác suất):\n{_format_models(model_results)}\n\n"
        f"EVIDENCE (TRUSTED):\n{_format_evidence(evidence_real)}\n\n"
        f"EVIDENCE (FLAGGED/FACT-CHECK):\n{_format_evidence(evidence_fake)}\n\n"
        f"{_assess_source_unanimity(evidence_real, evidence_fake) or ''}\n\n"
        f"DEBATE TRANSCRIPT:\n{_format_debate_turns(debate_turns)}\n"
        f"{convergence_note}"
    )

    data: dict = {}
    if llm is not None and not llm_failed:
        try:
            resp = llm.invoke([("system", JUDGE_SYSTEM_PROMPT), ("user", user)])
            data = parse_json(getattr(resp, "content", "") or "") or {}
        except Exception as exc:
            logger.exception("[judge_agent] LLM failed, falling back to model+evidence fusion: %s", exc)
            llm_failed = str(exc)[:200]

    turn_scores = data.get("turn_scores", [])
    explanation_data = data.get("explanation", {})
    debate_winner = explanation_data.get("debate_winner", "unknown")
    if llm_failed:
        debate_winner = "unscored (LLM unavailable)"

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
    # PhoBERT is only a weak reference (low weight below). COOLANT gets its own
    # signed signal: +1 = image fully consistent with the claim, -1 = clear
    # mismatch. It measures consistency, so it is weighted below evidence.
    coolant_signal = 0.0
    if co_avail:
        co_probs = model_detail.get("coolant", {}).get("probabilities") or {}
        p_real = co_probs.get("REAL")
        if p_real is None:
            co_label = str(model_detail.get("coolant", {}).get("label", "")).upper()
            p_real = 1.0 if co_label in REAL_MODEL_LABELS else 0.0
        coolant_signal = 2.0 * float(p_real) - 1.0
    model_signal = max(-1.0, min(1.0, model_signal))

    # Debate signal
    debate_direction = 0.0
    debate_conf = 0.0
    if debate_turns and not llm_failed:
        if debate_winner == "real_advocate":
            debate_direction = 1.0
        elif debate_winner == "fake_advocate":
            debate_direction = -1.0
        debate_conf = 0.7 if debate_converged else 0.4
        # Strength of the win: gap between the sides' judge scores (evidence-
        # grounded argument quality), so a clear win counts more than a narrow one.
        side = {"real_advocate": [], "fake_advocate": []}
        for sc in turn_scores:
            try:
                side[sc.get("agent")].append(
                    (sc["factuality"] + sc["rebuttal_engagement"] + sc["evidence_grounding"]) / 15
                )
            except (KeyError, TypeError):
                pass
        if side["real_advocate"] and side["fake_advocate"]:
            gap = abs(
                sum(side["real_advocate"]) / len(side["real_advocate"])
                - sum(side["fake_advocate"]) / len(side["fake_advocate"])
            )
            debate_conf = min(1.0, max(debate_conf, 0.4 + gap))

    _jw_ph = float(os.getenv("FACTCHECK_JUDGE_PHOBERT_WEIGHT", "0.05"))
    _jw_co = float(os.getenv("FACTCHECK_JUDGE_COOLANT_WEIGHT", "0.30"))
    _jw_ev = float(os.getenv("FACTCHECK_JUDGE_EVIDENCE_WEIGHT", "0.15"))
    _jw_db = float(os.getenv("FACTCHECK_JUDGE_DEBATE_WEIGHT", "0.50"))
    # Deterministic fusion: this score decides the final binary verdict
    # (expert_agent enforces it), so debate/models/evidence truly count.
    n_ev = len(evidence_real) + len(evidence_fake)
    evidence_signal = (len(evidence_real) - len(evidence_fake)) / n_ev if n_ev else 0.0
    w_ph = _jw_ph if ph_avail else 0.0
    w_co = _jw_co if co_avail else 0.0
    w_ev = _jw_ev if n_ev else 0.0
    w_db = _jw_db if debate_direction != 0.0 else 0.0
    denom = w_ph + w_co + w_ev + w_db
    fused_score = (
        (w_ph * model_signal + w_co * coolant_signal + w_ev * evidence_signal + w_db * debate_direction * min(1.0, debate_conf / 0.7))
        / denom
        if denom
        else 0.0
    )
    fused_score = max(-1.0, min(1.0, fused_score))
    fused_verdict = (
        "REAL" if fused_score > FUSED_MARGIN else "FAKE" if fused_score < -FUSED_MARGIN else "NEI"
    )

    weight_breakdown = {
        "fused_score": round(fused_score, 4),
        "fused_verdict": fused_verdict,
        "evidence_signal": round(evidence_signal, 4),
        "coolant_signal": round(coolant_signal, 4),
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
        "judge_fallback": llm_failed,
        "phobert_label": model_detail.get("phobert_vifactcheck", {}).get("label"),
        "phobert_probabilities": model_detail.get("phobert_vifactcheck", {}).get(
            "probabilities"
        ),
        "coolant_similarity": model_detail.get("coolant", {}).get("similarity_score"),
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
                (
                    f"[Judge] LLM không dùng được ({llm_failed}) — tự kết luận từ COOLANT + bằng chứng + PhoBERT: "
                    f"{fused_verdict} (điểm {fused_score:+.2f})"
                    if llm_failed
                    else f"[Judge] {len(turn_scores)} turns scored, winner={debate_winner}"
                ),
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
