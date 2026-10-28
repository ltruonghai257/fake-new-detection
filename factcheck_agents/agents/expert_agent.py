"""Expert agent: final reviewer who reads all evidence, debate transcripts,
judge scores, and model results, then produces the authoritative verdict with
a detailed explanation of why.

Role: chuyên gia kiểm duyệt thông tin — consumes the full pipeline output
and delivers the final yes/no/unverifiable decision.
"""

from __future__ import annotations

import logging

import json
from pathlib import Path
from typing import List

from ..state import Evidence, FactCheckState, Verdict, binary_to_vi, canonicalize_binary
from .debate_utils import _assess_source_unanimity
from .llm import get_llm, parse_json

logger = logging.getLogger(__name__)

EXPERT_SYSTEM_PROMPT = (
    "Bạn là CHUYÊN GIA KIỂM DUYỆT THÔNG TIN, người ra phán quyết CUỐI CÙNG. "
    "Trung lập tuyệt đối — chỉ kết luận từ hồ sơ được cung cấp, không dùng kiến thức ngoài.\n\n"
    "HỒ SƠ (trong tin nhắn user): claim; tín hiệu model (PhoBERT: fact-check văn bản; "
    "COOLANT: mức nhất quán claim–ảnh, 'FAKE' = claim và ảnh không khớp, chưa kết luận tin giả); "
    "bằng chứng web kèm tier; biên bản tranh luận REAL/FAKE; điểm giám khảo.\n\n"
    "QUY TẮC RA QUYẾT ĐỊNH:\n"
    "- PhoBERT chỉ là tham khảo yếu; trọng tâm là BẰNG CHỨNG và COOLANT (mức nhất quán claim–ảnh).\n"
    "- Tín hiệu model là luận điểm tham chiếu, không phải mệnh lệnh — bằng chứng tier cao "
    "mâu thuẫn có thể lấn át. Hai model đo hai khía cạnh khác nhau: kết quả khác nhau "
    "không hẳn là mâu thuẫn — giải thích theo đúng vai trò từng model, đối chiếu bằng chứng.\n"
    "- Bằng chứng mỏng hoặc mâu thuẫn không giải quyết được → chọn 'UNVERIFIED', đừng đoán.\n"
    "- 'MISLEADING' khi claim có phần đúng nhưng bị bóp méo hoặc thiếu ngữ cảnh.\n"
    "- 'citations' chỉ lấy URL/tiêu đề trong bằng chứng đã cho; không bịa số liệu hay nguồn.\n\n"
    "ĐẦU RA — DUY NHẤT một object JSON hợp lệ, không markdown, không văn bản trước/sau. "
    "'rationale' ≥ 200 từ, đúng 4 mục mở đầu bằng nhãn: 'Phân tích Model:', 'Phân tích Bằng chứng:', "
    "'Phân tích Tranh luận:', 'Kết luận:'. rationale và recommendation viết tiếng Việt; "
    "'confidence' phản ánh mức chắc chắn thực tế:\n"
    "{\n"
    '  "label": "TRUE | FALSE | MISLEADING | UNVERIFIED",\n'
    '  "confidence": 0.85,\n'
    '  "rationale": "Giải thích ≥ 200 từ, gồm 4 mục: Phân tích Model / Phân tích Bằng chứng / Phân tích Tranh luận / Kết luận.",\n'
    '  "citations": ["url hoặc tiêu đề từ bằng chứng đã cho"],\n'
    '  "recommendation": "Khuyến nghị cho người dùng, tiếng Việt.",\n'
    '  "evidence_quality": "strong | moderate | weak | none",\n'
    '  "model_agreement": "agree | disagree | partial | none"\n'
    "}"
)

EXPERT_FALLBACK_PROMPT = (
    "Bạn là chuyên gia kiểm duyệt thông tin. Dựa trên các thông tin có sẵn, "
    "hãy đưa ra kết luận cuối cùng. Không có LLM — dùng rule-based reasoning. "
    "Trả về JSON."
)


def _format_expert_models(model_results: List[dict]) -> str:
    if not model_results:
        return "(Không có kết quả phân tích model)"
    lines = []
    for r in model_results:
        model = r.get("model", "unknown").upper()
        if not r.get("available"):
            lines.append(f"- {model}: KHÔNG KHẢ DỤNG — {r.get('note', '')}")
            continue
        label = r.get("label", "N/A")
        confidence = r.get("confidence", 0.0)
        probs = r.get("probabilities") or {}
        prob_str = ", ".join(
            f"{k}={v:.1%}" for k, v in sorted(probs.items(), key=lambda x: -x[1])
        )
        lines.append(
            f"- {model}: TÍN HIỆU '{label}' (độ tin cậy {confidence:.1%}). "
            f"Xác suất: {prob_str}"
        )
    return "\n".join(lines)


def _format_expert_judge_scores(turn_scores: List[dict]) -> str:
    if not turn_scores:
        return "(Không có điểm giám khảo)"
    lines = []
    for s in turn_scores:
        agent = s.get("agent", "?")
        round_num = s.get("round", 0)
        lines.append(
            f"- {agent} (vòng {round_num}): "
            f"factuality={s.get('factuality', '?')}, "
            f"rebuttal={s.get('rebuttal_engagement', '?')}, "
            f"evidence={s.get('evidence_grounding', '?')}"
        )
    return "\n".join(lines)


_FUSED_TO_LABEL = {"REAL": "TRUE", "FAKE": "FALSE", "NEI": "UNVERIFIED"}


def _fused_note(bd: dict) -> str:
    fv = bd.get("fused_verdict")
    if not fv:
        return ""
    return (
        f"ĐIỂM HỢP NHẤT (model + bằng chứng + tranh luận, đã tính trọng số): "
        f"{bd.get('fused_score')} → {fv}. Đây là KẾT LUẬN BẮT BUỘC của hệ thống; "
        f"rationale phải giải thích vì sao dẫn tới {fv}."
    )


def expert_agent(state: FactCheckState) -> dict:
    statement = state["statement"]
    model_results = state.get("model_results") or []
    evidence_real = state.get("evidence_real") or []
    evidence_fake = state.get("evidence_fake") or []
    debate_turns = state.get("debate_turns") or []
    debate_converged = state.get("debate_converged", False)
    debate_agreed_verdict = state.get("debate_agreed_verdict")
    judge_breakdown = state.get("weight_breakdown") or {}
    judge_turn_scores = judge_breakdown.get("argument_scores") or []

    request_id = state.get("request_id", "unknown")

    all_evidence = evidence_real + evidence_fake

    def _fmt_ev(ev_list: List[Evidence], label: str) -> str:
        if not ev_list:
            return f"(Không có bằng chứng {label})"
        lines = []
        for i, e in enumerate(ev_list, 1):
            tier = e.get("source_tier", "unknown").upper()
            lines.append(
                f"[{i}] [{tier}] {e.get('title', '')} — {e.get('url', '')}\n"
                f"    {e.get('snippet', '')[:300]}"
            )
        return "\n".join(lines)

    def _fmt_debate(turns: List[dict]) -> str:
        if not turns:
            return "(Không có tranh luận)"
        lines = []
        for t in turns:
            agent = t.get("agent", "?")
            round_num = t.get("round", 0)
            verdict = t.get("verdict", "?")
            argument = t.get("argument", "")[:400]
            con = f" (nhượng bộ: {t.get('concession')})" if t.get("concession") else ""
            lines.append(
                f"[Vòng {round_num}] {agent} (verdict={verdict}):\n{argument}{con}\n"
            )
        return "\n".join(lines)

    llm = get_llm()
    if llm is None:
        # No LLM — preserve the judge verdict unchanged (graceful degrade)
        return {
            "messages": [
                ("assistant", "[Expert] Không có LLM — không thể phân tích chuyên sâu")
            ],
        }

    n_available_models = sum(1 for m in model_results if m.get("available"))

    convergence_note = ""
    if debate_converged and debate_agreed_verdict:
        if n_available_models >= 2:
            convergence_note = (
                f"\nTRANH LUẬN ĐÃ ĐỒNG THUẬN: Cả hai bên đồng ý verdict={debate_agreed_verdict}. "
                "Đây là tín hiệu mạnh — cân nhắc kỹ trước khi bác bỏ.\n"
            )
        else:
            convergence_note = (
                f"\nTRANH LUẬN ĐÃ ĐỒNG THUẬN: Cả hai bên đồng ý verdict={debate_agreed_verdict}. "
                "THẬN TRỌNG: chỉ có MỘT model khả dụng nên cả hai bên có thể neo vào cùng một "
                "tín hiệu — đồng thuận này KHÔNG phải bằng chứng độc lập.\n"
            )
    if n_available_models == 1:
        convergence_note += (
            "\nCHẾ ĐỘ SINGLE-MODEL: chỉ một model cho kết quả — coi nó là tín hiệu yếu. "
            "Phán quyết phải dựa chủ yếu vào bằng chứng; nếu bằng chứng mỏng, chọn UNVERIFIED.\n"
        )

    user = (
        f"===== HỒ SƠ KIỂM TRA THÔNG TIN =====\n\n"
        f"TUYÊN BỐ CẦN KIỂM TRA:\n{statement}\n\n"
        f"----- KẾT QUẢ PHÂN TÍCH MODEL (các model khả dụng) -----\n"
        f"{_format_expert_models(model_results)}\n\n"
        f"----- BẰNG CHỨNG TỪ NGUỒN CHÍNH THỐNG -----\n"
        f"{_fmt_ev(evidence_real, 'chính thống')}\n\n"
        f"----- BẰNG CHỨNG TỪ NGUỒN KHÔNG CHÍNH THỐNG -----\n"
        f"{_fmt_ev(evidence_fake, 'không chính thống')}\n\n"
        f"{_assess_source_unanimity(evidence_real, evidence_fake) or ''}\n\n"
        f"----- BIÊN BẢN TRANH LUẬN -----\n"
        f"{_fmt_debate(debate_turns)}\n"
        f"{convergence_note}"
        f"----- ĐIỂM GIÁM KHẢO -----\n"
        f"{_format_expert_judge_scores(judge_turn_scores)}\n"
        f"Debate winner: {judge_breakdown.get('debate_winner', 'unknown')}\n"
        f"Debate direction: {judge_breakdown.get('debate_direction', 0.0)}\n"
        f"{_fused_note(judge_breakdown)}\n"
        f"===== YÊU CẦU =====\n"
        f"Hãy đưa ra kết luận cuối cùng với giải thích CHI TIẾT. "
        f"Rationale phải ≥ 200 từ, chia mục rõ ràng. "
        f"Trả về JSON.\n"
    )

    try:
        resp = llm.invoke([("system", EXPERT_SYSTEM_PROMPT), ("user", user)])
        data = parse_json(getattr(resp, "content", "") or "") or {}
    except Exception as exc:
        logger.exception("[expert_agent] failed: %s", exc)
        # LLM error — conclude from the judge's deterministic fusion, never leave
        # the user without a verdict.
        fv = judge_breakdown.get("fused_verdict")
        if fv not in _FUSED_TO_LABEL or state.get("verdict"):
            return {"messages": [("assistant", f"[Expert] Lỗi LLM: {exc}")]}
        score = float(judge_breakdown.get("fused_score", 0.0))
        verdict = Verdict(
            label=_FUSED_TO_LABEL[fv],
            verdict_binary=fv,
            verdict_label_vi=binary_to_vi(fv),
            confidence=round(min(0.9, 0.5 + abs(score) / 2), 3),
            rationale=(
                "⚠️ Quá trình tranh luận / LLM bị gián đoạn (ví dụ hết hạn mức API) nên giám khảo "
                f"tự kết luận từ tín hiệu có sẵn. Điểm tổng hợp {score:+.2f}: "
                f"COOLANT (nhất quán ảnh–văn bản) {judge_breakdown.get('coolant_signal', 0):+.2f}, "
                f"bằng chứng {judge_breakdown.get('evidence_signal', 0):+.2f}, "
                f"PhoBERT {judge_breakdown.get('model_signal', 0):+.2f}. "
                f"Lỗi: {str(exc)[:160]}"
            ),
            citations=[x.get("url", "") for x in all_evidence if x.get("url")][:5],
            recommendation="Chạy lại khi LLM khả dụng để có phân tích tranh luận đầy đủ.",
            explanation={
                "model_summary": _format_expert_models(model_results),
                "evidence_quality": "unknown",
                "model_agreement": "none",
                "expert_name": "giám khảo (chế độ dự phòng)",
            },
        )
        return {
            "verdict": verdict,
            "messages": [("assistant", f"[Expert] LLM lỗi — giám khảo tự kết luận: {fv}")],
        }

    # Preserve judge's authoritative binary verdict — expert only augments rationale
    existing_verdict = state.get("verdict") or {}
    label = str(existing_verdict.get("label", data.get("label", "UNVERIFIED"))).upper()
    binary = existing_verdict.get("verdict_binary") or canonicalize_binary(label)
    label_vi = existing_verdict.get("verdict_label_vi") or binary_to_vi(binary)
    confidence = float(
        existing_verdict.get("confidence") or data.get("confidence", 0.5) or 0.5
    )

    # Judge's fused score is authoritative: it overrides the LLM's own label.
    fv = judge_breakdown.get("fused_verdict")
    if fv in _FUSED_TO_LABEL and not existing_verdict:
        binary = fv
        label_vi = binary_to_vi(binary)
        llm_label = str(data.get("label", "")).upper()
        label = llm_label if canonicalize_binary(llm_label) == fv else _FUSED_TO_LABEL[fv]
        confidence = min(0.95, 0.5 + abs(float(judge_breakdown.get("fused_score", 0.0))) / 2)

    # Convergence override: boost confidence when debate converged and expert agrees.
    # Only counts as independent agreement when ≥2 model signals were available —
    # with a single model both advocates may have anchored on the same signal.
    if debate_converged and debate_agreed_verdict and binary == debate_agreed_verdict:
        if n_available_models >= 2:
            confidence = max(confidence, 0.85)
        else:
            confidence = min(confidence, 0.7)

    verdict = Verdict(
        label=label,
        verdict_binary=binary,
        verdict_label_vi=label_vi,
        confidence=round(confidence, 3),
        rationale=str(data.get("rationale", existing_verdict.get("rationale", ""))),
        citations=list(
            data.get("citations", [])
            or [e.get("url", "") for e in all_evidence if e.get("url")][:5]
        ),
        recommendation=str(data.get("recommendation", "")),
        explanation={
            "model_summary": _format_expert_models(model_results),
            "evidence_quality": data.get("evidence_quality", "moderate"),
            "model_agreement": data.get("model_agreement", "none"),
            "expert_name": "chuyên gia kiểm duyệt thông tin",
        },
        debate_transcript=debate_turns,
        model_detail={},
    )

    # Write expert log
    try:
        log_dir = Path("logs/expert")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{request_id}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "verdict": verdict,
                    "evidence_quality": data.get("evidence_quality"),
                    "model_agreement": data.get("model_agreement"),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass

    return {
        "verdict": verdict,
        "messages": [
            (
                "assistant",
                f"[Expert] {verdict['label']} ({verdict['confidence']:.2f}) — {verdict['verdict_label_vi']}",
            )
        ],
    }
