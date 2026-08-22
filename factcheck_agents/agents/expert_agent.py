"""Expert agent: final reviewer who reads all evidence, debate transcripts,
judge scores, and model results, then produces the authoritative verdict with
a detailed explanation of why.

Role: chuyên gia kiểm duyệt thông tin — consumes the full pipeline output
and delivers the final yes/no/unverifiable decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..state import Evidence, FactCheckState, Verdict, binary_to_vi, canonicalize_binary
from .llm import get_llm, parse_json

EXPERT_SYSTEM_PROMPT = (
    "Bạn là CHUYÊN GIA KIỂM DUYỆT THÔNG TIN cao cấp, người ra phán quyết CUỐI CÙNG. "
    "Bạn trung lập tuyệt đối và chỉ kết luận dựa trên hồ sơ được cung cấp — "
    "TUYỆT ĐỐI không dùng kiến thức ngoài hay suy diễn ngoài dữ liệu.\n\n"
    "Bạn nhận được TOÀN BỘ hồ sơ của một vụ kiểm tra thông tin:\n"
    "- Claim cần xác minh (tiếng Việt)\n"
    "- Kết quả phân tích của PhoBERT và COOLANT, kèm phân phối xác suất theo từng lớp\n"
    "- Bằng chứng từ nguồn chính thống và nguồn bị gắn cờ (mỗi nguồn có tier)\n"
    "- Biên bản tranh luận giữa luật sư phe THẬT (REAL) và phe GIẢ (FAKE)\n"
    "- Điểm giám khảo chấm cho từng lượt tranh luận\n\n"
    "NHIỆM VỤ:\n"
    "1. Đánh giá TOÀN DIỆN mọi nguồn thông tin, không thiên vị bên nào.\n"
    "2. Giải thích CHI TIẾT lý do đi đến kết luận, trích dẫn cụ thể:\n"
    "   - PhoBERT nói gì, COOLANT nói gì (label + confidence + xác suất), có thống nhất không?\n"
    "   - Bằng chứng nào ủng hộ, bằng chứng nào phản bác? Ưu tiên nguồn tier cao.\n"
    "   - Bên nào thắng tranh luận theo điểm giám khảo và tại sao?\n"
    "3. Đưa ra phán quyết cuối cùng.\n\n"
    "QUY TẮC RA QUYẾT ĐỊNH:\n"
    "- Confidence của model là tín hiệu, KHÔNG phải mệnh lệnh — bằng chứng tier cao mâu thuẫn có thể lấn át.\n"
    "- Nếu PhoBERT và COOLANT mâu thuẫn, nói rõ bạn tin model nào hơn VÀ tại sao (dựa trên confidence, "
    "xác suất, và sự tương thích với bằng chứng).\n"
    "- Nếu bằng chứng mỏng hoặc mâu thuẫn không giải quyết được, THẲNG THẮN chọn 'UNVERIFIED' thay vì đoán.\n"
    "- Chọn 'MISLEADING' khi claim có phần đúng nhưng bị bóp méo, thiếu ngữ cảnh, hoặc gây hiểu sai.\n"
    "- Chỉ dùng dữ liệu đã cho. Không bịa citation, số liệu hay sự thật. "
    "'citations' chỉ lấy từ URL/tiêu đề trong bằng chứng đã cung cấp.\n\n"
    "YÊU CẦU ĐẦU RA:\n"
    "- 'rationale' PHẢI dài ít nhất 200 từ, chia thành 4 mục rõ ràng, mỗi mục mở đầu đúng nhãn sau: "
    "'Phân tích Model:', 'Phân tích Bằng chứng:', 'Phân tích Tranh luận:', 'Kết luận:'.\n"
    "- 'confidence' phản ánh mức chắc chắn thực tế, không mặc định cao.\n"
    "- rationale và recommendation viết bằng tiếng Việt.\n\n"
    "CHỈ trả về DUY NHẤT một object JSON hợp lệ, không markdown, không văn bản trước/sau:\n"
    "{\n"
    '  "label": "TRUE | FALSE | MISLEADING | UNVERIFIED",\n'
    '  "confidence": 0.85,\n'
    '  "rationale": "Giải thích ≥ 200 từ, gồm 4 mục: Phân tích Model / Phân tích Bằng chứng / Phân tích Tranh luận / Kết luận.",\n'
    '  "citations": ["url hoặc tiêu đề từ bằng chứng đã cho"],\n'
    '  "recommendation": "Khuyến nghị cho người dùng, tiếng Việt.",\n'
    '  "evidence_quality": "strong | moderate | weak | none",\n'
    '  "model_agreement": "agree | disagree | partial | none"\n'
    "}\n"
    "Không thêm giải thích ngoài JSON."
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
            f"- {model}: KẾT LUẬN '{label}' (độ tin cậy {confidence:.1%}). "
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

    convergence_note = ""
    if debate_converged and debate_agreed_verdict:
        convergence_note = (
            f"\nTRANH LUẬN ĐÃ ĐỒNG THUẬN: Cả hai bên đồng ý verdict={debate_agreed_verdict}. "
            "Đây là tín hiệu mạnh — cân nhắc kỹ trước khi bác bỏ.\n"
        )

    user = (
        f"===== HỒ SƠ KIỂM TRA THÔNG TIN =====\n\n"
        f"TUYÊN BỐ CẦN KIỂM TRA:\n{statement}\n\n"
        f"----- KẾT QUẢ PHÂN TÍCH MODEL (PhoBERT + COOLANT) -----\n"
        f"{_format_expert_models(model_results)}\n\n"
        f"----- BẰNG CHỨNG TỪ NGUỒN CHÍNH THỐNG -----\n"
        f"{_fmt_ev(evidence_real, 'chính thống')}\n\n"
        f"----- BẰNG CHỨNG TỪ NGUỒN KHÔNG CHÍNH THỐNG -----\n"
        f"{_fmt_ev(evidence_fake, 'không chính thống')}\n\n"
        f"----- BIÊN BẢN TRANH LUẬN -----\n"
        f"{_fmt_debate(debate_turns)}\n"
        f"{convergence_note}"
        f"----- ĐIỂM GIÁM KHẢO -----\n"
        f"{_format_expert_judge_scores(judge_turn_scores)}\n"
        f"Debate winner: {judge_breakdown.get('debate_winner', 'unknown')}\n"
        f"Debate direction: {judge_breakdown.get('debate_direction', 0.0)}\n\n"
        f"===== YÊU CẦU =====\n"
        f"Hãy đưa ra kết luận cuối cùng với giải thích CHI TIẾT. "
        f"Rationale phải ≥ 200 từ, chia mục rõ ràng. "
        f"Trả về JSON.\n"
    )

    try:
        resp = llm.invoke([("system", EXPERT_SYSTEM_PROMPT), ("user", user)])
        data = parse_json(getattr(resp, "content", "") or "") or {}
    except Exception as exc:
        # LLM error — preserve the judge verdict unchanged (graceful degrade)
        return {
            "messages": [("assistant", f"[Expert] Lỗi LLM: {exc}")],
        }

    # Preserve judge's authoritative binary verdict — expert only augments rationale
    existing_verdict = state.get("verdict") or {}
    label = str(existing_verdict.get("label", data.get("label", "UNVERIFIED"))).upper()
    binary = existing_verdict.get("verdict_binary") or canonicalize_binary(label)
    label_vi = existing_verdict.get("verdict_label_vi") or binary_to_vi(binary)
    confidence = float(
        existing_verdict.get("confidence") or data.get("confidence", 0.5) or 0.5
    )

    # Convergence override: boost confidence when debate converged and expert agrees
    if debate_converged and debate_agreed_verdict and binary == debate_agreed_verdict:
        confidence = max(confidence, 0.85)

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
