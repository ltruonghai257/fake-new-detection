"""Shared debate utilities for the real/fake advocate agents (D-08).

Extracted from the former ``debate_node.py``: advocate prompt templates,
evidence/model/history formatting helpers, JSON parsing, and the JSONL
turn logger. Both ``real_advocate`` and ``fake_advocate`` import from here.

D-09: COOLANT is injected as a ROLE-SPECIFIC debate weapon — each advocate
sees tailored guidance on how to wield or counter COOLANT's findings.
D-10: Anti-repetition — prior claims from history are extracted and listed
as forbidden territory, forcing advocates to produce genuinely new arguments.
D-11: Interaction enforcement — opponent arguments are presented in a
structured format requiring explicit citation → rebuttal → new point flow.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..state import Evidence

# ── Advocate prompts ─────────────────────────────────────────────────────────

REAL_ADVOCATE_PROMPT = (
    "Bạn là LUẬT SƯ BÀO CHỮA trong phiên tranh biện đối kháng xác minh tin tức tiếng Việt. "
    "Vị trí bảo vệ: claim là REAL. Đối thủ bảo vệ FAKE. "
    "Mục tiêu: mỗi lượt phải đẩy tranh luận tiến lên, không dậm chân.\n\n"
    "ĐẦU VÀO (trong tin nhắn user):\n"
    "- CLAIM: nội dung cần xác minh.\n"
    "- TÍN HIỆU MODEL: PhoBERT chỉ là tham khảo YẾU (độ tin cậy thấp) — đừng dùng nó làm trụ cột; ưu tiên BẰNG CHỨNG và COOLANT. PhoBERT đo độ đúng của claim theo văn bản; COOLANT đo mức nhất quán "
    "giữa claim và ảnh ('FAKE' = ảnh và claim không khớp, chưa kết luận tin giả). "
    "Luận điểm tham chiếu, không phải kết luận sự thật.\n"
    "- VŨ KHÍ COOLANT (DÀNH RIÊNG CHO BẠN): hướng dẫn cách DÙNG COOLANT để củng cố lập "
    "luận bảo vệ REAL. Đọc kỹ phần này — đây là lợi thế của bạn.\n"
    "- TOÀN BỘ BẰNG CHỨNG: tách NGUỒN CHÍNH THỐNG và NGUỒN KHÔNG CHÍNH THỐNG. "
    "Nếu TẤT CẢ nguồn cùng nói một điều (một phía, một domain, trích dẫn trùng) → NGHI NGỜ, "
    "không coi số lượng là xác nhận.\n"
    "- LẬP LUẬN ĐỐI THỦ: trình bày có cấu trúc để bạn phản bác chính xác.\n"
    "- LỊCH SỬ TRANH LUẬN: tất cả các lượt trước.\n"
    "- CÁC LUẬN ĐIỂM ĐÃ DÙNG (KHÔNG ĐƯỢC LẶP LẠI): danh sách luận điểm đã xuất hiện.\n\n"
    "CÁCH TRANH LUẬN (QUY TẮC CỨNG — VI PHẠM SẼ BỊ TÍNH LÀ THUA LƯỢT):\n"
    "1. TRÍCH DẪN → PHẢN BÁC → LUẬN ĐIỂM MỚI: mỗi lượt bắt buộc theo cấu trúc 3 phần:\n"
    "   a) Trích dẫn CHÍNH XÁC 1 ý của đối thủ (1-2 câu, trong ngoặc kép)\n"
    "   b) Phản bác trực tiếp ý đó — chỉ ra mâu thuẫn logic, thiếu bằng chứng, hoặc hệ quả sai\n"
    "   c) Đưa ra ÍT NHẤT MỘT luận điểm MỚI chưa từng xuất hiện trong LỊCH SỬ TRANH LUẬN\n"
    "2. KHÔNG LẶP LẠI: kiểm tra 'CÁC LUẬN ĐIỂM ĐÃ DÙNG' — nếu ý của bạn đã có trong đó, "
    "PHẢI đưa ra góc nhìn khác. Lặp lại = thua lượt.\n"
    "3. TÔN TRỌNG ĐỐI THỦ: nếu đối thủ đưa ra luận điểm hợp lệ dựa trên bằng chứng, "
    "BẮT BUỘC ghi nhận trong 'concession'. Phủ nhận bằng chứng rõ ràng = mất credibility.\n"
    "4. Được phép đổi verdict sang FAKE khi không còn phản bác được — ghi lý do trong "
    "'concession'. Mục tiêu là kết luận đúng, không phải thắng bằng mọi giá.\n\n"
    "Mỗi luận điểm phải dẫn BẰNG CHỨNG cụ thể (số thứ tự/nguồn/tier) — tranh luận là lớp kết luận cuối nên "
    "luận điểm không có bằng chứng sẽ bị giám khảo bỏ qua.\n"
    "QUY TẮC: chỉ dùng dữ liệu trong đầu vào, dùng đúng con số của TÍN HIỆU MODEL, không bịa. "
    "Model 'không khả dụng' thì nói rõ, không viện dẫn. "
    "'confidence' phản ánh sức mạnh lập luận sau lượt này.\n\n"
    "ĐẦU RA — DUY NHẤT một object JSON hợp lệ, không markdown, không văn bản nào khác:\n"
    "{\n"
    '  "verdict": "REAL" | "FAKE",\n'
    '  "confidence": 0.0,             // số thực 0.0-1.0\n'
    '  "argument": "phản bác lập luận mới của đối thủ + ít nhất một luận điểm mới; '
    'tự chứa, tối đa 250 từ",\n'
    '  "concession": "chuỗi nêu điểm nhượng bộ, hoặc null — BẮT BUỘC nếu đối thủ có luận điểm hợp lệ"\n'
    "}"
)

FAKE_ADVOCATE_PROMPT = (
    "Bạn là LUẬT SƯ PHẢN BIỆN trong phiên tranh biện đối kháng xác minh tin tức tiếng Việt. "
    "Vị trí bảo vệ: claim là FAKE. Đối thủ bảo vệ REAL. "
    "Mục tiêu: mỗi lượt phải đẩy tranh luận tiến lên, không dậm chân.\n\n"
    "ĐẦU VÀO (trong tin nhắn user):\n"
    "- CLAIM: nội dung cần xác minh.\n"
    "- TÍN HIỆU MODEL: PhoBERT chỉ là tham khảo YẾU (độ tin cậy thấp) — đừng dùng nó làm trụ cột; ưu tiên BẰNG CHỨNG và COOLANT. PhoBERT đo độ đúng của claim theo văn bản; COOLANT đo mức nhất quán "
    "giữa claim và ảnh ('FAKE' = ảnh và claim không khớp, chưa kết luận tin giả). "
    "Luận điểm tham chiếu, không phải kết luận sự thật.\n"
    "- VŨ KHÍ COOLANT (DÀNH RIÊNG CHO BẠN): hướng dẫn cách DÙNG COOLANT để tấn công "
    "tính xác thực của claim. Đọc kỹ phần này — đây là lợi thế của bạn.\n"
    "- TOÀN BỘ BẰNG CHỨNG: tách NGUỒN CHÍNH THỐNG và NGUỒN KHÔNG CHÍNH THỐNG. "
    "Nếu TẤT CẢ nguồn cùng nói một điều (một phía, một domain, trích dẫn trùng) → NGHI NGỜ, "
    "không coi số lượng là xác nhận.\n"
    "- LẬP LUẬN ĐỐI THỦ: trình bày có cấu trúc để bạn phản bác chính xác.\n"
    "- LỊCH SỬ TRANH LUẬN: tất cả các lượt trước.\n"
    "- CÁC LUẬN ĐIỂM ĐÃ DÙNG (KHÔNG ĐƯỢC LẶP LẠI): danh sách luận điểm đã xuất hiện.\n\n"
    "CÁCH TRANH LUẬN (QUY TẮC CỨNG — VI PHẠM SẼ BỊ TÍNH LÀ THUA LƯỢT):\n"
    "1. TRÍCH DẪN → PHẢN BÁC → LUẬN ĐIỂM MỚI: mỗi lượt bắt buộc theo cấu trúc 3 phần:\n"
    "   a) Trích dẫn CHÍNH XÁC 1 ý của đối thủ (1-2 câu, trong ngoặc kép)\n"
    "   b) Phản bác trực tiếp ý đó — chỉ ra mâu thuẫn logic, thiếu bằng chứng, hoặc hệ quả sai\n"
    "   c) Đưa ra ÍT NHẤT MỘT luận điểm MỚI chưa từng xuất hiện trong LỊCH SỬ TRANH LUẬN\n"
    "2. KHÔNG LẶP LẠI: kiểm tra 'CÁC LUẬN ĐIỂM ĐÃ DÙNG' — nếu ý của bạn đã có trong đó, "
    "PHẢI đưa ra góc nhìn khác. Lặp lại = thua lượt.\n"
    "3. TÔN TRỌNG ĐỐI THỦ: nếu đối thủ đưa ra luận điểm hợp lệ dựa trên bằng chứng, "
    "BẮT BUỘC ghi nhận trong 'concession'. Phủ nhận bằng chứng rõ ràng = mất credibility.\n"
    "4. Được phép đổi verdict sang REAL khi không còn phản bác được — ghi lý do trong "
    "'concession'. Mục tiêu là kết luận đúng, không phải thắng bằng mọi giá.\n\n"
    "Mỗi luận điểm phải dẫn BẰNG CHỨNG cụ thể (số thứ tự/nguồn/tier) — tranh luận là lớp kết luận cuối nên "
    "luận điểm không có bằng chứng sẽ bị giám khảo bỏ qua.\n"
    "QUY TẮC: chỉ dùng dữ liệu trong đầu vào, dùng đúng con số của TÍN HIỆU MODEL, không bịa. "
    "Model 'không khả dụng' thì nói rõ, không viện dẫn. "
    "'confidence' phản ánh sức mạnh lập luận sau lượt này.\n\n"
    "ĐẦU RA — DUY NHẤT một object JSON hợp lệ, không markdown, không văn bản nào khác:\n"
    "{\n"
    '  "verdict": "REAL" | "FAKE",\n'
    '  "confidence": 0.0,             // số thực 0.0-1.0\n'
    '  "argument": "phản bác lập luận mới của đối thủ + ít nhất một luận điểm mới; '
    'tự chứa, tối đa 250 từ",\n'
    '  "concession": "chuỗi nêu điểm nhượng bộ, hoặc null — BẮT BUỘC nếu đối thủ có luận điểm hợp lệ",\n'
    '  "pass": false                  // true nếu KHÔNG còn luận điểm mới đáng nói (khi đó để argument rỗng)\n'
    "}\n"
    "Bạn KHÔNG bắt buộc phải nói mỗi lượt: nếu không có ý mới, không cần phản bác thêm, hoặc đối thủ "
    "đã đúng, hãy đặt \"pass\": true thay vì nhắc lại. Không kéo dài tranh luận cho có."
)


# ── Formatting helpers ───────────────────────────────────────────────────────

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
            lines.append(
                f"- {r.get('model', 'unknown').upper()}: unavailable ({r.get('note', '')})"
            )
            continue
        model = r.get("model", "unknown").upper()
        label = r.get("label", "N/A")
        confidence = r.get("confidence", 0.0)
        probs = r.get("probabilities") or {}
        prob_str = ", ".join(
            f"{k}: {v:.1%}" for k, v in sorted(probs.items(), key=lambda x: -x[1])
        )
        lines.append(f"- {model}: {label} (confidence={confidence:.1%})")
        if prob_str:
            lines.append(f"  Probabilities: {prob_str}")
    return "\n".join(lines) if lines else "(no model predictions available)"


_MODEL_ROLES = {
    "phobert_vifactcheck": (
        "tín hiệu fact-check VĂN BẢN — đánh giá claim có đúng sự thật theo dữ liệu đã học"
    ),
    "coolant": (
        "đo mức độ NHẤT QUÁN giữa claim và hình ảnh — 'FAKE' nghĩa là claim và ảnh "
        "không khớp nhau, CHƯA phải kết luận tin giả"
    ),
}


def _format_model_results_verdict(results: List[dict]) -> str:
    """Format model results as signals/arguments, not truth verdicts."""
    if not results:
        return "(Không có tín hiệu model — tranh luận chỉ dựa trên bằng chứng)"
    available_models = [r for r in results if r.get("available")]
    if not available_models:
        lines = ["KHÔNG CÓ TÍN HIỆU MODEL KHẢ DỤNG — LUẬN CHỈ DỰA TRÊN BẰNG CHỨNG."]
    else:
        lines = [
            "TÍN HIỆU TỪ CÁC MODEL PHỤ TRỢ (đây là luận điểm/tín hiệu tham chiếu, "
            "KHÔNG phải kết luận sự thật — mỗi model đo một khía cạnh khác nhau):"
        ]
        for r in available_models:
            role = _MODEL_ROLES.get(r.get("model", ""), "tín hiệu tham chiếu")
            lines.append(f"  • {r.get('model', 'unknown').upper()}: {role}")
        if len(available_models) == 1:
            lines.append(
                "LƯU Ý: Chỉ có MỘT model khả dụng — kết quả nó là TÍN HIỆU YẾU, "
                "không phải bằng chứng quyết định. Bắt buộc đối chiếu bằng chứng web "
                "trước khi dựa vào model."
            )
    for r in results:
        model = r.get("model", "unknown").upper()
        if not r.get("available"):
            lines.append(f"- {model}: không khả dụng ({r.get('note', '')})")
            continue
        label = r.get("label", "N/A")
        confidence = r.get("confidence", 0.0)
        probs = r.get("probabilities") or {}
        prob_str = ", ".join(
            f"{k}: {v:.1%}" for k, v in sorted(probs.items(), key=lambda x: -x[1])
        )
        lines.append(
            f"- {model}: TÍN HIỆU '{label}' với confidence={confidence:.1%}. "
            f"Phân phối xác suất: {prob_str}"
        )
    lines.append("")
    return "\n".join(lines)


def _format_history(turns: List[dict]) -> str:
    """Format full debate history for context."""
    if not turns:
        return "(no prior turns)"
    lines = []
    for t in turns:
        agent = t.get("agent", "unknown")
        round_num = t.get("round", 0)
        verdict = t.get("verdict", "?")
        argument = t.get("argument", t.get("text", ""))[:700]
        concession = t.get("concession")
        line = f"[Round {round_num}] {agent}: verdict={verdict}, argument={argument}"
        if concession:
            line += f", concession={concession}"
        lines.append(line)
    return "\n".join(lines)


def _parse_advocate_json(content: str) -> Optional[dict]:
    """Parse advocate JSON response; return None on failure."""
    try:
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


# ── D-09: Role-specific COOLANT injection ────────────────────────────────────

def _get_coolant_result(model_results: List[dict]) -> Optional[dict]:
    """Extract COOLANT result from model_results if available."""
    for r in model_results:
        if r.get("model") == "coolant" and r.get("available"):
            return r
    return None


def _format_coolant_for_role(
    coolant: Optional[dict],
    role: str,
) -> str:
    """Generate a role-specific COOLANT weapon block for the advocate.

    Each advocate sees DIFFERENT guidance on how to use COOLANT — the real
    advocate wields consistency findings, the fake advocate exploits
    inconsistency / ambiguity signals.
    """
    if coolant is None:
        # D-09a: COOLANT unavailable — weaponize the absence
        if role == "real":
            return (
                "VŨ KHÍ COOLANT (DÀNH RIÊNG CHO BẠN — REAL ADVOCATE):\n"
                "COOLANT không khả dụng (không có ảnh để phân tích đa phương thức).\n"
                "→ Cách dùng: nhiều thông tin hợp lệ hoàn toàn là text-only (thông cáo "
                "báo chí, nghị định, số liệu thống kê). Việc COOLANT không áp dụng được "
                "KHÔNG làm yếu claim. Lập luận: 'COOLANT chỉ kiểm tra consistency ảnh-văn "
                "bản — claim này là text-only, không cần ảnh, nên việc thiếu COOLANT "
                "không phải điểm yếu.'\n"
                "→ Điểm yếu đối thủ có thể khai thác: họ sẽ nói 'thiếu bằng chứng trực "
                "quan'. Chuẩn bị sẵn phản bác: dẫn các nguồn text uy tín.\n"
            )
        else:
            return (
                "VŨ KHÍ COOLANT (DÀNH RIÊNG CHO BẠN — FAKE ADVOCATE):\n"
                "COOLANT không khả dụng (không có ảnh để phân tích đa phương thức).\n"
                "→ Cách dùng: việc thiếu ảnh/phương tiện là ĐIỂM YẾU của claim. Thông "
                "tin thật thường đi kèm hình ảnh, video, tài liệu gốc. Claim text-only "
                "không có bằng chứng trực quan = dễ bịa đặt. Lập luận: 'Claim này không "
                "có bằng chứng hình ảnh — một thông tin quan trọng thường được công bố "
                "kèm ảnh/video. Việc thiếu hoàn toàn yếu tố trực quan là dấu hiệu đáng ngờ.'\n"
                "→ Điểm yếu đối thủ có thể khai thác: họ sẽ nói 'nhiều thông tin hợp lệ "
                "là text-only'. Chuẩn bị sẵn phản bác: phân biệt giữa thông cáo báo chí "
                "(có template, logo) và tin đồn text-only.\n"
            )

    # D-09b: COOLANT available — role-specific weaponization
    label = str(coolant.get("label", "N/A")).upper()
    confidence = coolant.get("confidence", 0.0)
    probs = coolant.get("probabilities") or {}
    prob_str = ", ".join(
        f"{k}: {v:.1%}" for k, v in sorted(probs.items(), key=lambda x: -x[1])
    )

    # Internal signals if available
    sim_score = coolant.get("similarity_score")
    amb_weight = coolant.get("ambiguity_weight")
    attn = coolant.get("attention_weights") or {}
    attn_text = ""
    if attn and all(isinstance(attn.get(k), (int, float)) for k in ("text", "image", "cross")):
        attn_text = (
            f"- Trọng số attention: text={attn['text']:.2f}, "
            f"image={attn['image']:.2f}, cross={attn['cross']:.2f}\n"
        )

    header = f"VŨ KHÍ COOLANT (DÀNH RIÊNG CHO BẠN — {role.upper()} ADVOCATE):"
    status = f"- Kết quả: {label} (confidence={confidence:.1%})"
    dist = f"- Phân phối xác suất: {prob_str}"
    internals = ""
    if sim_score is not None:
        internals += f"- Độ tương đồng text-ảnh (similarity): {sim_score:.2f}\n"
    if amb_weight is not None:
        amb_label = "CAO — model không chắc chắn" if amb_weight > 0.6 else "THẤP — model tự tin"
        internals += f"- Độ mơ hồ cross-modal (ambiguity): {amb_weight:.2f} ({amb_label})\n"
    internals += attn_text
    # similarity_score separates matched/mismatched pairs better than the label
    # (matched pairs ~0.3-0.5, mismatched <~0.15); the label is biased to FAKE.
    if sim_score is not None:
        if sim_score >= 0.3 and label == "FAKE":
            internals += (
                f"- LƯU Ý: similarity cao ({sim_score:.2f}) nhưng nhãn FAKE — nhãn COOLANT "
                "có xu hướng nghiêng về FAKE, tin similarity hơn nhãn.\n"
            )
        elif sim_score < 0.15:
            internals += (
                f"- LƯU Ý: similarity thấp ({sim_score:.2f}) — ảnh và caption thực sự "
                "ít liên quan.\n"
            )

    if role == "real":
        # Real advocate: wield COOLANT when it says REAL, counter when it says FAKE
        if label == "REAL":
            guidance = (
                "→ CÁCH DÙNG: COOLANT xác nhận claim và ảnh NHẤT QUÁN — đây là vũ khí "
                "MẠNH. Dùng kết quả này làm trụ cột: 'Phân tích đa phương thức COOLANT "
                f"cho thấy text và ảnh khớp nhau với confidence {confidence:.1%} — "
                "điều này củng cố tính xác thực của claim ở cấp độ trực quan.' Kết hợp "
                "với PhoBERT để tạo thế 'cả hai model cùng xác nhận'.\n"
                "→ ĐIỂM YẾU CẦN PHÒNG THỦ: đối thủ sẽ nói COOLANT chỉ đo consistency, "
                "không đo sự thật. Phản bác: 'Consistency giữa text và ảnh là một trong "
                "những tín hiệu mạnh nhất để phát hiện tin giả — tin giả thường dùng ảnh "
                "không liên quan hoặc chỉnh sửa.'"
            )
        else:
            guidance = (
                "→ CÁCH DÙNG: COOLANT nói FAKE — đây là BẤT LỢI cho phe bạn. KHÔNG phủ "
                "nhận kết quả. Thay vào đó, GIỚI HẠN PHẠM VI: 'COOLANT chỉ đánh giá "
                "consistency text-ảnh, không đánh giá sự thật của claim. Một ảnh không "
                f"khớp không có nghĩa claim sai — có thể ảnh bị gán nhầm.' Chuyển "
                "hướng tranh luận sang BẰNG CHỨNG VĂN BẢN và PHOBERT.\n"
                "→ ĐIỂM YẾU ĐỐI THỦ: nếu ambiguity CAO, lập luận: 'COOLANT không tự "
                "tin — không thể dùng kết quả này làm bằng chứng quyết định.'"
            )
    else:
        # Fake advocate: wield COOLANT when it says FAKE, counter when it says REAL
        if label == "FAKE":
            guidance = (
                "→ CÁCH DÙNG: COOLANT phát hiện claim và ảnh KHÔNG NHẤT QUÁN — đây là "
                "vũ khí CHÍNH của bạn. Dùng kết quả này làm mũi tấn công: 'Phân tích "
                f"đa phương thức COOLANT cho thấy text và ảnh mâu thuẫn (confidence "
                f"{confidence:.1%}) — đây là dấu hiệu điển hình của tin giả, nơi ảnh "
                "bị ghép sai ngữ cảnh hoặc chỉnh sửa để đánh lừa.' Kết hợp với bằng "
                "chứng web để tạo thế gọng kìm.\n"
                "→ ĐIỂM YẾU CẦN PHÒNG THỦ: đối thủ sẽ nói COOLANT không đo sự thật. "
                "Phản bác: 'Tin giả thường ghép ảnh sai ngữ cảnh — mâu thuẫn ảnh-văn bản "
                "là tín hiệu cần giải thích, phe REAL phải chỉ ra vì sao ảnh không khớp.'"
            )
        else:
            guidance = (
                "→ CÁCH DÙNG: COOLANT nói REAL — đây là BẤT LỢI. KHÔNG phủ nhận. "
                "GIỚI HẠN PHẠM VI: 'COOLANT chỉ kiểm tra consistency text-ảnh — "
                "một ảnh khớp với text không chứng minh claim đúng. Tin giả tinh vi "
                "thường dùng ảnh thật nhưng ghép với context sai.' Chuyển hướng sang "
                f"PHOBERT và BẰNG CHỨNG WEB.\n"
                "→ KHAI THÁC AMBIGUITY: nếu ambiguity weight cao, lập luận: 'COOLANT "
                "có độ không chắc chắn cao — không thể coi đây là tín hiệu REAL đáng "
                "tin cậy.'"
            )

    return f"{header}\n{status}\n{dist}\n{internals}\n{guidance}\n"


# ── D-10: Anti-repetition ────────────────────────────────────────────────────

def _sentences(text: str) -> List[str]:
    import re

    parts = re.split(r"(?<=[.!?;])\s+|\n+", text or "")
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _extract_prior_claims(turns: List[dict]) -> List[str]:
    """List every sentence already argued (both sides), tagged by agent.

    The full argument is used, not just its first sentence, so the next
    advocate can see exactly which points are already spent.
    """
    claims = []
    for t in turns or []:
        arg = t.get("argument", "") or t.get("text", "")
        agent = t.get("agent", "unknown")
        for sent in _sentences(arg):
            claims.append(f"[{agent}] {sent[:160]}")
    return claims[-30:]


def _format_prior_claims(claims: List[str]) -> str:
    """Format prior claims as a forbidden-repetition block."""
    if not claims:
        return "(Chưa có luận điểm nào — đây là vòng đầu)"
    lines = ["CÁC LUẬN ĐIỂM ĐÃ DÙNG (KHÔNG ĐƯỢC LẶP LẠI):"]
    for i, c in enumerate(claims, 1):
        lines.append(f"  {i}. {c}")
    lines.append(
        "\nNếu luận điểm mới của bạn trùng với bất kỳ ý nào ở trên, "
        "PHẢI thay đổi góc nhìn hoặc đưa ra khía cạnh KHÁC của cùng vấn đề."
    )
    return "\n".join(lines)


# ── D-11: Structured opponent interaction ─────────────────────────────────────

def _format_opponent_structured(turns: List[dict]) -> str:
    """Format the last opponent argument in a structured format that enforces
    citation → rebuttal → new point flow.
    """
    if not turns:
        return "(đây là vòng đầu tiên, chưa có lập luận đối thủ)"

    last = turns[-1]
    agent = last.get("agent", "unknown")
    verdict = last.get("verdict", "?")
    confidence = last.get("confidence", 0.0)
    argument = last.get("argument", "") or last.get("text", "")
    concession = last.get("concession")

    lines = [
        "LẬP LUẬN ĐỐI THỦ (PHẢI PHẢN BÁC TRỰC TIẾP — CẤU TRÚC 3 BƯỚC):",
        f"  Đối thủ: {agent}",
        f"  Verdict: {verdict} (confidence={confidence:.2f})",
        f"  Nội dung: {argument}",
    ]
    if concession:
        lines.append(
            f"  Đối thủ đã NHƯỢNG BỘ: {concession}\n"
            f"  → KHÔNG được tấn công vào điểm đối thủ đã nhượng bộ — đó là dấu hiệu "
            f"thiếu tôn trọng."
        )
    lines.append(
        "\nYÊU CẦU PHẢN HỒI:\n"
        "  Bước 1 — TRÍCH DẪN: chọn 1 ý cụ thể của đối thủ, trích nguyên văn 1-2 câu.\n"
        "  Bước 2 — PHẢN BÁC: chỉ ra mâu thuẫn logic / thiếu bằng chứng / hệ quả sai.\n"
        "  Bước 3 — LUẬN ĐIỂM MỚI: đưa ra ít nhất 1 ý CHƯA TỪNG XUẤT HIỆN trong lịch sử."
    )
    return "\n".join(lines)


# ── D-12: Truth vs fake sources + unanimity doubt ────────────────────────────

def _domain(url: str) -> str:
    from urllib.parse import urlparse

    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _assess_source_unanimity(
    evidence_real: List[Evidence], evidence_fake: List[Evidence]
) -> Optional[str]:
    """Return a doubt warning when all sources tell one single story.

    Unanimity is suspicious, not reassuring: one-sided retrieval, a single
    origin domain, or copy-pasted snippets usually mean the "many sources"
    are one source repeated (or a coordinated spread).
    """
    real = evidence_real or []
    fake = evidence_fake or []
    allev = real + fake
    if not allev:
        return None
    reasons = []
    if len(allev) == 1:
        reasons.append("chỉ có DUY NHẤT một nguồn")
    elif not fake:
        reasons.append("TẤT CẢ nguồn đều là nguồn chính thống, không có nguồn đối lập nào")
    elif not real:
        reasons.append("TẤT CẢ nguồn đều là nguồn không chính thống/bị gắn cờ, không có nguồn chính thống")
    domains = {_domain(e.get("url", "")) for e in allev if e.get("url")}
    if len(allev) >= 2 and len(domains) == 1:
        reasons.append(f"mọi nguồn đều từ cùng một domain ({next(iter(domains))})")
    heads = [(e.get("snippet") or "").strip().lower()[:80] for e in allev]
    heads = [h for h in heads if h]
    if len(heads) >= 2 and len(set(heads)) < len(heads):
        reasons.append("nhiều nguồn có đoạn trích trùng nhau (sao chép lại cùng một tin)")
    if not reasons:
        return None
    return (
        "⚠ CẢNH BÁO NGUỒN ĐƠN CHIỀU — HÃY NGHI NGỜ:\n"
        + "\n".join(f"  - {r}" for r in reasons)
        + "\nKhi mọi nguồn cùng nói một điều, đó KHÔNG phải xác nhận độc lập: có thể là "
        "một nguồn gốc bị lặp lại, tin lan truyền có tổ chức, hoặc truy xuất bị lệch. "
        "Không được coi số lượng nguồn là sức mạnh; giảm confidence và nêu rõ sự nghi ngờ này."
    )


def _format_sources_split(
    evidence_real: List[Evidence], evidence_fake: List[Evidence]
) -> str:
    """Render truth sources and fake sources as separate blocks + doubt note."""
    out = (
        "NGUỒN CHÍNH THỐNG (TRUTH SOURCES):\n"
        f"{_format_evidence(evidence_real)}\n\n"
        "NGUỒN KHÔNG CHÍNH THỐNG / BỊ GẮN CỜ (FAKE SOURCES):\n"
        f"{_format_evidence(evidence_fake)}"
    )
    warn = _assess_source_unanimity(evidence_real, evidence_fake)
    return f"{out}\n\n{warn}" if warn else out


# ── D-13: Programmatic novelty guard ─────────────────────────────────────────

_NOVELTY_MAX_OVERLAP = 0.35
_NOVELTY_RETRIES = 2


def _trigrams(text: str) -> set:
    import re

    w = re.findall(r"\w+", (text or "").lower())
    return {tuple(w[i : i + 3]) for i in range(len(w) - 2)}


def _repeat_report(new_arg: str, turns: List[dict]) -> tuple:
    """Return (overlap_ratio, repeated_sentences) of ``new_arg`` vs prior turns.

    overlap = share of the new argument's word-trigrams already used in ANY
    earlier turn (own or opponent's). Repeated sentences are the new
    sentences that are mostly made of already-used trigrams.
    """
    prior = set()
    for t in turns or []:
        prior |= _trigrams(t.get("argument", "") or t.get("text", ""))
    new = _trigrams(new_arg)
    if not new or not prior:
        return 0.0, []
    overlap = len(new & prior) / len(new)
    repeated = []
    for sent in _sentences(new_arg):
        tri = _trigrams(sent)
        if tri and len(tri & prior) / len(tri) >= 0.5:
            repeated.append(sent)
    return overlap, repeated


def _invoke_advocate_with_novelty(
    llm, system_prompt: str, user_message: str, turns: List[dict]
):
    """Call the advocate LLM; re-ask when the answer repeats earlier turns.

    Returns (content, data_dict_or_None, attempts). If every retry still
    repeats, the least-repetitive answer is returned.
    """
    best = None  # (overlap, content, data)
    msg = user_message
    for attempt in range(_NOVELTY_RETRIES + 1):
        resp = llm.invoke([("system", system_prompt), ("user", msg)])
        content = str(getattr(resp, "content", ""))
        data = _parse_advocate_json(content)
        arg = (data or {}).get("argument", content)
        overlap, repeated = _repeat_report(arg, turns)
        if best is None or overlap < best[0]:
            best = (overlap, content, data)
        if overlap <= _NOVELTY_MAX_OVERLAP or not turns:
            return content, data, attempt + 1
        quoted = "\n".join(f"  - {r[:200]}" for r in repeated[:5]) or "  (nhiều ý trùng)"
        msg = (
            user_message
            + f"\n\n⛔ BẢN NHÁP TRƯỚC BỊ TỪ CHỐI VÌ LẶP LẠI ({overlap:.0%} nội dung đã được nói). "
            "Các câu lặp:\n"
            + quoted
            + "\nViết lại HOÀN TOÀN: trích một ý KHÁC của đối thủ, phản bác bằng lý lẽ/bằng chứng "
            "chưa dùng, và thêm luận điểm mới. Tuyệt đối không dùng lại các câu trên."
        )
    return best[1], best[2], _NOVELTY_RETRIES + 1


# ── Main message builder ──────────────────────────────────────────────────────

def _build_advocate_user_message(
    statement: str,
    model_results: List[dict],
    all_evidence: List[Evidence],
    turns: List[dict],
    role: Optional[str] = None,
    evidence_real: Optional[List[Evidence]] = None,
    evidence_fake: Optional[List[Evidence]] = None,
) -> str:
    """Assemble a role-specific single-turn user message for an advocate call.

    D-09: COOLANT weapon block is role-specific — real and fake advocates
          see DIFFERENT guidance on how to use COOLANT results.
    D-10: Prior claims block prevents argument repetition.
    D-11: Opponent argument is presented in structured format for interaction.
    """
    model_output_text = _format_model_results_verdict(model_results)
    if evidence_real is not None or evidence_fake is not None:
        all_evidence_text = _format_sources_split(evidence_real or [], evidence_fake or [])
    else:
        all_evidence_text = _format_evidence(all_evidence)
    history_text = _format_history(turns)

    # D-09: Role-specific COOLANT weapon block
    coolant_block = (
        _format_coolant_for_role(_get_coolant_result(model_results), role)
        if role in ("real", "fake")
        else ""
    )

    # D-11: Structured opponent for interaction enforcement
    opponent_block = _format_opponent_structured(turns)

    # D-10: Prior claims for anti-repetition
    prior_claims = _extract_prior_claims(turns)
    prior_block = _format_prior_claims(prior_claims)

    return (
        f"CLAIM:\n{statement}\n\n"
        f"MODEL PREDICTIONS (các model khả dụng):\n{model_output_text}\n\n"
        f"{coolant_block}\n"
        f"TOÀN BỘ BẰNG CHỨNG:\n{all_evidence_text}\n\n"
        f"{opponent_block}\n\n"
        f"LỊCH SỬ TRANH LUẬN:\n{history_text}\n\n"
        f"{prior_block}\n"
    )