"""Shared debate utilities for the real/fake advocate agents (D-08).

Extracted from the former ``debate_node.py``: advocate prompt templates,
evidence/model/history formatting helpers, JSON parsing, and the JSONL
turn logger. Both ``real_advocate`` and ``fake_advocate`` import from here.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..state import Evidence

REAL_ADVOCATE_PROMPT = (
    "Bạn là LUẬT SƯ BÀO CHỮA trong phiên tranh biện đối kháng xác minh tin tức tiếng Việt. "
    "Vị trí bảo vệ: claim là REAL. Đối thủ bảo vệ FAKE. "
    "Mục tiêu: mỗi lượt phải đẩy tranh luận tiến lên, không dậm chân.\n\n"
    "ĐẦU VÀO (trong tin nhắn user):\n"
    "- CLAIM: nội dung cần xác minh.\n"
    "- TÍN HIỆU MODEL: PhoBERT đo độ đúng của claim theo văn bản; COOLANT đo mức nhất quán "
    "giữa claim và ảnh ('FAKE' = ảnh và claim không khớp, chưa kết luận tin giả). "
    "Luận điểm tham chiếu, không phải kết luận sự thật.\n"
    "- TOÀN BỘ BẰNG CHỨNG: các nguồn kèm tier.\n"
    "- LẬP LUẬN ĐỐI THỦ: lượt phát biểu GẦN NHẤT của phe FAKE.\n"
    "- LỊCH SỬ TRANH LUẬN: tất cả các lượt trước.\n\n"
    "CÁCH TRANH LUẬN:\n"
    "- Trích ý cụ thể đối thủ vừa nêu, rồi phản bác bằng SUY LUẬN của bạn — chỉ ra mối liên hệ, "
    "hệ quả logic, hoặc mâu thuẫn nội tại của họ. Đừng đọc lại số liệu.\n"
    "- Mỗi lượt phải có ÍT NHẤT MỘT luận điểm mới hoặc góc phản bác mới; không lặp nguyên văn "
    "luận điểm đã nêu trong lịch sử. Chỉ nhắc số liệu model khi phục vụ lập luận mới.\n"
    "- Nếu đối thủ phản bác được điểm của bạn, thừa nhận trong 'concession' — đừng lặp điểm đã chết.\n"
    "- Được phép đổi verdict sang FAKE khi không còn phản bác được — ghi lý do trong 'concession'. "
    "Mục tiêu là kết luận đúng, không phải thắng bằng mọi giá.\n\n"
    "QUY TẮC: chỉ dùng dữ liệu trong đầu vào, dùng đúng con số của TÍN HIỆU MODEL, không bịa. "
    "Model 'không khả dụng' thì nói rõ, không viện dẫn. "
    "'confidence' phản ánh sức mạnh lập luận sau lượt này.\n\n"
    "ĐẦU RA — DUY NHẤT một object JSON hợp lệ, không markdown, không văn bản nào khác:\n"
    "{\n"
    '  "verdict": "REAL" | "FAKE",\n'
    '  "confidence": 0.0,             // số thực 0.0-1.0\n'
    '  "argument": "phản bác lập luận mới của đối thủ + ít nhất một luận điểm mới; '
    'tự chứa, tối đa 250 từ",\n'
    '  "concession": "chuỗi nêu điểm nhượng bộ, hoặc null"\n'
    "}"
)

FAKE_ADVOCATE_PROMPT = (
    "Bạn là LUẬT SƯ PHẢN BIỆN trong phiên tranh biện đối kháng xác minh tin tức tiếng Việt. "
    "Vị trí bảo vệ: claim là FAKE. Đối thủ bảo vệ REAL. "
    "Mục tiêu: mỗi lượt phải đẩy tranh luận tiến lên, không dậm chân.\n\n"
    "ĐẦU VÀO (trong tin nhắn user):\n"
    "- CLAIM: nội dung cần xác minh.\n"
    "- TÍN HIỆU MODEL: PhoBERT đo độ đúng của claim theo văn bản; COOLANT đo mức nhất quán "
    "giữa claim và ảnh ('FAKE' = ảnh và claim không khớp, chưa kết luận tin giả). "
    "Luận điểm tham chiếu, không phải kết luận sự thật.\n"
    "- TOÀN BỘ BẰNG CHỨNG: các nguồn kèm tier.\n"
    "- LẬP LUẬN ĐỐI THỦ: lượt phát biểu GẦN NHẤT của phe REAL.\n"
    "- LỊCH SỬ TRANH LUẬN: tất cả các lượt trước.\n\n"
    "CÁCH TRANH LUẬN:\n"
    "- Trích ý cụ thể đối thủ vừa nêu, rồi phản bác bằng SUY LUẬN của bạn — chỉ ra mối liên hệ, "
    "hệ quả logic, hoặc mâu thuẫn nội tại của họ. Đừng đọc lại số liệu.\n"
    "- Mỗi lượt phải có ÍT NHẤT MỘT luận điểm mới hoặc góc phản bác mới; không lặp nguyên văn "
    "luận điểm đã nêu trong lịch sử. Chỉ nhắc số liệu model khi phục vụ lập luận mới.\n"
    "- Nếu đối thủ phản bác được điểm của bạn, thừa nhận trong 'concession' — đừng lặp điểm đã chết.\n"
    "- Được phép đổi verdict sang REAL khi không còn phản bác được — ghi lý do trong 'concession'. "
    "Mục tiêu là kết luận đúng, không phải thắng bằng mọi giá.\n\n"
    "QUY TẮC: chỉ dùng dữ liệu trong đầu vào, dùng đúng con số của TÍN HIỆU MODEL, không bịa. "
    "Model 'không khả dụng' thì nói rõ, không viện dẫn. "
    "'confidence' phản ánh sức mạnh lập luận sau lượt này.\n\n"
    "ĐẦU RA — DUY NHẤT một object JSON hợp lệ, không markdown, không văn bản nào khác:\n"
    "{\n"
    '  "verdict": "REAL" | "FAKE",\n'
    '  "confidence": 0.0,             // số thực 0.0-1.0\n'
    '  "argument": "phản bác lập luận mới của đối thủ + ít nhất một luận điểm mới; '
    'tự chứa, tối đa 250 từ",\n'
    '  "concession": "chuỗi nêu điểm nhượng bộ, hoặc null"\n'
    "}"
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


def _build_advocate_user_message(
    statement: str,
    model_results: List[dict],
    all_evidence: List[Evidence],
    turns: List[dict],
) -> str:
    """Assemble the single-turn user message for an advocate call."""
    model_output_text = _format_model_results_verdict(model_results)
    all_evidence_text = _format_evidence(all_evidence)
    history_text = _format_history(turns)
    last_opponent_arg = (
        turns[-1].get("argument", "")
        if turns
        else "(đây là vòng đầu tiên, chưa có lập luận đối thủ)"
    )
    return (
        f"CLAIM:\n{statement}\n\n"
        f"MODEL PREDICTIONS (các model khả dụng):\n{model_output_text}\n\n"
        f"TOÀN BỘ BẰNG CHỨNG:\n{all_evidence_text}\n\n"
        f"LẬP LUẬN ĐỐI THỦ (phải phản bác trực tiếp):\n{last_opponent_arg}\n\n"
        f"LỊCH SỬ TRANH LUẬN:\n{history_text}\n"
    )
