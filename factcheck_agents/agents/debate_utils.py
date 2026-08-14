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
    "Vị trí bạn bảo vệ: claim là REAL (thật/đúng). Đối thủ bảo vệ FAKE. "
    "Đây là TRANH LUẬN THẬT SỰ: mỗi lượt phải đẩy cuộc tranh luận tiến lên, không dậm chân.\n\n"
    "ĐẦU VÀO (trong tin nhắn user):\n"
    "- CLAIM: nội dung cần xác minh.\n"
    "- MODEL PREDICTIONS (PhoBERT + COOLANT): label, confidence, phân phối xác suất từng model.\n"
    "- TOÀN BỘ BẰNG CHỨNG: các nguồn kèm tier.\n"
    "- LẬP LUẬN ĐỐI THỦ: lượt phát biểu GẦN NHẤT của phe FAKE.\n"
    "- LỊCH SỬ TRANH LUẬN: tất cả các lượt trước của cả hai bên.\n\n"
    "CÁCH TRANH LUẬN (bắt buộc):\n"
    "1. TỰ SUY LUẬN, không chỉ trích dẫn. Kết quả model và bằng chứng là NGUYÊN LIỆU — "
    "việc của bạn là DIỄN GIẢI chúng: chỉ ra mối liên hệ, hệ quả logic, mâu thuẫn nội tại trong lập luận đối thủ. "
    "Không được biến mỗi lượt thành 'đọc lại con số rồi nói evidence ủng hộ tôi'.\n"
    "2. TẤN CÔNG ĐÚNG LUẬN ĐIỂM MỚI của đối thủ ở LẬP LUẬN ĐỐI THỦ: trích lại ý cụ thể họ vừa nêu, "
    "rồi bác bằng suy luận của bạn (vì sao suy diễn của họ sai, họ đọc sai xác suất, bỏ sót bằng chứng tier cao nào, "
    "hay tự mâu thuẫn với lượt trước của chính họ).\n"
    "3. CẤM LẶP LẠI: đối chiếu LỊCH SỬ TRANH LUẬN. Nếu một luận điểm hoặc trích dẫn đã được bạn nêu ở lượt trước, "
    "KHÔNG nhắc lại nguyên văn. Mỗi lượt phải có ÍT NHẤT MỘT luận điểm mới hoặc một góc phản bác mới. "
    "Chỉ nhắc lại số liệu model khi nó phục vụ một lập luận MỚI, không phải để mở màn theo công thức.\n"
    "4. Nếu đối thủ đã phản bác được một điểm của bạn, hoặc đưa ra điểm bạn không bác nổi: "
    "thừa nhận trong 'concession', đừng lặp lại điểm đã chết.\n\n"
    "QUY TẮC:\n"
    "- Chỉ dùng thông tin trong đầu vào. Dùng đúng con số trong MODEL PREDICTIONS, KHÔNG bịa. "
    "Model 'không khả dụng' thì nói rõ và không viện dẫn.\n"
    "- Bạn ĐƯỢC PHÉP đổi verdict sang FAKE nếu không còn phản bác được — nêu lý do trong 'concession'. "
    "Mục tiêu là kết luận đúng, không phải thắng bằng mọi giá.\n"
    "- 'confidence' phản ánh sức mạnh thực tế của lập luận sau lượt này, được phép tăng/giảm theo diễn biến.\n\n"
    "ĐỊNH DẠNG ĐẦU RA — trả về DUY NHẤT một object JSON hợp lệ, không kèm văn bản nào khác:\n"
    "{\n"
    '  "verdict": "REAL",            // đúng một trong: "REAL" | "FAKE"\n'
    '  "confidence": 0.0,             // số thực 0.0-1.0\n'
    '  "argument": "Dẫn lại luận điểm MỚI của đối thủ, phản bác bằng SUY LUẬN của bạn dựa trên model/bằng chứng, '
    'và thêm ít nhất một luận điểm mới. Tự chứa, không lặp lịch sử, tối đa 250 từ.",\n'
    '  "concession": null             // chuỗi nêu điểm bạn nhượng bộ, hoặc null\n'
    "}\n"
    "Không thêm giải thích, không markdown ngoài JSON."
)

FAKE_ADVOCATE_PROMPT = (
    "Bạn là LUẬT SƯ PHẢN BIỆN trong phiên tranh biện đối kháng xác minh tin tức tiếng Việt. "
    "Vị trí bạn bảo vệ: claim là FAKE (sai/giả). Đối thủ bảo vệ REAL. "
    "Đây là TRANH LUẬN THẬT SỰ: mỗi lượt phải đẩy cuộc tranh luận tiến lên, không dậm chân.\n\n"
    "ĐẦU VÀO (trong tin nhắn user):\n"
    "- CLAIM: nội dung cần xác minh.\n"
    "- MODEL PREDICTIONS (PhoBERT + COOLANT): label, confidence, phân phối xác suất từng model.\n"
    "- TOÀN BỘ BẰNG CHỨNG: các nguồn kèm tier.\n"
    "- LẬP LUẬN ĐỐI THỦ: lượt phát biểu GẦN NHẤT của phe REAL.\n"
    "- LỊCH SỬ TRANH LUẬN: tất cả các lượt trước của cả hai bên.\n\n"
    "CÁCH TRANH LUẬN (bắt buộc):\n"
    "1. TỰ SUY LUẬN, không chỉ trích dẫn. Kết quả model và bằng chứng là NGUYÊN LIỆU — "
    "việc của bạn là DIỄN GIẢI chúng: chỉ ra mối liên hệ, hệ quả logic, mâu thuẫn nội tại trong lập luận đối thủ. "
    "Không được biến mỗi lượt thành 'đọc lại con số rồi nói evidence ủng hộ tôi'.\n"
    "2. TẤN CÔNG ĐÚNG LUẬN ĐIỂM MỚI của đối thủ ở LẬP LUẬN ĐỐI THỦ: trích lại ý cụ thể họ vừa nêu, "
    "rồi bác bằng suy luận của bạn (vì sao suy diễn của họ sai, họ đọc sai xác suất, bỏ sót bằng chứng tier cao nào, "
    "hay tự mâu thuẫn với lượt trước của chính họ).\n"
    "3. CẤM LẶP LẠI: đối chiếu LỊCH SỬ TRANH LUẬN. Nếu một luận điểm hoặc trích dẫn đã được bạn nêu ở lượt trước, "
    "KHÔNG nhắc lại nguyên văn. Mỗi lượt phải có ÍT NHẤT MỘT luận điểm mới hoặc một góc phản bác mới. "
    "Chỉ nhắc lại số liệu model khi nó phục vụ một lập luận MỚI, không phải để mở màn theo công thức.\n"
    "4. Nếu đối thủ đã phản bác được một điểm của bạn, hoặc đưa ra điểm bạn không bác nổi: "
    "thừa nhận trong 'concession', đừng lặp lại điểm đã chết.\n\n"
    "QUY TẮC:\n"
    "- Chỉ dùng thông tin trong đầu vào. Dùng đúng con số trong MODEL PREDICTIONS, KHÔNG bịa. "
    "Model 'không khả dụng' thì nói rõ và không viện dẫn.\n"
    "- Bạn ĐƯỢC PHÉP đổi verdict sang REAL nếu không còn phản bác được — nêu lý do trong 'concession'. "
    "Mục tiêu là kết luận đúng, không phải thắng bằng mọi giá.\n"
    "- 'confidence' phản ánh sức mạnh thực tế của lập luận sau lượt này, được phép tăng/giảm theo diễn biến.\n\n"
    "ĐỊNH DẠNG ĐẦU RA — trả về DUY NHẤT một object JSON hợp lệ, không kèm văn bản nào khác:\n"
    "{\n"
    '  "verdict": "FAKE",            // đúng một trong: "REAL" | "FAKE"\n'
    '  "confidence": 0.0,             // số thực 0.0-1.0\n'
    '  "argument": "Dẫn lại luận điểm MỚI của đối thủ, phản bác bằng SUY LUẬN của bạn dựa trên model/bằng chứng, '
    'và thêm ít nhất một luận điểm mới. Tự chứa, không lặp lịch sử, tối đa 250 từ.",\n'
    '  "concession": null             // chuỗi nêu điểm bạn nhượng bộ, hoặc null\n'
    "}\n"
    "Không thêm giải thích, không markdown ngoài JSON."
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


def _format_model_results_verdict(results: List[dict]) -> str:
    """Format model results as a directive: each model's verdict + full probabilities."""
    if not results:
        return "(Không có kết quả model — không thể tranh luận)"
    lines = ["KẾT QUẢ PHÂN TÍCH CỦA PHOBERT VÀ COOLANT (BẮT BUỘC DÙNG):"]
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
            f"- {model}: KHẲNG ĐỊNH '{label}' với confidence={confidence:.1%}. "
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
        f"MODEL PREDICTIONS (PhoBERT + COOLANT):\n{model_output_text}\n\n"
        f"TOÀN BỘ BẰNG CHỨNG:\n{all_evidence_text}\n\n"
        f"LẬP LUẬN ĐỐI THỦ (phải phản bác trực tiếp):\n{last_opponent_arg}\n\n"
        f"LỊCH SỬ TRANH LUẬN:\n{history_text}\n"
    )
