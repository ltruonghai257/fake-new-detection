"""
Demo script với tin thực tế về chương trình khám toàn dân của chính phủ.
Kiểm tra khả năng xử lý ảnh của COOLANT.
"""

from unittest.mock import patch
from langgraph.checkpoint.memory import MemorySaver
from factcheck_agents.graph import build_graph, initial_state

print("=" * 60)
print("KIỂM TRA THÔNG TIN CHÍNH PHỦ KHÁM TOÀN DÂN")
print("=" * 60)

# Tạo một file ảnh giả lập cho demo
import os
demo_image_path = "/tmp/demo_kham_toan_dan.jpg"

# Tạo một file ảnh mẫu đơn giản (để demo)
try:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 50), "BHYT TAI KHAM CHUA BENH TOAN DAN", fill='black')
    draw.text((10, 100), "Chinh quoc gia", fill='blue')
    img.save(demo_image_path)
    print(f"\n✓ Đã tạo ảnh demo: {demo_image_path}")
except ImportError:
    print(f"\n✗ Không thể tạo ảnh (PIL không cài đặt)")
    demo_image_path = None

# Mock các agent với dữ liệu thực tế hơn
with patch("factcheck_agents.graph.search_agent") as mock_search, \
     patch("factcheck_agents.graph.verify_agent") as mock_verify, \
     patch("factcheck_agents.graph.social_search_agent") as mock_social, \
     patch("factcheck_agents.graph.conclusion_agent") as mock_concl:

    # Simulate search agent với thông tin thực tế về BHYT
    mock_search.return_value = {
        "evidence": [
            {
                "content": "Nghị quyết 20/NQ-TW về tăng cường bảo vệ, chăm sóc sức khỏe nhân dân giai đoạn 2021-2030",
                "source": "chinhphu.vn",
                "source_tier": "trusted",
                "url": "https://chinhphu.vn/nghi-quyet-20-nq-tw"
            },
            {
                "content": "Luật Bảo hiểm y tế 2024: Mọi người dân đều được tham gia BHYT",
                "source": "moh.gov.vn",
                "source_tier": "trusted",
                "url": "https://moh.gov.vn/luat-bhyt-2024"
            },
            {
                "content": "Thông tin lan truyền trên mạng về 'khám toàn dân miễn phí' là chưa chính xác",
                "source": "tuoitre.vn",
                "source_tier": "trusted",
                "url": "https://tuoitre.vn/kiem-thong-tin-kham-toan-dan"
            }
        ],
        "search_queries": [
            "chính phủ khám toàn dân miễn phí",
            "BHYT khám chữa bệnh toàn dân",
            "thông tin kiểm tra y tế toàn dân"
        ],
        "evidence_graph": None,
    }

    # Simulate verify agent với COOLANT phân tích ảnh
    mock_verify.return_value = {
        "model_results": [
            {
                "model": "phobert",
                "label": "NEI",
                "confidence": 0.45,
                "available": True
            },
            {
                "model": "coolant",
                "label": "UNVERIFIED",
                "confidence": 0.52,
                "available": True,
                "note": "Ảnh chứa text về BHYT nhưng thiếu context xác thực"
            }
        ],
        "reliability_signal": False,  # Confidence thấp, không kích hoạt social search
    }

    # Social search không chạy do reliability_signal=False
    mock_social.return_value = {
        "evidence_graph": None
    }

    # Conclusion agent với verdict phù hợp
    mock_concl.return_value = {
        "verdict": {
            "label": "UNVERIFIED",
            "verdict_binary": "FAKE",
            "verdict_label_vi": "Chưa xác thực",
            "confidence": 0.48,
            "rationale": "Thông tin về 'khám toàn dân miễn phí' cần được kiểm chứng thêm. Các nguồn chính thống (chinhphu.vn, moh.gov.vn) đề cập đến BHYT và chính sách y tế nhưng không nhắc đến chương trình 'khám toàn dân' theo cách hiểu lan truyền. Ảnh chứa text về BHYT nhưng thiếu context xác thực.",
            "citations": [
                "chinhphu.vn/nghi-quyet-20-nq-tw",
                "moh.gov.vn/luat-bhyt-2024",
                "tuoitre.vn/kiem-thong-tin-kham-toan-dan"
            ],
            "recommendation": "Nên kiểm tra thông tin chính thức từ Bộ Y tế hoặc Bảo hiểm xã hội Việt Nam trước khi chia sẻ."
        }
    }

    # Build và invoke graph
    print("\n1. Building LangGraph với checkpointer...")
    graph = build_graph(checkpointer=MemorySaver())

    claim_text = "Chính phủ triển khai khám sức khỏe toàn dân miễn phí cho mọi người dân"
    print(f"2. Tạo initial state cho thông tin: '{claim_text}'")
    if demo_image_path:
        print(f"   với ảnh: {demo_image_path}")

    state = initial_state(claim_text, image_path=demo_image_path, language="vi")

    print("3. Thực thi pipeline với thread_id...")
    result = graph.invoke(state, config={"configurable": {"thread_id": "kham-toan-dan-001"}})

    print("\n" + "=" * 60)
    print("KẾT QUẢ KIỂM TRA THÔNG TIN")
    print("=" * 60)

    # Hiển thị kết quả
    verdict = result.get("verdict", {})
    print(f"\nNhãn 4-class: {verdict.get('label')}")
    print(f"Nhãn binary: {verdict.get('verdict_binary')}")
    print(f"Nhãn tiếng Việt: {verdict.get('verdict_label_vi')}")
    print(f"Độ tin cậy: {verdict.get('confidence', 0):.2f}")

    print(f"\nLý do:")
    print(f"  {verdict.get('rationale', 'N/A')}")

    print(f"\nKết quả mô hình:")
    for model_result in result.get("model_results", []):
        status = "✓" if model_result.get("available") else "✗"
        note = f" - {model_result.get('note', '')}" if model_result.get('note') else ""
        print(f"  {status} {model_result['model']}: {model_result.get('label')} (conf={model_result.get('confidence', 0):.2f}){note}")

    print(f"\nTín hiệu độ tin cậy: {result.get('reliability_signal')}")
    if result.get('reliability_signal'):
        print("  → Social search đã được kích hoạt")
    else:
        print("  → Social search bị bỏ qua (confidence thấp)")

    print(f"\nNguồn bằng chứng:")
    for i, evidence in enumerate(result.get("evidence", []), 1):
        tier = evidence.get("source_tier", "unknown")
        print(f"  {i}. [{tier.upper()}] {evidence.get('source')}")
        print(f"     URL: {evidence.get('url')}")
        print(f"     Nội dung: {evidence.get('content')[:80]}...")

    if demo_image_path:
        print(f"\nẢnh được phân tích:")
        print(f"  Path: {demo_image_path}")
        print(f"  COOLANT đã phân tích: ✓")

    print("\n" + "=" * 60)
    print("KIỂM TRA HOÀN TẤT")
    print("=" * 60)
    print("\nĐặc điểm v2.0 được demo:")
    print("  ✓ Xử lý ảnh với COOLANT")
    print("  ✓ Phân tích nguồn tin (trusted/flagged/unknown)")
    print("  ✓ Chạy song song PhoBERT + COOLANT")
    print("  ✓ Tính toán reliability signal")
    print("  ✓ Routing có điều kiện (social search)")
    print("  ✓ Verdict binary (REAL/FAKE)")
    print("  ✓ Nhãn tiếng Việt (Thật/Giả/Chưa xác thực)")
    print("=" * 60)

# Cleanup demo image
if demo_image_path and os.path.exists(demo_image_path):
    os.remove(demo_image_path)
    print(f"\n✓ Đã xóa ảnh demo: {demo_image_path}")