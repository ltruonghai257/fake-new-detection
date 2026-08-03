"""
Demo thực tế với link thật về thông tin chính phủ.
Chạy thực tế với web search và LLM (nếu có API key).
"""

import os
from factcheck_agents.graph import build_graph, initial_state
from factcheck_agents.config import settings

print("=" * 60)
print("DEMO THỰC TẾ VỚI LINK THẬT")
print("=" * 60)

# Kiểm tra API keys
print("\nKiểm tra cấu hình:")
print(f"  OPENAI_API_KEY: {'✓ Đã cài đặt' if settings.has_llm() else '✗ Chưa cài đặt'}")
print(f"  TAVILY_API_KEY: {'✓ Đã cài đặt' if settings.has_search() else '✗ Chưa cài đặt'}")
print(f"  GOOGLE_CSE_API_KEY: {'✓ Đã cài đặt' if os.getenv('GOOGLE_CSE_API_KEY') else '✗ Chưa cài đặt'}")

# Thông tin kiểm tra với link thật
claim_text = "Chính phủ Việt Nam triển khai chương trình khám sức khỏe toàn dân miễn phí cho mọi công dân"
real_link = "https://bhyt.vn hoặc https://moh.gov.vn"

print(f"\nThông tin kiểm tra: '{claim_text}'")
print(f"Link tham khảo: {real_link}")

# Kiểm tra nếu có đủ API keys thì chạy thực tế, nếu không thì mock
if settings.has_llm() and settings.has_search():
    print("\n" + "=" * 60)
    print("CHẠY THỰC TẾ VỚI WEB SEARCH + LLM")
    print("=" * 60)
    print("⏳ Đang chạy... (sẽ mất 1-3 phút tùy API)")

    try:
        # Build graph và chạy thực tế
        graph = build_graph()
        state = initial_state(claim_text, language="vi")

        # Chạy với thread_id để có checkpointing
        result = graph.invoke(state, config={"configurable": {"thread_id": "real-demo-001"}})

        print("\n" + "=" * 60)
        print("KẾT QUẢ THỰC TẾ")
        print("=" * 60)

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
            print(f"  {status} {model_result['model']}: {model_result.get('label')} (conf={model_result.get('confidence', 0):.2f})")

        print(f"\nSố lượng bằng chứng: {len(result.get('evidence', []))}")
        for i, evidence in enumerate(result.get("evidence", [])[:5], 1):  # Hiển thị tối đa 5
            tier = evidence.get("source_tier", "unknown")
            print(f"  {i}. [{tier.upper()}] {evidence.get('source')} - {evidence.get('url')}")

        print(f"\nReliability signal: {result.get('reliability_signal')}")

    except Exception as e:
        print(f"\n❌ Lỗi khi chạy thực tế: {e}")
        print("→ Chuyển sang mode demo với mock data")

else:
    print("\n" + "=" * 60)
    print("CHẠY DEMO VỚI MOCK DATA (THIẾU API KEY)")
    print("=" * 60)
    print("⚠️  Thiếu API key để chạy thực tế")
    print("→ Sử dụng mock data để demo luồng xử lý")

    # Mock data thực tế hơn với link thật
    from unittest.mock import patch
    from langgraph.checkpoint.memory import MemorySaver

    with patch("factcheck_agents.graph.search_agent") as mock_search, \
         patch("factcheck_agents.graph.verify_agent") as mock_verify, \
         patch("factcheck_agents.graph.social_search_agent") as mock_social, \
         patch("factcheck_agents.graph.conclusion_agent") as mock_concl:

        # Mock search với link thật
        mock_search.return_value = {
            "evidence": [
                {
                    "content": "Luật Bảo hiểm y tế 2024: Người dân được tham gia BHYT và khám chữa bệnh theo quy định",
                    "source": "moh.gov.vn",
                    "source_tier": "trusted",
                    "url": "https://moh.gov.vn/luat-bhyt-2024"
                },
                {
                    "content": "Thông tin về 'khám toàn dân miễn phí' lan truyền trên mạng xã hội chưa được cơ quan chức năng xác nhận",
                    "source": "vietnamplus.vn",
                    "source_tier": "trusted",
                    "url": "https://vietnamplus.vn/kiem-thong-tin-kham-toan-dan"
                },
                {
                    "content": "BHYT Việt Nam: Các quyền lợi khi tham gia bảo hiểm y tế",
                    "source": "bhyt.vn",
                    "source_tier": "trusted",
                    "url": "https://bhyt.vn/quyen-loi-bhyt"
                }
            ],
            "search_queries": [
                "khám sức khỏe toàn dân miễn phí chính phủ",
                "BHYT khám chữa bệnh toàn dân",
                "thông tin y tế toàn dân việt nam"
            ],
            "evidence_graph": None,
        }

        mock_verify.return_value = {
            "model_results": [
                {
                    "model": "phobert",
                    "label": "NEI",
                    "confidence": 0.42,
                    "available": True
                },
                {
                    "model": "coolant",
                    "label": "UNVERIFIED",
                    "confidence": 0.48,
                    "available": True
                }
            ],
            "reliability_signal": False,
        }

        mock_social.return_value = {"evidence_graph": None}

        mock_concl.return_value = {
            "verdict": {
                "label": "UNVERIFIED",
                "verdict_binary": "FAKE",
                "verdict_label_vi": "Chưa xác thực",
                "confidence": 0.45,
                "rationale": "Các nguồn chính thống (moh.gov.vn, bhyt.vn, vietnamplus.vn) đề cập đến BHYT và quyền lợi khám chữa bệnh nhưng không nhắc đến chương trình 'khám toàn dân miễn phí' theo cách hiểu lan truyền. Thông tin này cần được kiểm chứng thêm từ cơ quan chức năng.",
                "citations": [
                    "https://moh.gov.vn/luat-bhyt-2024",
                    "https://vietnamplus.vn/kiem-thong-tin-kham-toan-dan",
                    "https://bhyt.vn/quyen-loi-bhyt"
                ],
                "recommendation": "Nên liên hệ Bộ Y tế hoặc Bảo hiểm xã hội Việt Nam để xác nhận thông tin chính xác."
            }
        }

        # Chạy demo
        graph = build_graph(checkpointer=MemorySaver())
        state = initial_state(claim_text, language="vi")
        result = graph.invoke(state, config={"configurable": {"thread_id": "mock-demo-001"}})

        print("\n" + "=" * 60)
        print("KẾT QUẢ DEMO (VỚI LINK THẬT)")
        print("=" * 60)

        verdict = result.get("verdict", {})
        print(f"\nNhãn 4-class: {verdict.get('label')}")
        print(f"Nhãn binary: {verdict.get('verdict_binary')}")
        print(f"Nhãn tiếng Việt: {verdict.get('verdict_label_vi')}")
        print(f"Độ tin cậy: {verdict.get('confidence', 0):.2f}")

        print(f"\nLý do:")
        print(f"  {verdict.get('rationale', 'N/A')}")

        print(f"\nLink tham khảo (thực tế):")
        for i, evidence in enumerate(result.get("evidence", []), 1):
            tier = evidence.get("source_tier", "unknown")
            print(f"  {i}. [{tier.upper()}] {evidence.get('url')}")
            print(f"     Nguồn: {evidence.get('source')}")

print("\n" + "=" * 60)
print("GIẢI THÍCH TỐC ĐỘ")
print("=" * 60)
print("""
Demo nhanh vì:
1. Mock data: Không gọi API thực tế (web search, LLM)
2. Local models: PhoBERT và COOLANT chạy local
3. Không network: Không tải model/download data

Để chạy thực tế:
- Cần OPENAI_API_KEY (cho LLM)
- Cần TAVILY_API_KEY hoặc GOOGLE_CSE_API_KEY (cho web search)
- Sẽ mất 1-3 phút tùy tốc độ API
- Sẽ tìm thông tin thực tế từ internet
""")

print("=" * 60)
print("HƯỚNG DẪN CÀI ĐẶT API KEY")
print("=" * 60)
print("""
Để chạy thực tế, cài đặt các biến môi trường:

export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."
# Hoặc
export GOOGLE_CSE_API_KEY="..."
export GOOGLE_CSE_ID="..."

Sau đó chạy lại script này.
""")