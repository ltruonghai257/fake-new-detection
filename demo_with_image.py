"""
Demo thực tế với ảnh để test COOLANT
"""

from factcheck_agents.graph import build_graph, initial_state
from factcheck_agents.config import settings
from PIL import Image, ImageDraw, ImageFont

print("=" * 60)
print("DEMO THỰC TẾ VỚI ẢNH (COOLANT)")
print("=" * 60)

# Kiểm tra API keys
print("\nKiểm tra cấu hình:")
print(f"  OPENAI_API_KEY: {'✓ Đã cài đặt' if settings.has_llm() else '✗ Chưa cài đặt'}")
print(f"  TAVILY_API_KEY: {'✓ Đã cài đặt' if settings.has_search() else '✗ Chưa cài đặt'}")
print(f"  COOLANT_CKPT_PATH: {'✓ Đã cài đặt' if settings.coolant_ckpt_path else '✗ Chưa cài đặt'}")

# Tạo ảnh demo với text về tin giả
demo_image_path = "/tmp/demo_fake_news_image.jpg"
try:
    img = Image.new('RGB', (600, 300), color='white')
    draw = ImageDraw.Draw(img)

    # Thêm text giả lập tin giả
    draw.text((20, 50), "CHÍNH PHỦ VIỆT NAM", fill='blue', font=None)
    draw.text((20, 100), "TẶNG NGƯỜI DÂN 10 TRIỆU ĐỒNG", fill='red', font=None)
    draw.text((20, 150), "MỖI NGƯỜI ĐĂNG KÝ NGAY!", fill='black', font=None)
    draw.text((20, 200), "www.chinhphu-vietnam-gift.com", fill='gray', font=None)

    img.save(demo_image_path)
    print(f"\n✓ Đã tạo ảnh demo tin giả: {demo_image_path}")
except Exception as e:
    print(f"\n✗ Không thể tạo ảnh demo: {e}")
    demo_image_path = None

# Thông tin kiểm tra với ảnh
claim_text = "Chính phủ Việt Nam tặng mỗi người dân 10 triệu đồng khi đăng ký"
print(f"\nThông tin kiểm tra: '{claim_text}'")
if demo_image_path:
    print(f"Ảnh đi kèm: {demo_image_path}")

if settings.has_llm() and settings.has_search() and demo_image_path:
    print("\n" + "=" * 60)
    print("CHẠY THỰC TẾ VỚI WEB SEARCH + LLM + COOLANT")
    print("=" * 60)
    print("⏳ Đang chạy... (sẽ mất 1-3 phút tùy API)")

    try:
        # Build graph custom KHÔNG dùng checkpointer
        from langgraph.graph import END, START, StateGraph
        from factcheck_agents.agents import (
            conclusion_agent,
            search_agent,
            social_search_agent,
            verify_agent,
        )
        from factcheck_agents.state import FactCheckState

        def route_after_verify(state: FactCheckState) -> str:
            """Route after verify based on reliability_signal."""
            if state.get("reliability_signal"):
                return "social_search"
            return "conclusion"

        # Build graph without checkpointer
        g = StateGraph(FactCheckState)
        g.add_node("search", search_agent)
        g.add_node("verify", verify_agent)
        g.add_node("social_search", social_search_agent)
        g.add_node("conclusion", conclusion_agent)

        g.add_edge(START, "search")
        g.add_edge("search", "verify")
        g.add_conditional_edges(
            "verify",
            route_after_verify,
            {"social_search": "social_search", "conclusion": "conclusion"},
        )
        g.add_edge("social_search", "conclusion")
        g.add_edge("conclusion", END)

        graph = g.compile()  # No checkpointer

        state = initial_state(claim_text, image_path=demo_image_path, language="vi")

        # Chạy KHÔNG với config thread_id để tránh checkpointer
        result = graph.invoke(state)

        print("\n" + "=" * 60)
        print("KẾT QUẢ THỰC TẾ VỚI ẢNH")
        print("=" * 60)

        verdict = result.get("verdict", {})
        print(f"\nNhãn 4-class: {verdict.get('label')}")
        print(f"Nhãn binary: {verdict.get('verdict_binary')}")
        print(f"Nhãn tiếng Việt: {verdict.get('verdict_label_vi')}")
        print(f"Độ tin cậy: {verdict.get('confidence', 0):.2f}")

        print(f"\nLý do:")
        rationale = verdict.get('rationale', 'N/A')
        if len(rationale) > 200:
            rationale = rationale[:200] + "..."
        print(f"  {rationale}")

        print(f"\nKết quả mô hình:")
        for model_result in result.get("model_results", []):
            status = "✓" if model_result.get("available") else "✗"
            note = f" - {model_result.get('note', '')}" if model_result.get('note') else ""
            print(f"  {status} {model_result['model']}: {model_result.get('label')} (conf={model_result.get('confidence', 0):.2f}){note}")

        print(f"\nSố lượng bằng chứng: {len(result.get('evidence', []))}")
        for i, evidence in enumerate(result.get("evidence", [])[:3], 1):  # Hiển thị tối đa 3
            tier = evidence.get("source_tier", "unknown")
            print(f"  {i}. [{tier.upper()}] {evidence.get('source')}")
            print(f"     URL: {evidence.get('url')}")

        print(f"\nReliability signal: {result.get('reliability_signal')}")
        print(f"   → Social search {'được kích hoạt' if result.get('reliability_signal') else 'bị bỏ qua'}")

        print("\n" + "=" * 60)
        print("✅ DEMO THỰC TẾ VỚI ẢNH THÀNH CÔNG!")
        print("=" * 60)
        print("Đây là kết quả thực tế từ:")
        print("  - Web search Tavily API")
        print("  - OpenAI GPT cho LLM")
        print("  - Local PhoBERT model")
        print("  - Local COOLANT model (với ảnh)")

    except Exception as e:
        print(f"\n❌ Lỗi khi chạy thực tế: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n❌ Thiếu API key hoặc ảnh để chạy thực tế")

# Cleanup demo image
if demo_image_path and demo_image_path.startswith("/tmp/"):
    import os
    if os.path.exists(demo_image_path):
        os.remove(demo_image_path)
        print(f"\n✓ Đã xóa ảnh demo: {demo_image_path}")