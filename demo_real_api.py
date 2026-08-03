"""
Demo thực tế với API thật nhưng không dùng checkpointer để tránh serialization error.
"""

from factcheck_agents.graph import build_graph, initial_state
from factcheck_agents.config import settings

print("=" * 60)
print("DEMO THỰC TẾ VỚI API THẬT (KHÔNG DÙNG CHECKPOINTER)")
print("=" * 60)

# Kiểm tra API keys
print("\nKiểm tra cấu hình:")
print(f"  OPENAI_API_KEY: {'✓ Đã cài đặt' if settings.has_llm() else '✗ Chưa cài đặt'}")
print(
    f"  TAVILY_API_KEY: {'✓ Đã cài đặt' if settings.has_search() else '✗ Chưa cài đặt'}"
)

# Thông tin kiểm tra với link thật
claim_text = "Chính phủ Việt Nam triển khai chương trình khám sức khỏe toàn dân miễn phí cho mọi công dân"
real_link = "https://bhyt.vn hoặc https://moh.gov.vn"

print(f"\nThông tin kiểm tra: '{claim_text}'")
print(f"Link tham khảo: {real_link}")

if settings.has_llm() and settings.has_search():
    print("\n" + "=" * 60)
    print("CHẠY THỰC TẾ VỚI WEB SEARCH + LLM")
    print("=" * 60)
    print("⏳ Đang chạy... (sẽ mất 1-3 phút tùy API)")
    print("⚠️  Không dùng checkpointer để tránh serialization error")

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

        state = initial_state(claim_text, language="vi")

        # Chạy KHÔNG với config thread_id để tránh checkpointer
        result = graph.invoke(state)

        print("\n" + "=" * 60)
        print("KẾT QUẢ THỰC TẾ")
        print("=" * 60)

        verdict = result.get("verdict", {})
        print(f"\nNhãn 4-class: {verdict.get('label')}")
        print(f"Nhãn binary: {verdict.get('verdict_binary')}")
        print(f"Nhãn tiếng Việt: {verdict.get('verdict_label_vi')}")
        print(f"Độ tin cậy: {verdict.get('confidence', 0):.2f}")

        print(f"\nLý do:")
        rationale = verdict.get("rationale", "N/A")
        if len(rationale) > 200:
            rationale = rationale[:200] + "..."
        print(f"  {rationale}")

        print(f"\nKết quả mô hình:")
        for model_result in result.get("model_results", []):
            status = "✓" if model_result.get("available") else "✗"
            print(
                f"  {status} {model_result['model']}: {model_result.get('label')} (conf={model_result.get('confidence', 0):.2f})"
            )

        print(f"\nSố lượng bằng chứng: {len(result.get('evidence', []))}")
        for i, evidence in enumerate(
            result.get("evidence", [])[:5], 1
        ):  # Hiển thị tối đa 5
            tier = evidence.get("source_tier", "unknown")
            print(f"  {i}. [{tier.upper()}] {evidence.get('source')}")
            print(f"     URL: {evidence.get('url')}")
            content = evidence.get("content", "")
            if len(content) > 60:
                content = content[:60] + "..."
            print(f"     Nội dung: {content}")

        print(f"\nReliability signal: {result.get('reliability_signal')}")
        print(
            f"   → Social search {'được kích hoạt' if result.get('reliability_signal') else 'bị bỏ qua'}"
        )

        print("\n" + "=" * 60)
        print("✅ DEMO THỰC TẾ THÀNH CÔNG!")
        print("=" * 60)
        print("Đây là kết quả thực tế từ:")
        print("  - Web search Tavily API")
        print("  - OpenAI GPT cho LLM")
        print("  - Local PhoBERT + COOLANT models")

    except Exception as e:
        print(f"\n❌ Lỗi khi chạy thực tế: {e}")
        import traceback

        traceback.print_exc()
else:
    print("\n❌ Thiếu API key để chạy thực tế")
    print("→ Vui lòng cài đặt OPENAI_API_KEY và TAVILY_API_KEY trong .env")
