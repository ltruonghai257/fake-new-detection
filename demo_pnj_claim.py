"""
Demo kiểm tra thông tin: "Thủ tướng chính phủ tung gói hỗ trợ 60000 tỷ để hỗ trợ PNJ"
"""

from factcheck_agents.graph import build_graph, initial_state
from factcheck_agents.config import settings

print("=" * 60)
print("KIỂM TRA: THỦ TƯỚNG TUNG GÓI HỖ TRỢ 60000 TỶ CHO PNJ")
print("=" * 60)

claim_text = "Thủ tướng chính phủ tung gói hỗ trợ 60000 tỷ để hỗ trợ PNJ"
print(f"\nThông tin kiểm tra: '{claim_text}'")

print("\nKiểm tra cấu hình:")
print(f"  OPENAI_API_KEY: {'✓ Đã cài đặt' if settings.has_llm() else '✗ Chưa cài đặt'}")
print(
    f"  TAVILY_API_KEY: {'✓ Đã cài đặt' if settings.has_search() else '✗ Chưa cài đặt'}"
)

if settings.has_llm() and settings.has_search():
    print("\n" + "=" * 60)
    print("CHẠY FACT-CHECK VỚI WEB SEARCH + LLM")
    print("=" * 60)
    print("⏳ Đang chạy...")

    try:
        from langgraph.graph import END, START, StateGraph
        from factcheck_agents.agents import (
            conclusion_agent,
            search_agent,
            social_search_agent,
            verify_agent,
        )
        from factcheck_agents.state import FactCheckState

        def route_after_verify(state: FactCheckState) -> str:
            if state.get("reliability_signal"):
                return "social_search"
            return "conclusion"

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

        graph = g.compile()

        state = initial_state(claim_text, language="vi")

        result = graph.invoke(state)

        print("\n" + "=" * 60)
        print("KẾT QUẢ KIỂM TRA")
        print("=" * 60)

        verdict = result.get("verdict", {})
        print(f"\nNhãn 4-class: {verdict.get('label')}")
        print(f"Nhãn binary: {verdict.get('verdict_binary')}")
        print(f"Nhãn tiếng Việt: {verdict.get('verdict_label_vi')}")
        print(f"Độ tin cậy: {verdict.get('confidence', 0):.2f}")

        print(f"\nLý do:")
        rationale = verdict.get("rationale", "N/A")
        if len(rationale) > 400:
            rationale = rationale[:400] + "..."
        print(f"  {rationale}")

        print(f"\nKết quả mô hình:")
        for model_result in result.get("model_results", []):
            status = "✓" if model_result.get("available") else "✗"
            note = (
                f" - {model_result.get('note', '')}" if model_result.get("note") else ""
            )
            print(
                f"  {status} {model_result['model']}: {model_result.get('label')} (conf={model_result.get('confidence', 0):.2f}){note}"
            )

        print(f"\nBằng chứng tìm được:")
        for i, evidence in enumerate(result.get("evidence", [])[:5], 1):
            tier = evidence.get("source_tier", "unknown")
            print(f"  {i}. [{tier.upper()}] {evidence.get('source')}")
            print(f"     URL: {evidence.get('url')}")
            print(
                f"     Snippet: {evidence.get('snippet', '')[:100] if evidence.get('snippet') else 'EMPTY'}"
            )
            print(
                f"     Content: {evidence.get('content', '')[:100] if evidence.get('content') else 'EMPTY'}"
            )
            print(f"     All keys: {list(evidence.keys())}")

        print(f"\nReliability signal: {result.get('reliability_signal')}")
        print(
            f"   → Social search {'được kích hoạt' if result.get('reliability_signal') else 'bị bỏ qua'}"
        )

        print("\n" + "=" * 60)
        print("✅ KIỂM TRA HOÀN TẤT")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Lỗi khi chạy: {e}")
        import traceback

        traceback.print_exc()
else:
    print("\n❌ Thiếu API key để chạy thực tế")
