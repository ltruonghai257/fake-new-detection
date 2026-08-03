"""
Demo kiểm tra bài đăng Facebook từ URL
"""

import sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from factcheck_agents.graph import build_graph, initial_state
from factcheck_agents.config import settings

warnings.simplefilter("always")

def scrape_facebook_post(url: str):
    """
    Scrape text và ảnh từ Facebook post.

    Lưu ý: Facebook có anti-scraping mạnh, nên method này có thể không hoạt động.
    Nếu thất bại, sẽ hướng dẫn manual extraction.
    """
    print(f"Đang scrape Facebook URL: {url}")

    try:
        # Thử simple scraping (có thể bị Facebook block)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: Facebook likely blocked the request")

        soup = BeautifulSoup(response.text, 'html.parser')

        # Thử extract text (có thể không hoạt động do Facebook dynamic content)
        # Facebook uses React, nên simple scraping thường không hoạt động
        print("⚠️  Facebook sử dụng dynamic content, simple scraping có thể không hoạt động")
        print("   Hãy sử dụng phương pháp manual extraction thay thế")

        return None, None

    except Exception as e:
        print(f"❌ Scrape thất bại: {e}")
        print("\n" + "=" * 60)
        print("PHƯƠNG PHÁP MANUAL EXTRACTION")
        print("=" * 60)
        print("Do Facebook anti-scraping, hãy làm theo các bước sau:")
        print("1. Copy text từ bài Facebook")
        print("2. Save ảnh từ bài Facebook")
        print("3. Sử dụng demo với text và ảnh đó")
        return None, None

def manual_facebook_demo():
    """
    Demo với manual input từ Facebook post
    """
    print("=" * 60)
    print("DEMO KIỂM TRA BÀI FACEBOOK (MANUAL)")
    print("=" * 60)

    # Example Facebook post about fake news
    # Bạn có thể thay đổi text này bằng text từ Facebook thật
    fb_text = """
    Chính phủ vừa ban hành quyết định tặng mỗi người dân 10 triệu đồng
    để hỗ trợ kinh tế sau đại dịch. Mọi người cần đăng ký ngay tại
    trang web chinhphu-tientunguoidan.com trước ngày 30/6/2026.
    """

    fb_text = fb_text.strip()

    print(f"\nText từ Facebook:")
    print(f'  "{fb_text}"')

    # Tạo demo ảnh (trong thực tế bạn sẽ save ảnh từ Facebook)
    demo_image_path = "/tmp/demo_facebook_image.jpg"
    try:
        img = Image.new('RGB', (600, 300), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 50), "CHÍNH PHỦ VIỆT NAM", fill='blue', font=None)
        draw.text((20, 100), "TẶNG 10 TRIỆU ĐỒNG", fill='red', font=None)
        draw.text((20, 150), "ĐĂNG KÝ NGAY!", fill='black', font=None)
        draw.text((20, 200), "chinhphu-tientunguoidan.com", fill='gray', font=None)
        img.save(demo_image_path)
        print(f"\n✓ Đã tạo demo ảnh: {demo_image_path}")
        print("  (Trong thực tế: save ảnh thật từ Facebook)")
    except Exception as e:
        print(f"\n⚠️  Không thể tạo demo ảnh: {e}")
        demo_image_path = None

    return fb_text, demo_image_path

def run_factcheck(text: str, image_path: str = None):
    """
    Chạy fact-check với text và ảnh
    """
    print("\n" + "=" * 60)
    print("CHẠY FACT-CHECK")
    print("=" * 60)

    print("\nKiểm tra cấu hình:")
    print(f"  OPENAI_API_KEY: {'✓ Đã cài đặt' if settings.has_llm() else '✗ Chưa cài đặt'}")
    print(f"  TAVILY_API_KEY: {'✓ Đã cài đặt' if settings.has_search() else '✗ Chưa cài đặt'}")
    print(f"  COOLANT_CKPT_PATH: {'✓ Đã cài đặt' if settings.coolant_ckpt_path else '✗ Chưa cài đặt'}")

    if not settings.has_llm() or not settings.has_search():
        print("\n❌ Thiếu API key để chạy thực tế")
        return

    print("\n⏳ Đang chạy fact-check...")

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

        graph = g.compile()

        state = initial_state(text, image_path=image_path, language="vi")

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
        rationale = verdict.get('rationale', 'N/A')
        if len(rationale) > 300:
            rationale = rationale[:300] + "..."
        print(f"  {rationale}")

        print(f"\nKết quả mô hình:")
        for model_result in result.get("model_results", []):
            status = "✓" if model_result.get("available") else "✗"
            note = f" - {model_result.get('note', '')}" if model_result.get('note') else ""
            print(f"  {status} {model_result['model']}: {model_result.get('label')} (conf={model_result.get('confidence', 0):.2f}){note}")

        print(f"\nBằng chứng (tối đa 5):")
        for i, evidence in enumerate(result.get("evidence", [])[:5], 1):
            tier = evidence.get("source_tier", "unknown")
            print(f"  {i}. [{tier.upper()}] {evidence.get('source')}")
            print(f"     URL: {evidence.get('url')}")

        print(f"\nReliability signal: {result.get('reliability_signal')}")

    except Exception as e:
        print(f"\n❌ Lỗi khi chạy: {e}")
        import traceback
        traceback.print_exc()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kiểm tra bài đăng Facebook")
    parser.add_argument("--url", type=str, help="URL bài Facebook (scrape có thể không hoạt động)")
    parser.add_argument("--text", type=str, help="Text từ bài Facebook (manual)")
    parser.add_argument("--image", type=str, help="Đường dẫn ảnh từ bài Facebook (manual)")

    args = parser.parse_args()

    if args.url:
        # Thử scrape từ URL
        text, image = scrape_facebook_post(args.url)
        if text and image:
            run_factcheck(text, image)
        else:
            # Fallback to manual demo
            text, image = manual_facebook_demo()
            run_factcheck(text, image)
    elif args.text:
        # Manual input
        image_path = args.image if args.image else None
        run_factcheck(args.text, image_path)
    else:
        # Demo mặc định
        text, image = manual_facebook_demo()
        run_factcheck(text, image)

        # Cleanup demo image
        if image and image.startswith("/tmp/"):
            import os
            if os.path.exists(image):
                os.remove(image)
                print(f"\n✓ Đã xóa demo ảnh: {image}")

if __name__ == "__main__":
    main()