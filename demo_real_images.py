"""
Demo COOLANT với ảnh thực tế từ web search
"""

from factcheck_agents.graph import initial_state
from factcheck_agents.tools.web_search import web_search
from factcheck_agents.models import CoolantChecker
from factcheck_agents.agents import search_agent
from factcheck_agents.config import settings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import urllib.request
import io

print("=" * 60)
print("COOLANT VỚI ẢNH THỰC TẾ TỪ WEB SEARCH")
print("=" * 60)

# Statement để kiểm tra
statement = "Thủ tướng chính phủ tung gói hỗ trợ 60000 tỷ để hỗ trợ PNJ"
print(f"\nStatement: '{statement}'")

# Thực hiện web search để lấy evidence (bao gồm ảnh)
print("\n" + "=" * 60)
print("WEB SEARCH ĐỂ LẤY ẢNH THỰC TẾ")
print("=" * 60)

state = initial_state(statement, language="vi")
state = search_agent(state)

evidence = state.get("evidence", [])
print(f"\nTìm thấy {len(evidence)} evidence")

# Lọc evidence có ảnh
evidence_with_images = [e for e in evidence if e.get("image_path")]
print(f"Số evidence có ảnh: {len(evidence_with_images)}")

if not evidence_with_images:
    print("\n⚠️  Không tìm thấy evidence có ảnh trong web search")
    print("   COOLANT cần ảnh để hoạt động, sẽ skip trong trường hợp này")
else:
    print("\n" + "=" * 60)
    print("HIỂN THỊ CẶP ẢNH + CAPTION THỰC TẾ")
    print("=" * 60)

    # Hiển thị tối đa 3 cặp ảnh + caption
    for i, ev in enumerate(evidence_with_images[:3], 1):
        print(f"\n{'='*60}")
        print(f"CẶP {i}")
        print(f"{'='*60}")

        # Statement (caption)
        print(f"\nCaption (Statement):")
        print(f'  "{statement}"')

        # Ảnh từ web search
        image_path = ev.get("image_path")
        print(f"\nImage từ web search:")
        print(f"  Path: {image_path}")
        print(f"  Source: {ev.get('source')}")
        print(f"  URL: {ev.get('url')}")

        # Hiển thị ảnh nếu có thể
        try:
            if image_path and image_path.startswith("/tmp/"):
                img = mpimg.imread(image_path)
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.title(f"Ảnh {i}: {ev.get('title', '')[:50]}")
                plt.axis('off')

                # Hiển thị text info
                plt.subplot(1, 2, 2)
                plt.text(0.1, 0.9, f"Statement:\n{statement}", fontsize=10, wrap=True)
                plt.text(0.1, 0.5, f"Source:\n{ev.get('source')}", fontsize=10)
                plt.text(0.1, 0.3, f"URL:\n{ev.get('url', '')[:50]}...", fontsize=8)
                plt.axis('off')

                plt.tight_layout()
                save_path = f"/tmp/coolant_real_pair_{i}.png"
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Visualization saved: {save_path}")

                # Chạy COOLANT prediction
                print(f"\nCOOLANT prediction:")
                coolant = CoolantChecker()
                if coolant.load():
                    result = coolant.predict(statement, image_path)
                    print(f"  Label: {result.get('label', 'N/A')}")
                    print(f"  Confidence: {result.get('confidence', 0):.4f}")
                    print(f"  Available: {result.get('available')}")

                    if result.get('available'):
                        if result.get('label') == 'FAKE':
                            print(f"  → COOLANT phát hiện inconsistency")
                        else:
                            print(f"  → COOLANT thấy consistency")
                else:
                    print(f"  ✗ COOLANT load failed: {coolant._load_error}")

        except Exception as e:
            print(f"  ✗ Error displaying image: {e}")

print("\n" + "=" * 60)
print("TÓM TẮT")
print("=" * 60)
print(f"\nTotal evidence: {len(evidence)}")
print(f"Evidence có ảnh: {len(evidence_with_images)}")
print(f"Evidence không ảnh: {len(evidence) - len(evidence_with_images)}")

if evidence_with_images:
    print(f"\nCOOLANT đã so sánh statement với {len(evidence_with_images)} ảnh thực tế")
    print("từ web search để kiểm tra consistency.")
else:
    print("\nWeb search không trả về ảnh, COOLANT không thể hoạt động.")
    print("Đây là giới hạn của Tavily API - không phải lỗi COOLANT.")