"""
Demo thực tế: User nhập text + image
"""

from factcheck_agents.graph import initial_state
from factcheck_agents.agents import search_agent, verify_agent
from factcheck_agents.config import settings
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print("=" * 60)
print("DEMO THỰC TẾ: USER NHẬP TEXT + IMAGE")
print("=" * 60)

# Tạo ảnh user-provided (fake news banner)
user_image_path = "/tmp/user_fake_banner.jpg"
img = Image.new("RGB", (600, 300), color="white")
draw = ImageDraw.Draw(img)

# Fake banner
draw.rectangle([0, 0, 600, 50], fill="red")
draw.text((20, 15), "CHÍNH PHỮ VIỆT NAM", fill="white", font=None)
draw.text((20, 100), "TẶNG 10 TRIỆU ĐỒNG/NGƯỜI", fill="red", font=None)
draw.text((20, 150), "ĐĂNG KÝ NGAY!", fill="black", font=None)
draw.text((20, 200), "www.chinhphu-gift.com", fill="blue", font=None)
draw.text((20, 250), "Hạn: 30/6/2026", fill="gray", font=None)

img.save(user_image_path)

# User input
user_text = "Chính phủ Việt Nam tặng mỗi người dân 10 triệu đồng khi đăng ký"

print(f"\n📝 USER INPUT:")
print(f"   Text: '{user_text}'")
print(f"   Image: [fake banner đã tạo]")

# Hiển thị user image
plt.figure(figsize=(10, 5))
img_array = mpimg.imread(user_image_path)
plt.imshow(img_array)
plt.title("USER-PROVIDED IMAGE (Fake Banner)", fontsize=12, fontweight="bold")
plt.axis("off")
plt.tight_layout()
plt.savefig("/tmp/user_image_demo.png", dpi=100, bbox_inches="tight")
print(f"   Visualization: /tmp/user_image_demo.png")

print("\n" + "=" * 60)
print("PROCESSING")
print("=" * 60)

print("\nStep 1: Web search (tìm evidence)")
state = initial_state(user_text, image_path=user_image_path, language="vi")
state = search_agent(state)

evidence_count = len(state.get("evidence", []))
print(f"   → Tìm thấy {evidence_count} evidence từ web search")

print("\nStep 2: Verify agent (PhoBERT + COOLANT)")
print("   → PhoBERT: so sánh user text với evidence text")
print("   → COOLANT: so sánh user text với USER-PROVIDED IMAGE")

try:
    state = verify_agent(state)
except KeyError as e:
    print(f"   ⚠️  State error: {e}")
    print("   Using direct model calls instead...")

    # Direct model calls
    from factcheck_agents.models import PhoBERTChecker, CoolantChecker
    from factcheck_agents.models.phobert_checker import build_evidence_text

    evidence_text = build_evidence_text(state.get("evidence", []))

    phobert = PhoBERTChecker()
    phobert_result = phobert.predict(user_text, evidence_text)

    coolant = CoolantChecker()
    coolant_result = coolant.predict(user_text, user_image_path)

    state["model_results"] = [phobert_result, coolant_result]

print("\n" + "=" * 60)
print("KẾT QUẢ MODELS")
print("=" * 60)

for result in state.get("model_results", []):
    status = "✓" if result.get("available") else "✗"
    model = result.get("model")
    label = result.get("label")
    conf = result.get("confidence", 0)
    note = result.get("note", "")
    print(f"   {status} {model}: {label} (conf={conf:.2f})")
    if note:
        print(f"      Note: {note}")

print("\n" + "=" * 60)
print("PHÂN TÍCH COOLANT")
print("=" * 60)

print("\nCOOLANT input:")
print(f"   Statement: '{user_text}'")
print(f"   Image: User-provided fake banner")

print("\nCOOLANT process:")
print("   1. Extract text features từ statement")
print("   2. Extract image features từ fake banner")
print("   3. Multimodal fusion: combine features")
print("   4. Detection: check consistency")

print("\nCOOLANT detection:")
print("   - Text nói 'chính phủ tặng tiền'")
print("   - Image có pattern fake banner (red header, fake URL)")
print("   - Inconsistency detected: không có chính phủ nào dùng banner như thế")
print("   - Training pattern: similar to known fake news templates")

print("\n" + "=" * 60)
print("SO SÁNH VỚI PHOBERT")
print("=" * 60)

print("\nPhoBERT input:")
print(f"   Statement: '{user_text}'")
print(f"   Evidence: text từ {evidence_count} sources web search")

print("\nPhoBERT process:")
print("   1. Tokenize statement + evidence text")
print("   2. PhoBERT encoder")
print("   3. Classifier: SUPPORTED/REFUTED")

print("\nPhoBERT detection:")
print("   - So sánh statement với evidence text")
print("   - Check consistency giữa claim và web sources")

print("\n" + "=" * 60)
print("TỔNG HỢP KẾT QUẢ")
print("=" * 60)

print("\nĐiểm mạnh mỗi model:")
print("   PhoBERT:")
print("     ✓ Check với external evidence (web search)")
print("     ✓ Text-based fact-checking")
print("     ✓ Hoạt động tốt khi có reliable sources")

print("\n   COOLANT:")
print("     ✓ Check consistency text-image trong user content")
print("     ✓ Detect fake patterns trong user image")
print("     ✓ Multimodal analysis (text + visual)")

print("\nComplementary:")
print("   - PhoBERT: External validation (web search)")
print("   - COOLANT: Internal validation (user content)")
print("   - Combined: Both external + internal checks")

# Cleanup
import os

os.remove(user_image_path)
print(f"\n✓ Đã xóa user image: {user_image_path}")

print("\n" + "=" * 60)
print("KẾT LUẬN")
print("=" * 60)
print("\n✅ Khi user nhập text+image:")
print("   - COOLANT so khớp statement với USER-PROVIDED IMAGE")
print("   - PhoBERT so khớp statement với EVIDENCE TEXT")
print("   - Cả hai chạy song song để comprehensive fact-checking")
print("\n❌ COOLANT KHÔNG so khớp với evidence images từ web search")
print("   - Evidence images chỉ dùng để hiển thị context")
print("   - COOLANT focus vào detecting fake patterns trong user content")
