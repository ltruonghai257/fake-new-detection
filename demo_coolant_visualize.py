"""
Visualize COOLANT multimodal comparison: image + caption
"""

from factcheck_agents.models import CoolantChecker
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

print("=" * 60)
print("COOLANT MULTIMODAL VISUALIZATION")
print("=" * 60)

# Create sample image with fake news content
demo_image_path = "/tmp/coolant_demo_image.jpg"
img = Image.new("RGB", (800, 400), color="white")
draw = ImageDraw.Draw(img)

# Draw fake news banner
draw.rectangle([0, 0, 800, 60], fill="red")
draw.text((20, 20), "CHÍNH PHỦ VIỆT NAM", fill="white", font=None)

# Draw fake news content
draw.text((20, 100), "THÔNG BÁO KHẨN", fill="black", font=None)
draw.text((20, 150), "TẶNG MỖI NGƯỜI DÂN 10 TRIỆU ĐỒNG", fill="red", font=None)
draw.text((20, 200), "ĐĂNG KÝ NGAY TẠI:", fill="black", font=None)
draw.text((20, 250), "www.chinhphu-tientunguoidan.com", fill="blue", font=None)
draw.text((20, 350), "Hạn chót: 30/6/2026", fill="gray", font=None)

img.save(demo_image_path)
print(f"\n✓ Đã tạo ảnh demo: {demo_image_path}")

# Display the image
plt.figure(figsize=(12, 6))
img_array = mpimg.imread(demo_image_path)
plt.imshow(img_array)
plt.title("Ảnh Input cho COOLANT", fontsize=14, fontweight="bold")
plt.axis("off")
plt.tight_layout()
plt.savefig("/tmp/coolant_visualization.png", dpi=100, bbox_inches="tight")
print("✓ Đã lưu visualization: /tmp/coolant_visualization.png")

# Caption/Statement
statement = "Chính phủ Việt Nam tặng mỗi người dân 10 triệu đồng khi đăng ký"
print(f"\n" + "=" * 60)
print("INPUT CHO COOLANT")
print("=" * 60)
print(f"\nCaption (Statement):")
print(f'  "{statement}"')
print(f"\nImage:")
print(f"  Path: {demo_image_path}")
print(f"  Content: Fake news banner về chính phủ tặng tiền")

print("\n" + "=" * 60)
print("COOLANT PROCESSING")
print("=" * 60)

print("\nStep 1: Text Feature Extraction")
print("  Input: Statement text")
print("  Model: PhoBERT (vinai/phobert-base-v2)")
print("  Output: Text features [1, 768, seq_len]")

print("\nStep 2: Image Feature Extraction")
print("  Input: Image file")
print("  Model: ResNet50 (pre-trained on ImageNet)")
print("  Output: Image features [1, 2048]")

print("\nStep 3: Multimodal Fusion")
print("  Combine: Text features + Image features")
print("  Modules:")
print("    - Similarity module: Text-image similarity")
print("    - CLIP module: Contrastive learning")
print("    - Detection module: Final classification")

print("\nStep 4: Classification")
print("  Output: REAL or FAKE")
print("  Confidence: Probability score")

# Run COOLANT prediction
print("\n" + "=" * 60)
print("COOLANT PREDICTION")
print("=" * 60)

coolant = CoolantChecker()
if coolant.load():
    result = coolant.predict(statement, demo_image_path)

    print(f"\nModel: {result['model']}")
    print(f"Available: {result['available']}")
    print(f"Label: {result.get('label', 'N/A')}")
    print(f"Label ID: {result.get('label_id', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 0):.4f}")
    print(f"Probabilities:")
    for label, prob in result.get("probabilities", {}).items():
        print(f"  {label}: {prob:.4f}")
    print(f"Note: {result.get('note', 'N/A')}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if result.get("available"):
        label = result.get("label")
        confidence = result.get("confidence", 0)

        if label == "REAL":
            print(f"\n✅ COOLANT đánh giá: THẬT (REAL)")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   → Ảnh và caption nhất quán với nhau")
        else:
            print(f"\n❌ COOLANT đánh giá: GIẢ (FAKE)")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   → Ảnh và caption mâu thuẫn hoặc có dấu hiệu giả mạo")

        print(f"\nCách COOLANT ra quyết định:")
        print(f"  - Phân tích consistency giữa text và image")
        print(f"  - Kiểm tra visual patterns có dấu hiệu manipulation không")
        print(f"  - So sánh với training patterns của real/fake news")
else:
    print(f"\n❌ COOLANT không load được: {coolant._load_error}")

# Cleanup
import os

os.remove(demo_image_path)
print(f"\n✓ Đã xóa demo ảnh: {demo_image_path}")

print("\n" + "=" * 60)
print("VÍ DỤ CÁC CẶP ẢNH + CAPTION ĐƯỢC SO SÁNH")
print("=" * 60)

examples = [
    {
        "caption": "Chính phủ tuyên bố gói hỗ trợ 60.000 tỷ",
        "image_desc": "Ảnh banner chính phủ với số tiền 60.000 tỷ",
        "expected": "Cần kiểm tra - có thể half-truth",
    },
    {
        "caption": "Sao Hỏa có sự sống",
        "image_desc": "Ảnh surface sao Hỏa với dấu hiệu nước",
        "expected": "Cần kiểm tra - scientific claim",
    },
    {
        "caption": "Elon Musk tuyên bố Tesla miễn phí",
        "image_desc": "Ảnh Elon Musk với logo Tesla",
        "expected": "Cần kiểm tra - celebrity endorsement",
    },
]

for i, ex in enumerate(examples, 1):
    print(f"\nExample {i}:")
    print(f"  Caption: '{ex['caption']}'")
    print(f"  Image: {ex['image_desc']}")
    print(f"  COOLANT sẽ so sánh: Text consistency với visual content")
    print(f"  Expected: {ex['expected']}")

print("\n" + "=" * 60)
print("COOLANT NHẬN DIỆN ĐƯỢC CÁC LOẠI TIN GIẢ:")
print("=" * 60)
print("  1. Memes với text sai sự thật")
print("  2. Ảnh được manipulat/edit (photoshop)")
print("  3. Screenshots giả mạo tin tức")
print("  4. Ảnh không khớp với caption")
print("  5. Deepfake hoặc AI-generated images")
print("  6. Context mismatch (ảnh cũ dùng cho tin mới)")
