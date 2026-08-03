"""
Kiểm tra COOLANT loader đơn giản
"""

from factcheck_agents.models import CoolantChecker

print("=" * 60)
print("KIỂM TRA COOLANT LOADER")
print("=" * 60)

print("\nĐang khởi tạo CoolantChecker...")
checker = CoolantChecker()

print("Đang cố gắng load model...")
success = checker.load()

if success:
    print("✅ COOLANT model loaded thành công!")
    print(f"   Device: {checker._device}")
    print(f"   Image model: {checker._image_model}")
else:
    print(f"❌ COOLANT model load thất bại!")
    print(f"   Lỗi: {checker._load_error}")

# Test predict với image (nếu có)
print("\n" + "=" * 60)
print("TEST COOLANT PREDICT")
print("=" * 60)

if success:
    # Tạo một ảnh test đơn giản
    try:
        from PIL import Image, ImageDraw
        test_image_path = "/tmp/test_coolant_image.jpg"
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), "Test image for COOLANT", fill='black')
        img.save(test_image_path)
        print(f"✓ Đã tạo ảnh test: {test_image_path}")

        # Test predict
        print("Đang test predict với ảnh...")
        result = checker.predict("Test statement for COOLANT", test_image_path)
        print(f"✅ Predict result: {result}")

        # Cleanup
        import os
        os.remove(test_image_path)
        print(f"✓ Đã xóa ảnh test")

    except ImportError:
        print("⚠️  PIL không có sẵn, không thể tạo ảnh test")
    except Exception as e:
        print(f"❌ Lỗi khi test predict: {e}")
else:
    print("⚠️  Bỏ qua test predict vì model không load được")

# Test predict KHÔNG có image (expected to fail gracefully)
print("\n" + "=" * 60)
print("TEST COOLANT PREDICT KHÔNG ẢNH (EXPECTED FAIL)")
print("=" * 60)

result = checker.predict("Test statement without image", None)
print(f"Result (không ảnh): {result}")
if not result.get("available"):
    print("✅ Correctly returns unavailable when no image provided")
else:
    print("❌ Should return unavailable when no image provided")