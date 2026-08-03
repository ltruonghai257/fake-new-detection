"""
Kiểm tra tại sao COOLANT unavailable
"""

from factcheck_agents.config import settings
import os

print("=" * 60)
print("KIỂM TRA CẤU HÌNH COOLANT")
print("=" * 60)

# Kiểm tra các biến môi trường COOLANT
print("\nBiến môi trường:")
print(f"  COOLANT_CKPT_PATH: {os.getenv('COOLANT_CKPT_PATH')}")
print(f"  FACTCHECK_DEVICE: {os.getenv('FACTCHECK_DEVICE', 'auto')}")

# Kiểm tra config
print(f"\nConfig settings:")
print(f"  coolant_ckpt_path: {settings.coolant_ckpt_path}")
print(f"  device: {settings.device}")
print(f"  data_root: {settings.data_root}")
print(f"  coolant_search_root: {settings.coolant_search_root()}")

# Kiểm tra file checkpoint có tồn tại không
if settings.coolant_ckpt_path:
    if os.path.exists(settings.coolant_ckpt_path):
        print(f"  ✓ Checkpoint file tồn tại: {settings.coolant_ckpt_path}")
        file_size = os.path.getsize(settings.coolant_ckpt_path)
        print(f"    File size: {file_size / (1024*1024):.2f} MB")
    else:
        print(f"  ✗ Checkpoint file KHÔNG tồn tại: {settings.coolant_ckpt_path}")
else:
    print("  ✗ coolant_ckpt_path không được cấu hình")
    print(f"  → Tự động tìm trong: {settings.coolant_search_root()}")

    # Tự động tìm checkpoint file
    search_root = settings.coolant_search_root()
    if os.path.exists(search_root):
        print(f"  ✓ Thư mục search root tồn tại")
        import glob

        ckpt_files = glob.glob(str(search_root / "*.pt")) + glob.glob(
            str(search_root / "*.pth")
        )
        if ckpt_files:
            print(f"  → Tìm thấy {len(ckpt_files)} checkpoint file(s):")
            for f in ckpt_files:
                print(f"      - {f}")
        else:
            print(f"  ✗ Không tìm thấy checkpoint file nào trong {search_root}")
    else:
        print(f"  ✗ Thư mục search root KHÔNG tồn tại: {search_root}")

# Thử load COOLANT để xem lỗi cụ thể
print("\n" + "=" * 60)
print("THỬ LOAD COOLANT MODEL")
print("=" * 60)

try:
    from factcheck_agents.agents.verify_agent import _get_coolant_model

    print("Đang thử load COOLANT model...")
    model = _get_coolant_model()
    if model is None:
        print("✗ COOLANT model trả về None")
    else:
        print(f"✓ COOLANT model loaded thành công: {type(model)}")
except Exception as e:
    print(f"✗ Lỗi khi load COOLANT: {e}")
    import traceback

    traceback.print_exc()

# Thử chạy COOLANT inference
print("\n" + "=" * 60)
print("THỬ CHẠY COOLANT INFERENCE")
print("=" * 60)

try:
    from factcheck_agents.agents.verify_agent import _run_coolant_inference

    print("Đang thử chạy COOLANT inference...")

    # Test với sample data
    test_statement = "Test statement for COOLANT"
    test_evidence = [{"content": "Sample evidence text", "source": "test.com"}]

    result = _run_coolant_inference(test_statement, test_evidence, None)
    print(f"✓ COOLANT inference result: {result}")
except Exception as e:
    print(f"✗ Lỗi khi chạy COOLANT inference: {e}")
    import traceback

    traceback.print_exc()
