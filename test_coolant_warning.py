"""
Test COOLANT warning khi không có ảnh
"""

import warnings
from factcheck_agents.models import CoolantChecker

print("=" * 60)
print("TEST COOLANT WARNING")
print("=" * 60)

# Enable all warnings
warnings.simplefilter("always")

checker = CoolantChecker()

print("\nTest 1: Không có ảnh (expected warning)")
result = checker.predict("Test statement", None)
print(f"Result: {result}")

print("\nTest 2: Ảnh không tồn tại (expected warning)")
result = checker.predict("Test statement", "/nonexistent/image.jpg")
print(f"Result: {result}")

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)