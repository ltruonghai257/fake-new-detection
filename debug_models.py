"""
Debug script để hiểu cách models hoạt động
"""

from factcheck_agents.models import PhoBERTChecker, CoolantChecker
from factcheck_agents.models.phobert_checker import build_evidence_text

print("=" * 60)
print("DEBUG: CÁCH MODELS HOẠT ĐỘNG")
print("=" * 60)

# Example data
statement = "Thủ tướng chính phủ tung gói hỗ trợ 60000 tỷ để hỗ trợ PNJ"
evidence_sample = [
    {
        "source": "vnexpress.net",
        "url": "https://vnexpress.net/nghien-cuu-nang-goi-tin-dung-uu-dai-thuy-san-len-60-000-ty-dong",
        "content": "Nghiên cứu nâng gói tín dụng ưu đãi thủy sản lên 60.000 tỷ đồng"
    }
]

print("\n" + "=" * 60)
print("1. PHOBERT (TEXT-ONLY MODEL)")
print("=" * 60)
print("\nInput:")
print(f"  Statement: '{statement}'")
print(f"  Evidence text: '{build_evidence_text(evidence_sample)}'")
print("\nProcess:")
print("  - Tokenize: statement + evidence_text")
print("  - PhoBERT encoder: [CLS] statement [SEP] evidence [SEP]")
print("  - Classifier: 3 classes (SUPPORTED, REFUTED, NEI)")
print("  - Output: Label dựa trên sự nhất quán giữa statement và evidence")

print("\n" + "=" * 60)
print("2. COOLANT (MULTIMODAL MODEL)")
print("=" * 60)
print("\nInput:")
print(f"  Statement: '{statement}'")
print(f"  Image path: /path/to/image.jpg (nếu có)")
print("\nProcess:")
print("  - Text encoder: PhoBERT extract features từ statement")
print("  - Image encoder: ResNet50 extract features từ image")
print("  - Multimodal fusion: Combine text + image features")
print("  - Detection module: Binary classification (REAL/FAKE)")
print("  - Output: Label dựa trên consistency giữa text và image")

print("\n" + "=" * 60)
print("PHOBERT CHECKPOINT INFO")
print("=" * 60)

phobert = PhoBERTChecker()
if phobert.load():
    print(f"✓ PhoBERT loaded")
    print(f"  Device: {phobert._device}")
    print(f"  Labels: {phobert._labels}")
    print(f"  Max length: {phobert._max_length}")
else:
    print(f"✗ PhoBERT load failed: {phobert._load_error}")

print("\n" + "=" * 60)
print("COOLANT CHECKPOINT INFO")
print("=" * 60)

coolant = CoolantChecker()
if coolant.load():
    print(f"✓ COOLANT loaded")
    print(f"  Device: {coolant._device}")
    print(f"  Image model: {coolant._image_model}")
else:
    print(f"✗ COOLANT load failed: {coolant._load_error}")

print("\n" + "=" * 60)
print("TẠI SAO PHOBERT LUÔN SUPPORTED?")
print("=" * 60)
print("\nCác nguyên nhân có thể:")
print("  1. Model bias: Training data có nhiều positive samples")
print("  2. Evidence text không đủ rõ ràng để refutation")
print("  3. Statement ngắn, evidence không đầy đủ")
print("  4. Model trained trên dataset khác với use case hiện tại")
print("  5. Threshold quyết định label có thể thấp")

print("\n" + "=" * 60)
print("PHOBERT MODEL LÀ GÌ?")
print("=" * 60)
print("\nBase model: vinai/phobert-base-v2")
print("  - Pre-trained RoBERTa cho tiếng Việt")
print("  - 125M parameters")
print("  - Trained trên large Vietnamese corpus")
print("\nFine-tuned: ViFactCheck dataset")
print("  - Task: Fact-checking classification")
print("  - Classes: SUPPORTED, REFUTED, NEI (Not Enough Info)")
print("  - Input: Statement + Evidence text")
print("  - Architecture: CLS-token classifier on top of PhoBERT")
print("  - Performance: ~81% validation macro-F1")