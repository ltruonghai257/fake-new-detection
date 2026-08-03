"""
Đánh giá khả năng train thành công của các notebook pipeline
"""

print("=" * 60)
print("ĐÁNH GIÁ NOTEBOOK TRAINING")
print("=" * 60)

training_notebooks = {
    "03.9_vifactcheck_training.ipynb": {
        "model": "PhoBERT ViFactCheck (Enhanced)",
        "data": "HuggingFace tranthaihoa/vifactcheck",
        "complexity": "Medium",
        "dependencies": ["torch", "transformers", "datasets", "mlflow"],
        "pros": [
            "Enhanced architecture với CLS-token classifier",
            "Supports multiple label strategies",
            "MLflow logging",
            "Early stopping",
            "Warmup scheduler",
        ],
        "cons": ["Cần HuggingFace dataset access", "PhoBERT-base (nhỏ hơn original)"],
        "success_probability": "HIGH",
    },
    "03.9_vifactcheck_original_training.ipynb": {
        "model": "PhoBERT ViFactCheck (Original Paper)",
        "data": "HuggingFace tranthaihoa/vifactcheck",
        "complexity": "Low",
        "dependencies": ["torch", "transformers", "datasets"],
        "pros": [
            "Theo sát paper gốc (AAAI 2025)",
            "Architecture đơn giản (pooler output)",
            "Không cần MLflow",
            "Training code đơn giản",
        ],
        "cons": [
            "PhoBERT-large (nặng hơn)",
            "Không có early stopping",
            "Fixed 10 epochs",
            "Không có scheduler",
        ],
        "success_probability": "VERY HIGH",
    },
    "03_coolant_training.ipynb": {
        "model": "COOLANT (Multimodal)",
        "data": "HDF5 files (coolant_train.h5, etc)",
        "complexity": "High",
        "dependencies": ["torch", "h5py", "mlflow"],
        "pros": [
            "Đã train thành công trước đó",
            "Checkpoint đã tồn tại",
            "Multimodal (text + image)",
        ],
        "cons": [
            "Cần HDF5 preprocessed files",
            "Complex architecture",
            "Cần nhiều GPU memory",
        ],
        "success_probability": "VERY HIGH",
    },
    "03a_anchored_coolant_training.ipynb": {
        "model": "Anchored COOLANT",
        "data": "HDF5 files",
        "complexity": "Very High",
        "dependencies": ["torch", "h5py", "mlflow"],
        "pros": ["Advanced multimodal", "Anchored training"],
        "cons": ["Rất complex", "Cần HDF5 files đặc biệt", "Chưa test kỹ"],
        "success_probability": "MEDIUM",
    },
    "03.5_ai_art_detection_training.ipynb": {
        "model": "AI Art Detection",
        "data": "Custom dataset",
        "complexity": "Medium",
        "dependencies": ["torch", "transformers"],
        "pros": ["AI art detection useful", "Modern architecture"],
        "cons": ["Cần custom dataset", "Chưa rõ data availability"],
        "success_probability": "MEDIUM",
    },
    "07c_vietnamese_coolant_finetune.ipynb": {
        "model": "Vietnamese COOLANT Finetune",
        "data": "Vietnamese dataset",
        "complexity": "High",
        "dependencies": ["torch", "h5py"],
        "pros": ["Vietnamese-specific", "Finetune từ pretrained"],
        "cons": ["Cần Vietnamese dataset", "Complex setup"],
        "success_probability": "MEDIUM",
    },
}

print("\n" + "=" * 60)
print("BẢNG ĐÁNH GIÁ")
print("=" * 60)

for notebook, info in training_notebooks.items():
    print(f"\n📓 {notebook}")
    print(f"   Model: {info['model']}")
    print(f"   Data: {info['data']}")
    print(f"   Complexity: {info['complexity']}")
    print(f"   Success Probability: {info['success_probability']}")
    print(f"   Dependencies: {', '.join(info['dependencies'][:3])}")
    if len(info["dependencies"]) > 3:
        print(f"                    + {len(info['dependencies']) - 3} more")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)

print("\n🏆 TOP 3 NOTEBOOK ĐỀ TRAIN THÀNH CÔNG:")
print("\n1️⃣ 03.9_vifactcheck_original_training.ipynb (VERY HIGH)")
print("   - Theo sát paper gốc")
print("   - Architecture đơn giản")
print("   - Ít dependencies")
print("   - Dễ debug")

print("\n2️⃣ 03_coolant_training.ipynb (VERY HIGH)")
print("   - Đã train thành công trước đó")
print("   - Checkpoint đã tồn tại")
print("   - Có thể resume hoặc retrain")

print("\n3️⃣ 03.9_vifactcheck_training.ipynb (HIGH)")
print("   - Enhanced architecture")
print("   - Tính năng hiện đại")
print("   - MLflow logging")

print("\n" + "=" * 60)
print("PHÂN TÍCH CHI TIẾT")
print("=" * 60)

print("\n🔍 TẠI SAO PHOBERT CÓ THỂ FAIL?")
print("\nNguyên nhân có thể:")
print("  1. HuggingFace dataset access issue")
print("  2. Memory不足 (MPS device warning)")
print("  3. Data preprocessing error")
print("  4. Tokenizer mismatch")

print("\n💡 GIẢI PHÁP:")
print("  - Dùng 03.9_vifactcheck_original_training.ipynb (đơn giản hơn)")
print("  - Enable smoke_test mode để test nhanh")
print("  - Check HuggingFace access:")
print("    from datasets import load_dataset")
print("    ds = load_dataset('tranthaihoa/vifactcheck')")

print("\n🎯 KHUYẾN NHẤT:")
print("  1. Chạy 03.9_vifactcheck_original_training.ipynb trước")
print("  2. Nếu success → thử 03.9_vifactcheck_training.ipynb")
print("  3. COOLANT đã có checkpoint → có thể skip training")
print("  4. Focus vào inference và integration thay vì retrain")
