#!/usr/bin/env python3
"""
Add synthetic labels to ViFactCheck JSON files FOR TESTING ONLY.

WARNING: This creates random labels for pipeline testing.
For real experiments, you MUST use actual ViFactCheck verification results!
"""
import json
import random
from pathlib import Path

random.seed(42)  # Reproducible

data_root = Path("/Users/haila/Library/CloudStorage/GoogleDrive-latruonghai@gmail.com/My Drive/Thesis_Final/fake-news-data-for-thesis")
json_dir = data_root / "data" / "json"

# Binary classification: 0 = real, 1 = misinformation
# Creates roughly balanced classes
for split in ["train", "dev", "test"]:
    json_path = json_dir / f"news_data_vifactcheck_{split}_labeled.json"
    
    if not json_path.exists():
        print(f"⚠️  File not found: {json_path}")
        continue
    
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    
    # Add synthetic labels (roughly 50/50 split)
    label_0_count = 0
    label_1_count = 0
    for i, art in enumerate(articles):
        # Deterministic based on index for reproducibility
        label = 0 if i % 2 == 0 else 1
        art["label"] = label
        if label == 0:
            label_0_count += 1
        else:
            label_1_count += 1
    
    # Save back
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {split}: Added synthetic labels to {len(articles)} articles")
    print(f"   Class distribution: 0={label_0_count}, 1={label_1_count}")

print("\n" + "="*60)
print("⚠️  WARNING: These are SYNTHETIC labels for testing only!")
print("   For real experiments, replace with actual ViFactCheck")
print("   claim verification results.")
print("="*60)
