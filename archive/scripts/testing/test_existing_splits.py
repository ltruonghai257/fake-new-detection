#!/usr/bin/env python3
"""
Test Processing Existing Vietnamese Dataset Splits
Uses the new process_existing_splits function to process train/dev/test files
"""

import sys
import os
import torch

# Add src to path
sys.path.append('./src')

from preprocessing.combined_preprocessing import CombinedPreprocessor

def main():
    """Test processing existing Vietnamese dataset splits"""
    
    print("🇻🇳 Processing Existing Vietnamese Dataset Splits")
    print("=" * 60)
    
    # Proper device detection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"🔧 Using device: {device}")
    
    # Initialize combined preprocessor for Vietnamese
    combined_preprocessor = CombinedPreprocessor(
        text_model_name="vinai/phobert-base",
        image_model_name="resnet18",
        language="vi",
        max_text_length=512,
        image_size=(224, 224),
        device=device
    )
    
    print(f"✓ Text model: {combined_preprocessor.text_preprocessor.model_name}")
    print(f"✓ Image model: {combined_preprocessor.image_preprocessor.model_name}")
    
    # Process existing splits
    print("\n📊 Processing existing splits...")
    
    # Example 1: Use default prefix (news_data_vifactcheck_)
    print("Using default file prefix: 'news_data_vifactcheck_'")
    results = combined_preprocessor.process_existing_splits(
        data_dir="./src/data/json",
        save_base_dir="./processed_data",
        save_format="pkl",
        batch_size=32,
        splits=["train", "dev", "test"],
        file_prefix="news_data_vifactcheck_"  # Default prefix
    )
    
    # Example 2: Custom prefix usage (commented out)
    # print("Using custom file prefix: 'my_dataset_'")
    # custom_results = combined_preprocessor.process_existing_splits(
    #     data_dir="./src/data/json",
    #     save_base_dir="./processed_data",
    #     save_format="pkl",
    #     batch_size=32,
    #     splits=["train", "dev", "test"],
    #     file_prefix="my_dataset_"  # Custom prefix
    # )
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Vietnamese Dataset Processing Completed!")
    print("\n📊 Summary:")
    
    total_samples = 0
    for split_name, (dataset, metadata) in results.items():
        total_samples += len(dataset)
        print(f"   {split_name.upper()}: {len(dataset)} samples")
        print(f"     Text features: {metadata['text_feature_shape']}")
        print(f"     Image features: {metadata['image_feature_shape']}")
    
    print(f"\n🚀 Total: {total_samples} samples ready for COOLANT training!")
    
    print("\n📁 Processed files:")
    for split_name in results.keys():
        print(f"   ./processed_data/vietnamese_{split_name}/")
        print(f"     - text_features.pkl")
        print(f"     - image_features.pkl")
        print(f"     - combined_dataset.pkl")
        print(f"     - metadata.json")
    
    print("\n✨ Your Vietnamese multimodal dataset is ready for COOLANT!")

if __name__ == "__main__":
    main()
