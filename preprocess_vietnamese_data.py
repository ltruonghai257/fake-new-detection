#!/usr/bin/env python3
"""
Quick Start Script: Preprocess Vietnamese Fake News Data
Run this script to process your crawled Vietnamese news data.

Usage:
    python preprocess_vietnamese_data.py
"""

import sys
import os
sys.path.append('./src')

from preprocessing import CombinedPreprocessor
import torch

def main():
    """Main preprocessing workflow"""
    
    print("=" * 70)
    print("🇻🇳 VIETNAMESE FAKE NEWS DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Configuration
    config = {
        "text_model": "vinai/phobert-base",  # Vietnamese PhoBERT
        "image_model": "resnet18",           # ResNet for images
        "language": "vi",                    # Vietnamese
        "max_text_length": 512,
        "batch_size": 16,                    # Adjust based on your memory
        "save_format": "pkl",                # 'pkl' or 'npz'
    }
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"\n🔧 Configuration:")
    print(f"  • Text Model: {config['text_model']}")
    print(f"  • Image Model: {config['image_model']}")
    print(f"  • Device: {device}")
    print(f"  • Batch Size: {config['batch_size']}")
    print(f"  • Output Format: {config['save_format']}")
    
    # Initialize preprocessor
    print(f"\n🚀 Initializing preprocessor...")
    preprocessor = CombinedPreprocessor(
        text_model_name=config['text_model'],
        image_model_name=config['image_model'],
        language=config['language'],
        max_text_length=config['max_text_length'],
        device=device
    )
    print("✓ Preprocessor ready!")
    
    # Process existing train/dev/test splits
    print(f"\n📊 Processing dataset splits...")
    print("-" * 70)
    
    try:
        results = preprocessor.process_existing_splits(
            data_dir="./src/data/json",
            save_base_dir="./processed_data",
            save_format=config['save_format'],
            batch_size=config['batch_size'],
            splits=["train", "dev", "test"],
            file_prefix="news_data_vifactcheck_"
        )
        
        # Print results
        print("\n" + "=" * 70)
        print("✅ PREPROCESSING COMPLETE!")
        print("=" * 70)
        
        total_samples = 0
        for split_name, (dataset, metadata) in results.items():
            print(f"\n📁 {split_name.upper()} Split:")
            print(f"  • Samples: {metadata['num_samples']}")
            print(f"  • Text shape: {metadata['text_feature_shape']}")
            print(f"  • Image shape: {metadata['image_feature_shape']}")
            print(f"  • Classes: {metadata['num_classes']}")
            print(f"  • Saved to: {metadata['save_dir']}")
            total_samples += metadata['num_samples']
        
        print("\n" + "=" * 70)
        print(f"📊 Total samples processed: {total_samples}")
        print("=" * 70)
        
        print("\n🎯 Next Steps:")
        print("  1. Check the processed_data/ directory for output files")
        print("  2. Review metadata.json in each split folder")
        print("  3. Use the processed data to train your COOLANT model")
        
        return results
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Data files not found!")
        print(f"   {e}")
        print("\n💡 Make sure your JSON files are in: ./src/data/json/")
        print("   Expected files:")
        print("     - news_data_vifactcheck_train.json")
        print("     - news_data_vifactcheck_dev.json")
        print("     - news_data_vifactcheck_test.json")
        return None
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    
    if results:
        print("\n✨ Success! Your Vietnamese fake news data is ready for training!")
    else:
        print("\n⚠️  Preprocessing failed. Please check the errors above.")
