#!/usr/bin/env python3
"""
Test Multimodal Vietnamese Preprocessing
Tests combined text and image preprocessing with Vietnamese dataset
"""

import sys
import os
import json
import numpy as np
import torch
from PIL import Image
import random

# Add src to path
sys.path.append('./src')

from preprocessing.combined_preprocessing import CombinedPreprocessor

def extract_texts_and_images_from_json(data_path: str):
    """Extract texts and image paths from Vietnamese JSON dataset"""
    
    print("📂 Loading Vietnamese dataset with image paths...")
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    texts = []
    labels = []
    image_paths = []
    
    # Base directory for images
    base_data_dir = "./src/data"
    
    for item in raw_data:
        # Extract text
        text = item.get('text', item.get('content', ''))
        if text:  # Only include items with text
            texts.append(text)
            
            # Extract label
            label = item.get('label', item.get('is_fake', 0))
            labels.append(label)
            
            # Extract image path
            images = item.get('images', [])
            if images and len(images) > 0:
                # Get the first image's folder_path
                folder_path = images[0].get('folder_path', '')
                if folder_path:
                    # Construct full path from base data directory
                    full_image_path = os.path.join(base_data_dir, folder_path)
                    image_paths.append(full_image_path)
                else:
                    # No folder_path, use None
                    image_paths.append(None)
            else:
                # No images, use None
                image_paths.append(None)
    
    print(f"✓ Extracted {len(texts)} texts")
    print(f"✓ Found {sum(1 for path in image_paths if path is not None)} images")
    print(f"✓ Missing images: {sum(1 for path in image_paths if path is None)}")
    
    return texts, labels, image_paths

def create_missing_images(image_paths: list):
    """Create placeholder images for missing image paths"""
    
    print("🖼️ Creating placeholder images for missing paths...")
    
    created_count = 0
    for i, image_path in enumerate(image_paths):
        if image_path is None or not os.path.exists(image_path):
            # Create a placeholder image
            placeholder_dir = "./placeholder_images"
            os.makedirs(placeholder_dir, exist_ok=True)
            
            placeholder_path = os.path.join(placeholder_dir, f"placeholder_{i}.jpg")
            
            if not os.path.exists(placeholder_path):
                # Create a simple placeholder image
                placeholder_array = np.random.randint(128, 200, (224, 224, 3), dtype=np.uint8)
                placeholder_image = Image.fromarray(placeholder_array)
                placeholder_image.save(placeholder_path)
                created_count += 1
            
            # Update the path
            image_paths[i] = placeholder_path
    
    print(f"✓ Created {created_count} placeholder images")
    return image_paths

def test_multimodal_preprocessing():
    """Test multimodal preprocessing with Vietnamese dataset"""
    
    print("🇻🇳 Testing Multimodal Vietnamese Preprocessing for COOLANT")
    print("=" * 70)
    
    # Dataset path
    data_path = "./src/data/json/news_data_vifactcheck_dev.json"
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return
    
    # Extract texts, labels, and image paths from JSON
    texts, labels, image_paths = extract_texts_and_images_from_json(data_path)
    
    print(f"✓ Labels distribution: Real: {sum(1 for l in labels if l == 0)}, Fake: {sum(1 for l in labels if l == 1)}")
    
    # Limit for testing (use first 50 samples)
    test_size = min(50, len(texts))
    test_texts = texts[:test_size]
    test_labels = labels[:test_size]
    test_image_paths = image_paths[:test_size]
    
    print(f"🧪 Testing with {test_size} samples")
    
    # Create placeholder images for missing paths
    test_image_paths = create_missing_images(test_image_paths.copy())
    
    # Verify image paths exist
    valid_images = [path for path in test_image_paths if path and os.path.exists(path)]
    print(f"✓ Valid image paths: {len(valid_images)}/{len(test_image_paths)}")
    
    # Initialize multimodal preprocessor for Vietnamese
    print("\n🔧 Initializing multimodal preprocessor...")
    
    # Proper device detection for MPS
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    combined_preprocessor = CombinedPreprocessor(
        text_model_name="vinai/phobert-base",
        image_model_name="resnet18",
        language="vi",
        max_text_length=256,  # Shorter for faster testing
        image_size=(224, 224),
        device=device
    )
    
    print(f"✓ Text model: {combined_preprocessor.text_preprocessor.model_name}")
    print(f"✓ Image model: {combined_preprocessor.image_preprocessor.model_name}")
    print(f"✓ Device: {combined_preprocessor.text_preprocessor.device}")
    
    # Test single sample preprocessing
    print("\n🎯 Testing single sample preprocessing...")
    
    # Find first valid sample with existing image
    valid_sample_idx = None
    for i, img_path in enumerate(test_image_paths):
        if img_path and os.path.exists(img_path):
            valid_sample_idx = i
            break
    
    if valid_sample_idx is not None:
        sample_text = test_texts[valid_sample_idx]
        sample_image = test_image_paths[valid_sample_idx]
        sample_label = test_labels[valid_sample_idx]
        
        print(f"Sample text: {sample_text[:100]}...")
        print(f"Sample image: {sample_image}")
        print(f"Sample label: {sample_label}")
        
        # Process single sample
        text_feat, image_feat = combined_preprocessor.preprocess_sample(
            sample_text, sample_image
        )
        
        print(f"✓ Text feature shape: {text_feat.shape}")
        print(f"✓ Image feature shape: {image_feat.shape}")
    else:
        print("⚠️ No valid images found, using placeholder images only")
    
    # Test full dataset preprocessing
    print(f"\n📊 Processing {test_size} samples...")
    
    # Create output directory
    output_dir = "./processed_data/vietnamese_multimodal_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process dataset
    dataset, metadata = combined_preprocessor.preprocess_dataset(
        test_texts, test_image_paths, test_labels,
        save_dir=output_dir,
        save_format="pkl",
        batch_size=16
    )
    
    print(f"✓ Created multimodal dataset with {len(dataset)} samples")
    print(f"✓ Dataset metadata: {metadata}")
    
    # Extract features from dataset for demonstration
    sample_text_feat, sample_img_feat, sample_label = dataset[0]
    print(f"✓ Sample text features shape: {sample_text_feat.shape}")
    print(f"✓ Sample image features shape: {sample_img_feat.shape}")
    print(f"✓ Sample label: {sample_label}")
    
    # Save the dataset manually for testing
    combined_preprocessor.save_combined_dataset(
        dataset.text_features.numpy(),
        dataset.image_features.numpy(), 
        dataset.labels.numpy(),
        f"{output_dir}/vietnamese_multimodal_test.pkl"
    )
    
    # Test dataset splitting
    print(f"\n✂️ Testing dataset splitting...")
    
    datasets = combined_preprocessor.create_data_splits(
        test_texts, test_image_paths, test_labels,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        save_dir=output_dir,
        save_format="pkl",
        random_seed=42
    )
    
    # Print split information
    for split_name, dataset in datasets.items():
        text_feat, img_feat, label = dataset[0]
        print(f"✓ {split_name.upper()}: {len(dataset)} samples")
        print(f"  Text features: {text_feat.shape}")
        print(f"  Image features: {img_feat.shape}")
        print(f"  Label: {label}")
    
    print(f"\n🎉 Multimodal Vietnamese preprocessing test completed!")
    print(f"\n📁 Files created in {output_dir}:")
    print(f"  - vietnamese_multimodal_test.pkl")
    for split_name in datasets.keys():
        print(f"  - {split_name}_combined_dataset.pkl")
    
    return dataset, datasets

def test_data_loading():
    """Test loading the processed multimodal data"""
    
    print("\n🔍 Testing data loading...")
    
    from preprocessing.combined_preprocessing import CombinedPreprocessor
    
    # Load processed data
    data_path = "./processed_data/vietnamese_multimodal_test/train_combined_dataset.pkl"
    
    if os.path.exists(data_path):
        text_features, image_features, labels = CombinedPreprocessor.load_combined_dataset(data_path)
        
        print(f"✓ Loaded text features: {text_features.shape}")
        print(f"✓ Loaded image features: {image_features.shape}")
        print(f"✓ Loaded labels: {labels.shape}")
        
        # Test PyTorch dataset creation
        from preprocessing.combined_preprocessing import MultimodalDataset
        dataset = MultimodalDataset(text_features, image_features, labels)
        
        print(f"✓ Created PyTorch dataset with {len(dataset)} samples")
        
        # Test data loading
        sample_text, sample_image, sample_label = dataset[0]
        print(f"✓ Sample data shapes: Text={sample_text.shape}, Image={sample_image.shape}, Label={sample_label.shape}")
        
        return True
    else:
        print(f"❌ Processed data not found: {data_path}")
        return False

def main():
    """Main function to run multimodal testing"""
    
    try:
        # Run multimodal preprocessing test
        dataset, datasets = test_multimodal_preprocessing()
        
        if dataset and datasets:
            # Test data loading
            test_data_loading()
            
            print("\n🚀 Your Vietnamese multimodal dataset is ready for COOLANT training!")
            print("📊 Dataset summary:")
            print(f"   Main dataset: {len(dataset)} samples")
            for split_name, split_dataset in datasets.items():
                print(f"   {split_name}: {len(split_dataset)} samples")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
