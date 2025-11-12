#!/usr/bin/env python3
"""
Example usage of the multimodal data preprocessing module.

This script shows how to:
1. Load and preprocess your JSON/JPG data
2. Create train/val/test splits
3. Use the processed data with the models
4. Create sample data for testing
"""

import os
import sys
import torch
import logging

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from processing.multimodal_processor import (
    DataPreprocessor, 
    MultimodalDataset,
    load_processed_data,
    create_dataloader
)
from models import create_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_preprocess_real_data():
    """Example: Preprocess your actual JSON/JPG data."""
    print("=== Preprocessing Real Data Example ===")
    
    # Define your data paths
    json_path = "/Users/haila/Library/CloudStorage/GoogleDrive-ladohaingan@gmail.com/My Drive/MyFile/fake-new-detection/news_data.json"
    image_base_dir = "/Users/haila/Library/CloudStorage/GoogleDrive-ladohaingan@gmail.com/My Drive/MyFile/fake-new-detection/src/data/jpg"
    output_dir = "./processed_data"
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        json_path=json_path,
        image_base_dir=image_base_dir,
        output_dir=output_dir,
        max_length=30,
        embed_dim=200,
        feature_dim=512
    )
    
    try:
        # Preprocess the dataset
        metadata = preprocessor.preprocess_dataset(
            labels=None,  # You can provide labels here: [0, 1, 0, 1, ...]
            test_size=0.2,
            val_size=0.1
        )
        
        print(f"✅ Preprocessing completed!")
        print(f"   Total samples: {metadata['total_samples']}")
        print(f"   Train: {metadata['train_size']}")
        print(f"   Val: {metadata['val_size']}")
        print(f"   Test: {metadata['test_size']}")
        print(f"   Text shape: {metadata['text_shape']}")
        print(f"   Image shape: {metadata['image_shape']}")
        
        return output_dir
        
    except Exception as e:
        print(f"❌ Error preprocessing data: {e}")
        return None


def example_load_and_use_data():
    """Example: Load processed data and use with models."""
    print("\n=== Loading and Using Processed Data ===")
    
    # Load processed data
    data_dir = "./processed_data"
    
    if not os.path.exists(data_dir):
        print("❌ Processed data not found. Run preprocessing first.")
        return
    
    # Load train split
    train_data = load_processed_data(data_dir, split='train')
    val_data = load_processed_data(data_dir, split='val')
    
    print(f"✅ Loaded data:")
    print(f"   Train: {train_data['text_features'].shape}")
    print(f"   Val: {val_data['text_features'].shape}")
    
    # Create data loaders
    train_loader = create_dataloader(train_data, batch_size=16, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size=16, shuffle=False)
    
    print(f"✅ Created data loaders:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Test with model
    print("\n--- Testing with COOLANT model ---")
    model = create_model('coolant', num_classes=2)
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (text_batch, image_batch, label_batch) in enumerate(train_loader):
            if batch_idx >= 2:  # Only test first 2 batches
                break
                
            outputs = model(text_batch, image_batch)
            
            print(f"Batch {batch_idx + 1}:")
            print(f"   Input shapes: text={text_batch.shape}, image={image_batch.shape}")
            print(f"   Output logits: {outputs['logits'].shape}")
            print(f"   Attention weights: {outputs['attention_weights'].shape}")
            print(f"   Sample prediction: {torch.softmax(outputs['logits'][0], dim=-1)}")
    
    print("✅ Model testing completed!")


def example_create_sample_data():
    """Example: Create sample data for quick testing."""
    print("\n=== Creating Sample Data ===")
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        json_path="dummy.json",  # Not used for sample data
        image_base_dir="dummy",  # Not used for sample data
        output_dir="./sample_data",
        max_length=30,
        embed_dim=200,
        feature_dim=512
    )
    
    # Create sample dataset
    sample_data = preprocessor.create_sample_data(num_samples=1000)
    
    print(f"✅ Created sample data:")
    print(f"   Text features: {sample_data['text_features'].shape}")
    print(f"   Image features: {sample_data['image_features'].shape}")
    print(f"   Labels: {sample_data['labels'].shape}")
    
    # Test with all models
    print("\n--- Testing sample data with all models ---")
    
    models_to_test = ['coolant', 'clip', 'senet']
    
    for model_name in models_to_test:
        try:
            print(f"\nTesting {model_name.upper()}:")
            
            if model_name == 'senet':
                # SENet needs different input format
                model = create_model(model_name, num_classes=2)
                # Reshape text features for SENet (B, C, L)
                senet_input = sample_data['text_features'].permute(0, 2, 1)
                with torch.no_grad():
                    outputs = model(senet_input)
                print(f"   Output shape: {outputs.shape}")
            else:
                model = create_model(model_name, num_classes=2)
                with torch.no_grad():
                    outputs = model(sample_data['text_features'], sample_data['image_features'])
                
                if isinstance(outputs, dict):
                    print(f"   Logits shape: {outputs['logits'].shape}")
                else:
                    print(f"   Output shape: {outputs.shape}")
            
            print(f"   ✅ {model_name} works correctly!")
            
        except Exception as e:
            print(f"   ❌ {model_name} failed: {e}")


def example_individual_processing():
    """Example: Process individual text and image samples."""
    print("\n=== Individual Processing Example ===")
    
    from processing.multimodal_processor import TextProcessor, ImageProcessor
    
    # Text processing
    text_processor = TextProcessor(max_length=30, embed_dim=200)
    
    sample_text = """
    Chính phủ nước cộng hòa xã hội chủ nghĩa việt nam Báo Điện tử Chính phủ 
    Phó Thủ tướng Trần Hồng Hà: Các tác phẩm truyền hình đã vun đắp, làm giàu 
    cho nền văn hóa Việt Nam tiên tiến, đậm đà bản sắc dân tộc.
    """
    
    text_features = text_processor.preprocess_text(sample_text)
    print(f"✅ Processed text: {text_features.shape}")
    
    # Image processing
    image_processor = ImageProcessor(feature_dim=512)
    
    # Try to find a sample image
    image_dir = "/Users/haila/Library/CloudStorage/GoogleDrive-ladohaingan@gmail.com/My Drive/MyFile/fake-new-detection/src/data/jpg/thanh_nien"
    if os.path.exists(image_dir):
        sample_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
        if sample_images:
            sample_image_path = os.path.join(image_dir, sample_images[0])
            image_features = image_processor.preprocess_image(sample_image_path)
            print(f"✅ Processed image: {image_features.shape}")
            print(f"   Image file: {sample_images[0]}")
        else:
            print("❌ No images found in directory")
    else:
        print("❌ Image directory not found")


def example_training_loop():
    """Example: Simple training loop with processed data."""
    print("\n=== Training Loop Example ===")
    
    # Load or create sample data
    data_dir = "./sample_data"
    if not os.path.exists(data_dir):
        print("Creating sample data first...")
        example_create_sample_data()
    
    # Load data
    train_data = load_processed_data(data_dir, 'sample')
    train_loader = create_dataloader(train_data, batch_size=32, shuffle=True)
    
    # Create model and optimizer
    model = create_model('coolant', num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Starting training...")
    model.train()
    
    for epoch in range(3):  # Train for 3 epochs
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (text_batch, image_batch, label_batch) in enumerate(train_loader):
            if batch_idx >= 10:  # Limit to 10 batches for demo
                break
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(text_batch, image_batch)
            loss = criterion(outputs['logits'], label_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")
    
    print("✅ Training completed!")
    
    # Test prediction
    model.eval()
    with torch.no_grad():
        sample_text = torch.randn(1, 30, 200)
        sample_image = torch.randn(1, 512)
        outputs = model(sample_text, sample_image)
        prediction = torch.softmax(outputs['logits'], dim=-1)
        
        print(f"Sample prediction: {prediction}")
        print(f"Predicted class: {prediction.argmax(dim=-1).item()}")


def main():
    """Run all examples."""
    print("🚀 Multimodal Data Preprocessing Examples")
    print("=" * 50)
    
    try:
        # Example 1: Individual processing
        example_individual_processing()
        
        # Example 2: Create sample data
        example_create_sample_data()
        
        # Example 3: Load and use data
        example_load_and_use_data()
        
        # Example 4: Training loop
        example_training_loop()
        
        # Example 5: Preprocess real data (commented out by default)
        print("\n=== Real Data Preprocessing ===")
        print("To preprocess your actual data, uncomment the line below:")
        print("# example_preprocess_real_data()")
        
        # Uncomment to preprocess your real data
        # example_preprocess_real_data()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
