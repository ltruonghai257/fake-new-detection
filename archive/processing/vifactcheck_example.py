#!/usr/bin/env python3
"""
Example usage of ViFactCheck data preprocessing and DataLoader integration.

This script demonstrates how to use the ViFactCheck processor with the format:
- images[i].caption contains the text
- images[i].folder_path contains the image path
"""

import os
import sys
import torch
import torch.nn as nn
import logging

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from processing.vifactcheck_processor import (
    ViFactCheckDataset, 
    create_vifactcheck_dataloaders,
    create_sample_vifactcheck_dataloader
)
from models import create_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_vifactcheck_dataset():
    """Example: Create and use ViFactCheck dataset."""
    print("=== ViFactCheck Dataset Example ===")
    
    # Define paths to your ViFactCheck data
    json_path = "/path/to/your/news_data_vifactcheck_train.json"
    image_base_dir = "/path/to/your/images/folder"
    
    # Since we can't access the actual files, let's create a sample dataset
    print("Creating sample ViFactCheck dataset for demonstration...")
    
    # Create sample data in ViFactCheck format
    sample_data = {
        "images": [
            {
                "caption": "Đây là một bài tin giả về tình hình chính trị. Nội dung không chính xác và gây hoang mang dư luận.",
                "folder_path": "fake_news/image_001.jpg"
            },
            {
                "caption": "Chính phủ công bố các biện pháp hỗ trợ người dân bị ảnh hưởng bởi thiên tai. Các gói cứu trợ sẽ được triển khai trong tuần tới.",
                "folder_path": "real_news/image_002.jpg"
            },
            {
                "caption": "Thông tin về virus mới không có cơ sở khoa học, các chuyên gia y tế khẳng định đây là tin đồn vô căn cứ.",
                "folder_path": "fake_news/image_003.jpg"
            }
        ]
    }
    
    # Save sample data to temporary file
    import json
    temp_json_path = "./temp_vifactcheck_sample.json"
    with open(temp_json_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    try:
        # Create dataset
        dataset = ViFactCheckDataset(
            json_path=temp_json_path,
            image_base_dir=image_base_dir,
            max_length=30,
            embed_dim=200,
            feature_dim=512,
            labels=[1, 0, 1],  # 1=fake, 0=real
            cache_images=True,
            cache_text=False
        )
        
        print(f"✅ ViFactCheck dataset created with {len(dataset)} samples")
        
        # Get dataset statistics
        stats = dataset.get_statistics()
        print(f"   Statistics: {stats}")
        
        # Get a single sample
        sample = dataset[0]
        print(f"   Sample text shape: {sample['text_features'].shape}")
        print(f"   Sample image shape: {sample['image_features'].shape}")
        print(f"   Sample label: {sample['label'].item()}")
        print(f"   Sample caption: {sample['caption'][:100]}...")
        print(f"   Sample image path: {sample['image_path']}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error creating ViFactCheck dataset: {e}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)


def example_vifactcheck_dataloaders():
    """Example: Create train/val/test dataloaders for ViFactCheck data."""
    print("\n=== ViFactCheck DataLoaders Example ===")
    
    # Create sample ViFactCheck dataloader
    dataloader = create_sample_vifactcheck_dataloader(batch_size=16, num_samples=100)
    
    print(f"✅ Sample ViFactCheck DataLoader created")
    print(f"   Number of batches: {len(dataloader)}")
    print(f"   Batch size: {dataloader.batch_size}")
    
    # Test a few batches
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:  # Only test first 2 batches
            break
        
        print(f"   Batch {batch_idx + 1}:")
        print(f"     Text features: {batch['text_features'].shape}")
        print(f"     Image features: {batch['image_features'].shape}")
        print(f"     Labels: {batch['labels'].shape}")
        print(f"     Sample captions: {batch['captions'][:2]}")
        print(f"     Label distribution: {torch.bincount(batch['labels'])}")
    
    return dataloader


def example_vifactcheck_training():
    """Example: Training with ViFactCheck DataLoader."""
    print("\n=== ViFactCheck Training Example ===")
    
    # Create sample dataloaders
    train_loader = create_sample_vifactcheck_dataloader(batch_size=32, num_samples=500)
    val_loader = create_sample_vifactcheck_dataloader(batch_size=32, num_samples=100)
    
    # Create model
    model = create_model('coolant', num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"✅ Starting ViFactCheck training")
    print(f"   Model: COOLANT")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Training loop
    model.train()
    for epoch in range(2):  # Train for 2 epochs
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 5:  # Limit for demo
                break
            
            # Extract data from batch
            text_features = batch['text_features']
            image_features = batch['image_features']
            labels = batch['labels']
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(text_features, image_features)
            loss = criterion(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 2 == 0:
                print(f"   Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {loss.item():.4f}")
        
        avg_train_loss = train_loss / num_batches
        print(f"   Epoch {epoch+1} completed: Avg Loss = {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 3:  # Limit for demo
                    break
                
                text_features = batch['text_features']
                image_features = batch['image_features']
                labels = batch['labels']
                
                outputs = model(text_features, image_features)
                loss = criterion(outputs['logits'], labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs['logits'], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / min(3, len(val_loader))
        val_accuracy = 100 * correct / total if total > 0 else 0
        print(f"   Validation: Loss = {avg_val_loss:.4f}, Accuracy = {val_accuracy:.2f}%")
        model.train()
    
    print(f"✅ ViFactCheck training completed!")


def main():
    """Run all ViFactCheck examples."""
    print("🚀 ViFactCheck Dataset and DataLoader Examples")
    print("=" * 60)
    
    try:
        # Example 1: Basic dataset usage
        example_vifactcheck_dataset()
        
        # Example 2: DataLoader usage
        example_vifactcheck_dataloaders()
        
        # Example 3: Training with ViFactCheck data
        example_vifactcheck_training()
        
        print("\n✅ All ViFactCheck examples completed successfully!")
        print("\n📋 How to use with your actual data:")
        print("1. Update the paths in example_vifactcheck_dataset():")
        print("   json_path = '/path/to/your/news_data_vifactcheck_train.json'")
        print("   image_base_dir = '/path/to/your/images/folder'")
        print("2. Run: python vifactcheck_example.py")
        print("3. The processor will handle the format:")
        print("   - images[i].caption → text processing")
        print("   - images[i].folder_path → image processing")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
