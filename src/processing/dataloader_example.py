#!/usr/bin/env python3
"""
Example usage of PyTorch Dataset and DataLoader integration for multimodal fake news detection.

This script demonstrates:
1. Creating PyTorch Datasets from your JSON/JPG data
2. Using DataLoader for efficient batch processing
3. Training models with the integrated dataset
4. Custom collate functions and data loading strategies
"""

import os
import sys
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from processing.pytorch_dataset import (
    FakeNewsDataset, 
    FakeNewsDataLoader,
    PreprocessedDataset,
    create_dataloaders,
    create_sample_dataloader,
    get_dataloader_info,
    preview_dataloader
)
from models import create_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_dataset():
    """Example: Create and use basic FakeNewsDataset."""
    print("=== Basic Dataset Example ===")
    
    # Define your data paths
    json_path = "/Users/haila/Library/CloudStorage/GoogleDrive-ladohaingan@gmail.com/My Drive/MyFile/fake-new-detection/news_data.json"
    image_base_dir = "/Users/haila/Library/CloudStorage/GoogleDrive-ladohaingan@gmail.com/My Drive/MyFile/fake-new-detection/src/data/jpg"
    
    try:
        # Create dataset
        dataset = FakeNewsDataset(
            json_path=json_path,
            image_base_dir=image_base_dir,
            max_length=30,
            embed_dim=200,
            feature_dim=512,
            cache_images=True,  # Cache processed images for speed
            cache_text=False    # Text processing is fast, no need to cache
        )
        
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Get dataset statistics
        stats = dataset.get_statistics()
        print(f"   Statistics: {stats}")
        
        # Get a single sample
        sample = dataset[0]
        print(f"   Sample text shape: {sample['text_features'].shape}")
        print(f"   Sample image shape: {sample['image_features'].shape}")
        print(f"   Sample label: {sample['label'].item()}")
        print(f"   Sample title: {sample['title'][:50]}...")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return None


def example_dataloader_usage():
    """Example: Using DataLoader for batch processing."""
    print("\n=== DataLoader Usage Example ===")
    
    # Create sample dataset for demonstration
    dataloader = create_sample_dataloader(batch_size=16, num_samples=100)
    
    print(f"✅ Sample DataLoader created")
    
    # Get dataloader information
    info = get_dataloader_info(dataloader)
    print(f"   DataLoader info: {info}")
    
    # Preview batches
    preview_dataloader(dataloader, num_batches=2)
    
    return dataloader


def example_custom_dataloader():
    """Example: Create custom DataLoader with specific settings."""
    print("\n=== Custom DataLoader Example ===")
    
    # Create sample dataset
    dataset = create_sample_dataloader(batch_size=32, num_samples=200).dataset
    
    # Create custom dataloader
    custom_loader = FakeNewsDataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"✅ Custom DataLoader created")
    print(f"   Batch size: {custom_loader.batch_size}")
    print(f"   Number of batches: {len(custom_loader)}")
    print(f"   Number of workers: {custom_loader.num_workers}")
    
    # Get a sample batch
    sample_batch = custom_loader.get_sample_batch()
    print(f"   Sample batch shapes:")
    print(f"     Text: {sample_batch['text_features'].shape}")
    print(f"     Image: {sample_batch['image_features'].shape}")
    print(f"     Labels: {sample_batch['labels'].shape}")
    
    return custom_loader


def example_train_val_test_split():
    """Example: Create train/val/test dataloaders from your data."""
    print("\n=== Train/Val/Test Split Example ===")
    
    # Define your data paths
    json_path = "/Users/haila/Library/CloudStorage/GoogleDrive-ladohaingan@gmail.com/My Drive/MyFile/fake-new-detection/news_data.json"
    image_base_dir = "/Users/haila/Library/CloudStorage/GoogleDrive-ladohaingan@gmail.com/My Drive/MyFile/fake-new-detection/src/data/jpg"
    
    try:
        # Create train/val/test dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            json_path=json_path,
            image_base_dir=image_base_dir,
            batch_size=16,
            test_size=0.2,
            val_size=0.1,
            num_workers=2,
            labels=None  # You can provide labels here
        )
        
        print(f"✅ Train/Val/Test dataloaders created")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Check first batch from each
        for name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
            batch = next(iter(loader))
            print(f"   {name} batch shapes: text={batch['text_features'].shape}, "
                  f"image={batch['image_features'].shape}, labels={batch['labels'].shape}")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"❌ Error creating dataloaders: {e}")
        return None, None, None


def example_training_with_dataloader():
    """Example: Training loop using DataLoader."""
    print("\n=== Training with DataLoader Example ===")
    
    # Create sample dataloader
    train_loader = create_sample_dataloader(batch_size=32, num_samples=500)
    val_loader = create_sample_dataloader(batch_size=32, num_samples=100)
    
    # Create model
    model = create_model('coolant', num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"✅ Starting training with DataLoader")
    print(f"   Model: COOLANT")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Training loop
    model.train()
    for epoch in range(3):
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 10:  # Limit for demo
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
            
            if batch_idx % 5 == 0:
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
                if batch_idx >= 5:  # Limit for demo
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
        
        avg_val_loss = val_loss / min(5, len(val_loader))
        val_accuracy = 100 * correct / total if total > 0 else 0
        print(f"   Validation: Loss = {avg_val_loss:.4f}, Accuracy = {val_accuracy:.2f}%")
        model.train()
    
    print(f"✅ Training completed!")


def example_preprocessed_dataset():
    """Example: Using pre-computed features for faster training."""
    print("\n=== Preprocessed Dataset Example ===")
    
    # Generate pre-computed features
    num_samples = 1000
    text_features = torch.randn(num_samples, 30, 200)
    image_features = torch.randn(num_samples, 512)
    labels = torch.randint(0, 2, (num_samples,))
    titles = [f"News article {i}" for i in range(num_samples)]
    
    # Create preprocessed dataset
    dataset = PreprocessedDataset(
        text_features=text_features,
        image_features=image_features,
        labels=labels,
        titles=titles
    )
    
    print(f"✅ Preprocessed dataset created with {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"   DataLoader batches: {len(dataloader)}")
    
    # Test with model
    model = create_model('clip', num_classes=2)
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:  # Test first 2 batches
                break
            
            outputs = model(batch['text_features'], batch['image_features'])
            print(f"   Batch {batch_idx+1}: Output shape = {outputs['multimodal_logits'].shape}")
    
    print(f"✅ Preprocessed dataset test completed!")


def example_custom_collate_function():
    """Example: Custom collate function for special batching needs."""
    print("\n=== Custom Collate Function Example ===")
    
    def custom_collate_fn(batch):
        """
        Custom collate function that handles variable-length sequences
        and adds additional metadata.
        """
        # Stack tensors
        text_features = torch.stack([item['text_features'] for item in batch])
        image_features = torch.stack([item['image_features'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        # Calculate text lengths (for potential use in models)
        text_lengths = torch.tensor([len(item['title'].split()) for item in batch])
        
        # Create batch dictionary
        return {
            'text_features': text_features,
            'image_features': image_features,
            'labels': labels,
            'titles': [item['title'] for item in batch],
            'image_paths': [item['image_path'] for item in batch],
            'text_lengths': text_lengths,
            'batch_size': len(batch)
        }
    
    # Create sample dataset
    dataset = create_sample_dataloader(batch_size=16, num_samples=50).dataset
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    print(f"✅ Custom collate function dataloader created")
    
    # Test custom batching
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 1:
            break
        
        print(f"   Batch {batch_idx+1}:")
        print(f"     Text features: {batch['text_features'].shape}")
        print(f"     Image features: {batch['image_features'].shape}")
        print(f"     Labels: {batch['labels'].shape}")
        print(f"     Text lengths: {batch['text_lengths']}")
        print(f"     Batch size: {batch['batch_size']}")
    
    print(f"✅ Custom collate function test completed!")


def example_memory_efficient_loading():
    """Example: Memory-efficient data loading for large datasets."""
    print("\n=== Memory Efficient Loading Example ===")
    
    # Create dataset with minimal caching
    dataset = FakeNewsDataset(
        json_path="/Users/haila/Library/CloudStorage/GoogleDrive-ladohaingan@gmail.com/My Drive/MyFile/fake-new-detection/news_data.json",
        image_base_dir="/Users/haila/Library/CloudStorage/GoogleDrive-ladohaingan@gmail.com/My Drive/MyFile/fake-new-detection/src/data/jpg",
        cache_images=False,  # Don't cache images to save memory
        cache_text=False     # Don't cache text
    )
    
    # Create dataloader with optimal settings for memory efficiency
    dataloader = DataLoader(
        dataset,
        batch_size=16,        # Smaller batch size
        shuffle=True,
        num_workers=2,        # Use workers for parallel loading
        pin_memory=True,      # Faster GPU transfer
        drop_last=True        # Drop incomplete batches for consistent size
    )
    
    print(f"✅ Memory-efficient dataloader created")
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Batch size: {dataloader.batch_size}")
    print(f"   Number of workers: {dataloader.num_workers}")
    print(f"   Pin memory: {dataloader.pin_memory}")
    
    # Test memory usage by processing a few batches
    model = create_model('senet', num_classes=2)
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:  # Process only first 3 batches
                break
            
            # For SENet, we need to reshape text features
            # SENet expects (B, 64, L) but we have (B, 200, L)
            text_input = batch['text_features'][:, :64, :].permute(0, 2, 1)  # (B, 64, L) -> (B, L, 64)
            outputs = model(text_input)
            
            print(f"   Batch {batch_idx+1}: Processed successfully, output shape = {outputs.shape}")
    
    print(f"✅ Memory efficient loading test completed!")


def main():
    """Run all DataLoader examples."""
    print("🚀 PyTorch Dataset and DataLoader Examples")
    print("=" * 60)
    
    try:
        # Example 1: Basic dataset usage
        example_basic_dataset()
        
        # Example 2: DataLoader usage
        example_dataloader_usage()
        
        # Example 3: Custom DataLoader
        example_custom_dataloader()
        
        # Example 4: Train/val/test split
        example_train_val_test_split()
        
        # Example 5: Training with DataLoader
        example_training_with_dataloader()
        
        # Example 6: Preprocessed dataset
        example_preprocessed_dataset()
        
        # Example 7: Custom collate function
        example_custom_collate_function()
        
        # Example 8: Memory efficient loading
        example_memory_efficient_loading()
        
        print("\n✅ All DataLoader examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
