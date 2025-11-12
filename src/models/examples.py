"""
Example usage of the models module for fake news detection.

This script demonstrates how to:
1. Create different models using the factory
2. Configure models with custom parameters
3. Train and evaluate models
4. Save and load model checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from . import (
    create_model, 
    load_model,
    ModelBuilder,
    get_model_summary,
    create_baseline_models,
    get_experiment_config,
    ExperimentConfig
)


def create_dummy_data(batch_size=32, num_samples=1000):
    """Create dummy data for testing."""
    # Text features (batch_size, seq_len, embed_dim)
    text_data = torch.randn(num_samples, 30, 200)
    
    # Image features (batch_size, feature_dim) 
    image_data = torch.randn(num_samples, 512)
    
    # Labels (0: real, 1: fake)
    labels = torch.randint(0, 2, (num_samples,))
    
    dataset = TensorDataset(text_data, image_data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def example_basic_model_creation():
    """Example: Basic model creation."""
    print("=== Basic Model Creation ===")
    
    # Create COOLANT model with default configuration
    coolant_model = create_model('coolant')
    print(f"Created COOLANT model: {coolant_model.__class__.__name__}")
    
    # Create CLIP model with custom configuration
    clip_model = create_model('clip', output_dim=256, temperature=0.1)
    print(f"Created CLIP model: {clip_model.__class__.__name__}")
    
    # Create SENet model
    senet_model = create_model('senet', filters=64, blocks=10, num_classes=2)
    print(f"Created SENet model: {senet_model.__class__.__name__}")
    
    print()


def example_model_builder():
    """Example: Using ModelBuilder for fluent interface."""
    print("=== Model Builder Example ===")
    
    # Build COOLANT model with custom configuration
    model = (ModelBuilder()
             .model('coolant')
             .set_param('shared_dim', 256)
             .set_param('sim_dim', 128)
             .set_param('contrastive_weight', 2.0)
             .set_device('cpu')
             .build())
    
    print(f"Built model: {model.__class__.__name__}")
    print(f"Model config: {model.get_config()}")
    print()


def example_model_training():
    """Example: Basic model training loop."""
    print("=== Model Training Example ===")
    
    # Create model and data
    model = create_model('coolant', num_classes=2)
    dataloader = create_dummy_data(batch_size=16, num_samples=100)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0
    
    print("Training for 3 batches...")
    for i, (text, image, labels) in enumerate(dataloader):
        if i >= 3:  # Only train for 3 batches for demo
            break
            
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(text, image)
        loss = criterion(outputs['logits'], labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Batch {i+1}, Loss: {loss.item():.4f}")
    
    print(f"Average Loss: {total_loss/3:.4f}")
    print()


def example_multimodal_features():
    """Example: Working with multimodal features."""
    print("=== Multimodal Features Example ===")
    
    model = create_model('coolant')
    
    # Create sample data
    text_features = torch.randn(8, 30, 200)  # (batch, seq_len, embed_dim)
    image_features = torch.randn(8, 512)     # (batch, feature_dim)
    
    model.eval()
    with torch.no_grad():
        # Get model outputs
        outputs = model(text_features, image_features, return_all=True)
        
        print("Model outputs:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
    
    print()


def example_contrastive_learning():
    """Example: Contrastive learning with CLIP."""
    print("=== Contrastive Learning Example ===")
    
    clip_model = create_model('clip', output_dim=128)
    
    # Sample data
    text_data = torch.randn(16, 30, 200)
    image_data = torch.randn(16, 512)
    
    clip_model.eval()
    with torch.no_grad():
        # Get features and similarity scores
        outputs = clip_model(text_data, image_data, return_features=True)
        
        print("CLIP outputs:")
        print(f"  Text features: {outputs['text_features'].shape}")
        print(f"  Image features: {outputs['image_features'].shape}")
        print(f"  Contrastive logits: {outputs['contrastive_logits'].shape}")
        print(f"  Logit scale: {outputs['logit_scale'].item():.4f}")
        
        # Compute similarity scores
        similarity_scores = clip_model.get_similarity_scores(text_data, image_data)
        print(f"  Similarity scores: {similarity_scores.shape}")
        print(f"  Sample similarities: {similarity_scores[:5].tolist()}")
    
    print()


def example_model_saving_loading():
    """Example: Saving and loading models."""
    print("=== Model Saving/Loading Example ===")
    
    # Create and configure model
    original_model = create_model('coolant', shared_dim=256, sim_dim=128)
    
    # Save model
    save_path = "/tmp/test_model.pt"
    original_model.save_pretrained("/tmp")
    print(f"Saved model to {save_path}")
    
    # Load model
    try:
        loaded_model = load_model("/tmp/model.pt")
        print("Successfully loaded model")
        print(f"Original config: {original_model.get_config()}")
        print(f"Loaded config: {loaded_model.get_config()}")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    print()


def example_model_comparison():
    """Example: Comparing different models."""
    print("=== Model Comparison ===")
    
    models = create_baseline_models()
    sample_text = torch.randn(4, 30, 200)
    sample_image = torch.randn(4, 512)
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            try:
                if name == 'senet':
                    # SENet expects different input format
                    sample_input = torch.randn(4, 64, 30)
                    outputs = model(sample_input)
                    print(f"{name.upper()}: Output shape {outputs.shape}")
                else:
                    outputs = model(sample_text, sample_image)
                    if isinstance(outputs, dict):
                        print(f"{name.upper()}: Logits shape {outputs['logits'].shape}")
                    else:
                        print(f"{name.upper()}: Output shape {outputs.shape}")
            except Exception as e:
                print(f"{name.upper()}: Error - {e}")
    
    print()


def example_experiment_config():
    """Example: Using experiment configuration."""
    print("=== Experiment Configuration ===")
    
    # Create experiment configuration
    config = get_experiment_config('coolant')
    config.experiment_name = "my_fake_news_experiment"
    config.model_config.shared_dim = 256
    config.training_config.batch_size = 64
    config.training_config.learning_rate = 2e-4
    
    print("Experiment configuration:")
    print(f"  Name: {config.experiment_name}")
    print(f"  Model: {config.model_config.model_name}")
    print(f"  Shared dim: {config.model_config.shared_dim}")
    print(f"  Batch size: {config.training_config.batch_size}")
    print(f"  Learning rate: {config.training_config.learning_rate}")
    
    # Save configuration
    config.save("/tmp/experiment_config.json")
    print("Saved experiment configuration")
    
    # Load configuration
    loaded_config = ExperimentConfig.load("/tmp/experiment_config.json")
    print(f"Loaded config name: {loaded_config.experiment_name}")
    
    print()


def main():
    """Run all examples."""
    print("🚀 Models Module Examples")
    print("=" * 50)
    
    # Print available models
    print("Available models:")
    model_summary = get_model_summary()
    for name, info in model_summary.items():
        if 'error' not in info:
            print(f"  - {name}: {info.get('total_parameters', 'Unknown')} parameters")
        else:
            print(f"  - {name}: Error - {info['error']}")
    print()
    
    # Run examples
    try:
        example_basic_model_creation()
        example_model_builder()
        example_multimodal_features()
        example_contrastive_learning()
        example_model_training()
        example_model_saving_loading()
        example_model_comparison()
        example_experiment_config()
        
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
