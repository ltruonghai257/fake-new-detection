# Models Module

This module provides state-of-the-art deep learning models for multimodal fake news detection, based on the COOLANT (Cross-modal Contrastive Learning for Multimodal Fake News Detection) framework.

## Overview

The models module includes implementations of:

- **COOLANT**: Cross-modal contrastive learning model with ambiguity learning and SE attention
- **CLIP**: Cross-modal contrastive learning model for text-image alignment  
- **SENet**: Squeeze-and-Excitation networks for attention-based feature learning

## Key Features

- 🔥 **State-of-the-art architectures** based on recent research
- 🧩 **Modular design** with reusable components
- ⚙️ **Flexible configuration** system with dataclasses
- 🏭 **Factory pattern** for easy model creation
- 💾 **Save/load functionality** for model persistence
- 📊 **Comprehensive examples** and documentation

## Quick Start

### Basic Usage

```python
from src.models import create_model

# Create COOLANT model with default configuration
model = create_model('coolant')

# Create CLIP model with custom parameters
clip_model = create_model('clip', output_dim=256, temperature=0.1)

# Create SENet model
senet_model = create_model('senet', filters=128, blocks=19, num_classes=2)
```

### Using Model Builder

```python
from src.models import ModelBuilder

model = (ModelBuilder()
         .model('coolant')
         .set_param('shared_dim', 256)
         .set_param('contrastive_weight', 2.0)
         .set_device('cuda')
         .build())
```

### Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Create model and data
model = create_model('coolant', num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for text, image, labels in dataloader:
    optimizer.zero_grad()
    
    outputs = model(text, image)
    loss = criterion(outputs['logits'], labels)
    
    loss.backward()
    optimizer.step()
```

## Model Architectures

### COOLANT (Recommended)

The main model implementing cross-modal contrastive learning with:

- **Encoding Module**: Shared text and image encoders
- **Similarity Module**: Cross-modal similarity computation
- **Ambiguity Learning**: Variational inference for handling uncertainty
- **SE Attention**: Squeeze-and-Excitation attention mechanism
- **Detection Module**: Final classification with multimodal fusion

**Key Features:**
- Cross-modal contrastive learning
- Ambiguity-aware learning with variational inference
- Attention-based multimodal fusion
- Robust to modality misalignment

**Configuration:**
```python
config = {
    'shared_dim': 128,      # Shared representation dimension
    'sim_dim': 64,          # Similarity space dimension  
    'feature_dim': 96,      # Final feature dimension
    'contrastive_weight': 1.0,
    'classification_weight': 1.0,
    'similarity_weight': 0.5,
    'temperature': 0.07
}
```

### CLIP

Cross-modal contrastive learning model with:

- **Text Encoder**: CNN or Transformer-based text encoding
- **Image Encoder**: MLP-based image encoding
- **Contrastive Learning**: Cross-modal alignment objective
- **Classification Heads**: Separate heads for text, image, and multimodal classification

**Key Features:**
- Flexible text encoding (CNN or Transformer)
- Contrastive learning for cross-modal alignment
- Multiple classification objectives
- Learnable temperature parameter

**Configuration:**
```python
config = {
    'output_dim': 512,
    'temperature': 0.07,
    'text_encoder': {
        'encoder_type': 'cnn',  # or 'transformer'
        'cnn_channel': 32,
        'cnn_kernel_size': (1, 2, 4, 8)
    },
    'image_encoder': {
        'hidden_dim': 256,
        'dropout': 0.1
    }
}
```

### SENet

Squeeze-and-Excitation networks for attention-based feature learning:

- **SE Blocks**: Channel-wise attention mechanism
- **Residual Connections**: Skip connections for gradient flow
- **Multi-scale Processing**: Different kernel sizes for feature extraction

**Key Features:**
- Channel-wise attention mechanism
- Residual learning
- Adaptive feature recalibration
- Efficient parameter usage

**Configuration:**
```python
config = {
    'in_channel': 64,
    'filters': 128,
    'blocks': 19,
    'reduction': 16,
    'num_classes': 2
}
```

## Configuration System

The module uses a comprehensive configuration system with dataclasses:

### Model Configurations

- `CLIPConfig`: Configuration for CLIP model
- `COOLANTConfig`: Configuration for COOLANT model  
- `SENetConfig`: Configuration for SENet model

### Training Configuration

```python
from src.models import TrainingConfig

training_config = TrainingConfig(
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=100,
    weight_decay=1e-5,
    use_amp=True,  # Mixed precision training
    early_stopping_patience=10
)
```

### Experiment Configuration

```python
from src.models import get_experiment_config

config = get_experiment_config('coolant')
config.experiment_name = "my_experiment"
config.model_config.shared_dim = 256
config.training_config.batch_size = 64

# Save configuration
config.save("experiment_config.json")

# Load configuration
config = ExperimentConfig.load("experiment_config.json")
```

## Advanced Usage

### Custom Model Registration

```python
from src.models import ModelFactory

class MyCustomModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Custom implementation
    
    def forward(self, x):
        # Custom forward pass
        pass

# Register custom model
ModelFactory.register_model('custom', MyCustomModel, MyCustomConfig)

# Use custom model
model = create_model('custom', custom_param=value)
```

### Model Comparison

```python
from src.models import create_baseline_models, get_model_summary

# Create all baseline models
models = create_baseline_models()

# Get model information
model_info = get_model_summary()
for name, info in model_info.items():
    print(f"{name}: {info['total_parameters']} parameters")
```

### Saving and Loading

```python
# Save model
model.save_pretrained("./checkpoints/my_model")

# Load model
from src.models import load_model
model = load_model("./checkpoints/my_model/model.pt")
```

## Input/Output Specifications

### COOLANT Model

**Inputs:**
- `text_raw`: Raw text features `(batch_size, seq_len, embed_dim)`
- `image_raw`: Raw image features `(batch_size, image_dim)`
- `text_aligned`: Pre-aligned text features (optional)
- `image_aligned`: Pre-aligned image features (optional)

**Outputs:**
- `logits`: Classification logits `(batch_size, num_classes)`
- `attention_weights`: SE attention weights `(batch_size, 3)`
- `ambiguity_weights`: Ambiguity learning weights `(batch_size, 3)`
- `similarity_pred`: Similarity predictions (optional)

### CLIP Model

**Inputs:**
- `text`: Text input `(batch_size, seq_len, embed_dim)` or `(batch_size, seq_len)`
- `image`: Image features `(batch_size, image_dim)`

**Outputs:**
- `text_logits`: Text classification logits
- `image_logits`: Image classification logits  
- `multimodal_logits`: Multimodal classification logits
- `contrastive_logits`: Contrastive learning logits
- `text_features`: Encoded text features (optional)
- `image_features`: Encoded image features (optional)

## Loss Functions

### Contrastive Loss

```python
# Compute contrastive loss
contrastive_loss = model.compute_contrastive_loss(text_features, image_features)
```

### Classification Loss

```python
# Compute classification loss
classification_loss = model.compute_classification_loss(text, image, labels)
```

### Total Loss (COOLANT)

```python
# Compute total loss with all components
losses = model.compute_total_loss(text_raw, image_raw, labels, similarity_labels)
total_loss = losses['total_loss']
```

## Best Practices

1. **Model Selection**: Use COOLANT for best performance, CLIP for simplicity, SENet for attention analysis
2. **Configuration**: Start with default configs and tune based on your dataset
3. **Training**: Use mixed precision training (`use_amp=True`) for faster training
4. **Evaluation**: Evaluate on multiple metrics (accuracy, F1, AUC)
5. **Checkpointing**: Save models regularly during training
6. **Hyperparameter Tuning**: Focus on learning rate, batch size, and loss weights

## Examples

See `examples.py` for comprehensive usage examples including:

- Basic model creation
- Training loops
- Multimodal feature extraction
- Contrastive learning
- Model comparison
- Configuration management

## Performance Tips

1. **Batch Size**: Use larger batch sizes for contrastive learning (32-128)
2. **Learning Rate**: Start with 1e-4 and adjust based on convergence
3. **Mixed Precision**: Enable AMP for faster training on modern GPUs
4. **Gradient Clipping**: Use gradient clipping to prevent exploding gradients
5. **Warmup**: Use learning rate warmup for stable training

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **NaN Loss**: Check learning rate, use gradient clipping
3. **Poor Convergence**: Adjust loss weights, learning rate schedule
4. **Model Loading Error**: Ensure model name matches saved configuration

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model info
from src.models import get_model_info
info = get_model_info('coolant')
print(info)
```

## Contributing

To add new models:

1. Inherit from `BaseModel` or `MultimodalModel`
2. Implement required methods (`forward`, `encode_text`, `encode_image`, etc.)
3. Create corresponding configuration class
4. Register with `ModelFactory`
5. Add tests and documentation

## References

- [COOLANT Paper](https://github.com/wishever/COOLANT) - Cross-modal Contrastive Learning for Multimodal Fake News Detection
- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models From Natural Language Supervision  
- [SENet Paper](https://arxiv.org/abs/1709.01507) - Squeeze-and-Excitation Networks
