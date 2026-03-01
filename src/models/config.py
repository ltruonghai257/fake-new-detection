"""
Configuration classes for multimodal fake news detection models.

This module provides configuration classes for different model architectures
and training setups, ensuring consistent parameter management across the codebase.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import json


@dataclass
class BaseModelConfig:
    """Base configuration for all models."""
    
    # Model architecture
    num_classes: int = 2
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    
    # Device and optimization
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    
    # Regularization
    gradient_clip_norm: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class TextEncoderConfig(BaseModelConfig):
    """Configuration for text encoder models."""
    
    # Text processing
    vocab_size: int = 30000
    max_seq_length: int = 512
    embed_dim: int = 768
    
    # Model architecture
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    
    # Text-specific
    use_positional_encoding: bool = True
    use_attention: bool = True


@dataclass
class ImageEncoderConfig(BaseModelConfig):
    """Configuration for image encoder models."""
    
    # Image processing
    input_channels: int = 3
    image_size: int = 224
    
    # Model architecture
    backbone: str = "resnet50"  # resnet18, resnet34, resnet50, resnet101
    pretrained: bool = True
    feature_dim: int = 2048  # Output dimension of backbone
    
    # Custom layers
    use_attention: bool = True
    attention_dim: int = 512


@dataclass
class CLIPConfig(BaseModelConfig):
    """Configuration for CLIP model."""
    
    # Input dimensions
    text_input_dim: int = 768
    image_input_dim: int = 2048
    embed_dim: int = 512
    
    # Model architecture
    text_hidden_dim: int = 1024
    image_hidden_dim: int = 1024
    
    # Contrastive learning
    temperature: float = 0.07
    max_seq_len: int = 512
    
    # Loss weights
    contrastive_weight: float = 1.0
    
    # Training
    use_momentum: bool = False
    momentum: float = 0.999


@dataclass
class COOLANTConfig(BaseModelConfig):
    """Configuration for COOLANT model."""
    
    # Input dimensions
    text_input_dim: int = 768
    image_input_dim: int = 2048  # ResNet50 features
    
    # Encoding part
    shared_dim: int = 128
    shared_text_dim: int = 128
    shared_image_dim: int = 128
    
    # Similarity module
    sim_dim: int = 64
    
    # Detection module
    feature_dim: int = 96
    h_dim: int = 64
    num_classes: int = 2
    
    # Loss weights
    similarity_weight: float = 0.5
    classification_weight: float = 1.0
    contrastive_weight: float = 1.0
    clip_weight: float = 0.2
    
    # Training parameters
    temperature: float = 0.07
    margin: float = 0.2
    
    # Architecture options
    use_ambiguity_learning: bool = True
    use_se_attention: bool = True
    use_cross_modal_correlation: bool = True
    
    # CNN text encoder (FastCNN)
    cnn_channel: int = 32
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    text_embed_dim: int = 200  # Expected input for FastCNN


@dataclass
class SENetConfig(BaseModelConfig):
    """Configuration for SENet model."""
    
    # Architecture
    filters: int = 128
    blocks: int = 19
    reduction: int = 16
    
    # Input dimensions
    text_input_dim: int = 768
    image_input_dim: int = 2048
    
    # SE attention
    use_se: bool = True
    se_reduction: int = 16
    
    # Fusion
    fusion_dim: int = 512
    fusion_type: str = "concat"  # concat, add, multiply


@dataclass
class TrainingConfig(BaseModelConfig):
    """Configuration for training process."""
    
    # Training schedule
    num_epochs: int = 20
    warmup_epochs: int = 2
    
    # Learning rate scheduling
    scheduler: str = "plateau"  # plateau, cosine, step, exponential
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 7
    min_delta: float = 1e-4
    
    # Validation
    validation_split: float = 0.1
    test_split: float = 0.1
    
    # Logging
    log_interval: int = 10
    save_best_only: bool = True
    save_interval: int = 5
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None


@dataclass
class DataConfig(BaseModelConfig):
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    data_dir: str = "./data"
    train_file: str = "train.json"
    val_file: str = "val.json"
    test_file: str = "test.json"
    
    # Preprocessing
    text_max_length: int = 512
    image_size: int = 224
    
    # Augmentation
    use_text_augmentation: bool = True
    use_image_augmentation: bool = True
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Batch processing
    batch_size: int = 32
    gradient_accumulation_steps: int = 1


@dataclass
class ExperimentConfig(BaseModelConfig):
    """Complete experiment configuration."""
    
    # Experiment metadata
    name: str = "multimodal_fake_news_detection"
    version: str = "1.0"
    description: str = "Multimodal fake news detection experiment"
    
    # Model configuration
    model_type: str = "coolant"  # clip, coolant, senet
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Data configuration
    data: DataConfig = field(default_factory=DataConfig)
    
    # Logging and tracking
    use_wandb: bool = False
    wandb_project: str = "fake-news-detection"
    log_dir: str = "./logs"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


# Configuration factory functions
def get_clip_config(**kwargs) -> CLIPConfig:
    """Get default CLIP configuration with optional overrides."""
    config = CLIPConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_coolant_config(**kwargs) -> COOLANTConfig:
    """Get default COOLANT configuration with optional overrides."""
    config = COOLANTConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_senet_config(**kwargs) -> SENetConfig:
    """Get default SENet configuration with optional overrides."""
    config = SENetConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_experiment_config(**kwargs) -> ExperimentConfig:
    """Get default experiment configuration with optional overrides."""
    config = ExperimentConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config_to_file(config: BaseModelConfig, config_path: str) -> None:
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


# Default configurations dictionary
DEFAULT_CONFIGS = {
    'clip': get_clip_config(),
    'coolant': get_coolant_config(),
    'senet': get_senet_config(),
    'experiment': get_experiment_config()
}
