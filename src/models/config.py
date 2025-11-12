from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Union
import json


@dataclass
class BaseModelConfig:
    """Base configuration for all models."""
    model_name: str = "base_model"
    num_classes: int = 2
    dropout: float = 0.1
    hidden_dim: int = 256
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class TextEncoderConfig:
    """Configuration for text encoders."""
    encoder_type: str = "cnn"  # "cnn" or "transformer"
    vocab_size: int = 49408
    embed_dim: int = 768
    num_heads: int = 8
    num_layers: int = 12
    hidden_dim: int = 2048
    dropout: float = 0.1
    cnn_channel: int = 32
    cnn_kernel_size: Tuple[int, ...] = (1, 2, 4, 8)
    output_dim: int = 512


@dataclass
class ImageEncoderConfig:
    """Configuration for image encoders."""
    input_dim: int = 512
    hidden_dim: int = 256
    output_dim: int = 512
    dropout: float = 0.1


@dataclass
class CLIPConfig(BaseModelConfig):
    """Configuration for CLIP model."""
    model_name: str = "clip"
    vocab_size: int = 49408
    text_embed_dim: int = 768
    image_input_dim: int = 512
    output_dim: int = 512
    temperature: float = 0.07
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    image_encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = super().to_dict()
        config_dict['text_encoder'] = self.text_encoder.__dict__
        config_dict['image_encoder'] = self.image_encoder.__dict__
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        text_encoder_config = TextEncoderConfig(**config_dict.pop('text_encoder', {}))
        image_encoder_config = ImageEncoderConfig(**config_dict.pop('image_encoder', {}))
        return cls(
            text_encoder=text_encoder_config,
            image_encoder=image_encoder_config,
            **config_dict
        )


@dataclass
class COOLANTConfig(BaseModelConfig):
    """Configuration for COOLANT model."""
    model_name: str = "coolant"
    text_dim: int = 768
    image_dim: int = 512
    shared_dim: int = 128
    sim_dim: int = 64
    feature_dim: int = 96  # 64 + 16 + 16
    h_dim: int = 64
    cnn_channel: int = 32
    cnn_kernel_size: Tuple[int, ...] = (1, 2, 4, 8)
    
    # Loss weights
    contrastive_weight: float = 1.0
    classification_weight: float = 1.0
    similarity_weight: float = 0.5
    temperature: float = 0.07
    
    # SE attention configuration
    se_filters: int = 128
    se_blocks: int = 19
    se_reduction: int = 16


@dataclass
class SENetConfig(BaseModelConfig):
    """Configuration for SENet model."""
    model_name: str = "senet"
    in_channel: int = 64
    filters: int = 128
    blocks: int = 19
    reduction: int = 16
    scales: Tuple[int, ...] = (1, 3, 5)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    
    # Scheduler configuration
    scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    scheduler_warmup_ratio: float = 0.1
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Logging and checkpointing
    log_every_n_steps: int = 100
    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 1
    
    # Mixed precision training
    use_amp: bool = True
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class DataConfig:
    """Configuration for data processing."""
    # Text processing
    max_text_length: int = 512
    text_tokenizer: str = "bert-base-uncased"
    text_embedding_dim: int = 768
    
    # Image processing
    image_size: Tuple[int, int] = (224, 224)
    image_channels: int = 3
    image_feature_dim: int = 512
    
    # Data augmentation
    use_text_augmentation: bool = True
    use_image_augmentation: bool = True
    augmentation_probability: float = 0.5
    
    # Dataset splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Data loading
    shuffle_train: bool = True
    shuffle_val: bool = False
    shuffle_test: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str = "fake_news_detection"
    model_config: Union[CLIPConfig, COOLANTConfig, SENetConfig] = field(default_factory=COOLANTConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    
    # Paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    num_gpus: int = 1
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'model_config': self.model_config.to_dict(),
            'training_config': self.training_config.__dict__,
            'data_config': self.data_config.__dict__,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'device': self.device,
            'num_gpus': self.num_gpus,
            'seed': self.seed,
            'deterministic': self.deterministic
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        model_config_dict = config_dict.pop('model_config')
        model_name = model_config_dict.get('model_name', 'coolant')
        
        if model_name == 'clip':
            model_config = CLIPConfig.from_dict(model_config_dict)
        elif model_name == 'coolant':
            model_config = COOLANTConfig.from_dict(model_config_dict)
        elif model_name == 'senet':
            model_config = SENetConfig.from_dict(model_config_dict)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        training_config = TrainingConfig(**config_dict.pop('training_config', {}))
        data_config = DataConfig(**config_dict.pop('data_config', {}))
        
        return cls(
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            **config_dict
        )
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations
def get_clip_config() -> CLIPConfig:
    """Get default CLIP configuration."""
    return CLIPConfig(
        model_name="clip",
        output_dim=512,
        temperature=0.07,
        text_encoder=TextEncoderConfig(
            encoder_type="cnn",
            cnn_channel=32,
            output_dim=512
        ),
        image_encoder=ImageEncoderConfig(
            input_dim=512,
            output_dim=512
        )
    )


def get_coolant_config() -> COOLANTConfig:
    """Get default COOLANT configuration."""
    return COOLANTConfig(
        model_name="coolant",
        shared_dim=128,
        sim_dim=64,
        feature_dim=96,
        contrastive_weight=1.0,
        classification_weight=1.0,
        similarity_weight=0.5
    )


def get_senet_config() -> SENetConfig:
    """Get default SENet configuration."""
    return SENetConfig(
        model_name="senet",
        in_channel=64,
        filters=128,
        blocks=19,
        reduction=16
    )


def get_experiment_config(model_name: str = "coolant") -> ExperimentConfig:
    """Get default experiment configuration."""
    if model_name == "clip":
        model_config = get_clip_config()
    elif model_name == "coolant":
        model_config = get_coolant_config()
    elif model_name == "senet":
        model_config = get_senet_config()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return ExperimentConfig(
        experiment_name=f"fake_news_{model_name}",
        model_config=model_config,
        training_config=TrainingConfig(),
        data_config=DataConfig()
    )
