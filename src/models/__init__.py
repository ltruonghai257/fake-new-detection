"""
Models module for fake news detection.

This module provides implementations of various deep learning models for multimodal
fake news detection, including:

- CLIP: Cross-modal contrastive learning model
- COOLANT: Cross-modal contrastive learning for multimodal fake news detection
- SENet: Squeeze-and-Excitation networks for attention-based feature learning

The module also includes base classes, utilities, and a factory pattern for
easy model creation and configuration.
"""

from .base import (
    BaseModel,
    MultimodalModel,
    TextEncoder,
    ImageEncoder,
    FastCNN,
    AttentionFusion,
    ContrastiveLoss
)

from .clip_model import (
    CLIP,
    CLIPTextEncoder,
    CLIPImageEncoder,
    PositionalEncoding
)

from .coolant import (
    COOLANT,
    EncodingPart,
    SimilarityModule,
    AmbiguityLearning,
    UnimodalDetection,
    CrossModule4Batch,
    DetectionModule,
    Encoder
)

from .senet import (
    SEBlock,
    ResBlock,
    SENetwork,
    SEAttentionModule,
    AdaptiveSEBlock,
    MultiScaleSEBlock
)

from .config import (
    BaseModelConfig,
    TextEncoderConfig,
    ImageEncoderConfig,
    CLIPConfig,
    COOLANTConfig,
    SENetConfig,
    TrainingConfig,
    DataConfig,
    ExperimentConfig,
    get_clip_config,
    get_coolant_config,
    get_senet_config,
    get_experiment_config
)

from .factory import (
    ModelFactory,
    ModelBuilder,
    create_clip_model,
    create_coolant_model,
    create_senet_model,
    get_model_info,
    list_models,
    create_baseline_models,
    create_experiment_models
)

# Version info
__version__ = "1.0.0"
__author__ = "Fake News Detection Team"

# Model registry for easy access
AVAILABLE_MODELS = {
    'clip': CLIP,
    'coolant': COOLANT,
    'senet': SENetwork,
}

# Default configurations
DEFAULT_CONFIGS = {
    'clip': get_clip_config(),
    'coolant': get_coolant_config(),
    'senet': get_senet_config(),
}

def create_model(model_name: str, config=None, **kwargs):
    """
    Convenience function to create a model.
    
    Args:
        model_name: Name of the model ('clip', 'coolant', 'senet')
        config: Model configuration (optional)
        **kwargs: Additional configuration parameters
        
    Returns:
        Model instance
        
    Example:
        >>> model = create_model('coolant', num_classes=2, shared_dim=128)
        >>> model = create_model('clip', output_dim=512, temperature=0.07)
    """
    return ModelFactory.create_model(model_name, config, **kwargs)

def load_model(checkpoint_path: str, model_name=None):
    """
    Load a model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_name: Model name (if not in checkpoint)
        
    Returns:
        Loaded model instance
        
    Example:
        >>> model = load_model('checkpoints/coolant_best.pt')
        >>> model = load_model('checkpoints/model.pt', model_name='clip')
    """
    return ModelFactory.create_model_from_checkpoint(checkpoint_path, model_name)

def get_model_summary():
    """
    Get a summary of all available models.
    
    Returns:
        Dictionary with model information
    """
    return list_models()

# Expose commonly used classes and functions at package level
__all__ = [
    # Base classes
    'BaseModel',
    'MultimodalModel',
    'TextEncoder', 
    'ImageEncoder',
    'FastCNN',
    'AttentionFusion',
    'ContrastiveLoss',
    
    # Model implementations
    'CLIP',
    'COOLANT', 
    'SENetwork',
    'CLIPTextEncoder',
    'CLIPImageEncoder',
    'PositionalEncoding',
    'EncodingPart',
    'SimilarityModule',
    'AmbiguityLearning',
    'UnimodalDetection',
    'CrossModule4Batch',
    'DetectionModule',
    'Encoder',
    'SEBlock',
    'ResBlock',
    'SEAttentionModule',
    'AdaptiveSEBlock',
    'MultiScaleSEBlock',
    
    # Configuration classes
    'BaseModelConfig',
    'TextEncoderConfig',
    'ImageEncoderConfig',
    'CLIPConfig',
    'COOLANTConfig',
    'SENetConfig',
    'TrainingConfig',
    'DataConfig',
    'ExperimentConfig',
    
    # Factory and utilities
    'ModelFactory',
    'ModelBuilder',
    'create_model',
    'load_model',
    'create_clip_model',
    'create_coolant_model',
    'create_senet_model',
    'get_model_info',
    'list_models',
    'get_model_summary',
    'create_baseline_models',
    'create_experiment_models',
    
    # Configuration functions
    'get_clip_config',
    'get_coolant_config', 
    'get_senet_config',
    'get_experiment_config',
    
    # Constants
    'AVAILABLE_MODELS',
    'DEFAULT_CONFIGS',
]
