import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import logging

from .base import BaseModel, MultimodalModel
from .clip_model import CLIP
from .coolant import COOLANT
from .senet import SENetwork, SEAttentionModule
from .config import (
    BaseModelConfig, CLIPConfig, COOLANTConfig, SENetConfig,
    ExperimentConfig, get_clip_config, get_coolant_config, get_senet_config
)

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating models."""
    
    _model_registry = {
        'clip': CLIP,
        'coolant': COOLANT,
        'senet': SENetwork,
    }
    
    _config_registry = {
        'clip': CLIPConfig,
        'coolant': COOLANTConfig,
        'senet': SENetConfig,
    }
    
    @classmethod
    def register_model(cls, name: str, model_class: type, config_class: type):
        """Register a new model type."""
        cls._model_registry[name] = model_class
        cls._config_registry[name] = config_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model names."""
        return list(cls._model_registry.keys())
    
    @classmethod
    def create_model(cls, 
                    model_name: str, 
                    config: Optional[Union[Dict[str, Any], BaseModelConfig]] = None,
                    **kwargs) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model to create
            config: Model configuration (dict or config object)
            **kwargs: Additional arguments to override config
            
        Returns:
            Model instance
        """
        if model_name not in cls._model_registry:
            raise ValueError(f"Unknown model: {model_name}. Available models: {cls.get_available_models()}")
        
        model_class = cls._model_registry[model_name]
        config_class = cls._config_registry[model_name]
        
        # Handle configuration
        if config is None:
            # Use default configuration
            if model_name == 'clip':
                config = get_clip_config()
            elif model_name == 'coolant':
                config = get_coolant_config()
            elif model_name == 'senet':
                config = get_senet_config()
            else:
                config = config_class()
        elif isinstance(config, dict):
            # Convert dict to config object
            config = config_class.from_dict(config)
        
        # Override with kwargs
        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = config_class.from_dict(config_dict)
        
        # Create model
        if model_name == 'senet':
            # SENet has different constructor signature
            model = model_class(
                in_channel=config.in_channel,
                filters=config.filters,
                blocks=config.blocks,
                num_classes=config.num_classes
            )
        else:
            model = model_class(config.to_dict())
        
        logger.info(f"Created model: {model_name} with config: {config}")
        return model
    
    @classmethod
    def create_model_from_checkpoint(cls, 
                                   checkpoint_path: str,
                                   model_name: Optional[str] = None) -> BaseModel:
        """
        Create model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_name: Model name (if not in checkpoint)
            
        Returns:
            Model instance loaded from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            if 'model_name' in config:
                model_name = config['model_name']
        
        if model_name is None:
            raise ValueError("Model name not found in checkpoint and not provided")
        
        model = cls.create_model(model_name, config)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
        return model


class ModelBuilder:
    """Builder class for creating models with fluent interface."""
    
    def __init__(self):
        self.model_name = None
        self.config = {}
        self.device = None
        self.pretrained_path = None
    
    def model(self, name: str) -> 'ModelBuilder':
        """Set model name."""
        self.model_name = name
        return self
    
    def config_from_dict(self, config: Dict[str, Any]) -> 'ModelBuilder':
        """Set configuration from dictionary."""
        self.config.update(config)
        return self
    
    def config_from_file(self, config_path: str) -> 'ModelBuilder':
        """Set configuration from file."""
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.config.update(config)
        return self
    
    def set_device(self, device: str) -> 'ModelBuilder':
        """Set device for model."""
        self.device = device
        return self
    
    def from_pretrained(self, path: str) -> 'ModelBuilder':
        """Load from pretrained checkpoint."""
        self.pretrained_path = path
        return self
    
    def set_param(self, key: str, value: Any) -> 'ModelBuilder':
        """Set a specific parameter."""
        self.config[key] = value
        return self
    
    def build(self) -> BaseModel:
        """Build the model."""
        if self.model_name is None:
            raise ValueError("Model name must be specified")
        
        if self.pretrained_path:
            model = ModelFactory.create_model_from_checkpoint(
                self.pretrained_path, self.model_name
            )
        else:
            model = ModelFactory.create_model(self.model_name, self.config)
        
        if self.device:
            model = model.to(self.device)
        
        return model


def create_clip_model(config: Optional[Dict[str, Any]] = None, **kwargs) -> CLIP:
    """Convenience function to create CLIP model."""
    return ModelFactory.create_model('clip', config, **kwargs)


def create_coolant_model(config: Optional[Dict[str, Any]] = None, **kwargs) -> COOLANT:
    """Convenience function to create COOLANT model."""
    return ModelFactory.create_model('coolant', config, **kwargs)


def create_senet_model(config: Optional[Dict[str, Any]] = None, **kwargs) -> SENetwork:
    """Convenience function to create SENet model."""
    return ModelFactory.create_model('senet', config, **kwargs)


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a model."""
    if model_name not in ModelFactory._model_registry:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_class = ModelFactory._model_registry[model_name]
    config_class = ModelFactory._config_registry[model_name]
    
    # Create a dummy model to get parameter count
    try:
        if model_name == 'senet':
            dummy_model = model_class(64, 128, 19, 2)
        else:
            dummy_config = config_class()
            dummy_model = model_class(dummy_config.to_dict())
        
        param_count = sum(p.numel() for p in dummy_model.parameters())
        trainable_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    except Exception as e:
        param_count = "Unknown"
        trainable_params = "Unknown"
        logger.warning(f"Could not compute parameter count for {model_name}: {e}")
    
    return {
        'name': model_name,
        'class': model_class.__name__,
        'config_class': config_class.__name__,
        'total_parameters': param_count,
        'trainable_parameters': trainable_params,
        'description': model_class.__doc__ or "No description available"
    }


def list_models() -> Dict[str, Dict[str, Any]]:
    """List all available models with their information."""
    models_info = {}
    for model_name in ModelFactory.get_available_models():
        try:
            models_info[model_name] = get_model_info(model_name)
        except Exception as e:
            logger.error(f"Error getting info for model {model_name}: {e}")
            models_info[model_name] = {
                'name': model_name,
                'error': str(e)
            }
    
    return models_info


# Example usage functions
def create_baseline_models() -> Dict[str, BaseModel]:
    """Create baseline models for comparison."""
    models = {}
    
    # CLIP baseline
    models['clip'] = create_clip_model({
        'output_dim': 256,
        'temperature': 0.07
    })
    
    # COOLANT model
    models['coolant'] = create_coolant_model({
        'shared_dim': 128,
        'sim_dim': 64,
        'contrastive_weight': 1.0,
        'classification_weight': 1.0
    })
    
    # SENet model
    models['senet'] = create_senet_model({
        'filters': 128,
        'blocks': 19,
        'num_classes': 2
    })
    
    return models


def create_experiment_models(experiment_config: ExperimentConfig) -> BaseModel:
    """Create model from experiment configuration."""
    model_config = experiment_config.model_config
    return ModelFactory.create_model(model_config.model_name, model_config)
