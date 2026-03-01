"""
Factory module for creating multimodal fake news detection models.

This module provides a factory pattern for creating models with different
configurations, ensuring consistent model instantiation across the codebase.
"""

import torch
from typing import Dict, Any, Optional, Type, Union
from pathlib import Path

from .clip_model import CLIP, create_clip_model
from .coolant import COOLANT
from .coolant_official import COOLANT_Official
from .config import (
    CLIPConfig,
    COOLANTConfig,
    SENetConfig,
    ExperimentConfig,
    get_clip_config,
    get_coolant_config,
    get_senet_config,
    get_experiment_config,
)


class ModelBuilder:
    """Builder class for constructing models with specific configurations."""

    def __init__(self):
        self._model_type = None
        self._config = None
        self._custom_params = {}

    def set_model_type(self, model_type: str) -> "ModelBuilder":
        """Set the model type."""
        self._model_type = model_type
        return self

    def set_config(self, config: Union[Dict[str, Any], Any]) -> "ModelBuilder":
        """Set the model configuration."""
        self._config = config
        return self

    def add_params(self, **kwargs) -> "ModelBuilder":
        """Add custom parameters."""
        self._custom_params.update(kwargs)
        return self

    def build(self) -> torch.nn.Module:
        """Build the model."""
        if self._model_type is None:
            raise ValueError("Model type must be set before building")

        # Get default config if none provided
        if self._config is None:
            self._config = self._get_default_config(self._model_type)

        # Merge custom parameters
        if isinstance(self._config, dict):
            config_dict = self._config.copy()
            config_dict.update(self._custom_params)
        else:
            config_dict = self._config.to_dict()
            config_dict.update(self._custom_params)

        # Create model
        return self._create_model(self._model_type, config_dict)

    def _get_default_config(self, model_type: str):
        """Get default configuration for model type."""
        config_getters = {
            "clip": get_clip_config,
            "coolant": get_coolant_config,
            "coolant_official": get_coolant_config,
            "senet": get_senet_config,
        }

        if model_type not in config_getters:
            raise ValueError(f"Unknown model type: {model_type}")

        return config_getters[model_type]()

    def _create_model(self, model_type: str, config: Dict[str, Any]) -> torch.nn.Module:
        """Create model instance."""
        model_creators = {
            "clip": self._create_clip,
            "coolant": self._create_coolant,
            "coolant_official": self._create_coolant_official,
            "senet": self._create_senet,
        }

        if model_type not in model_creators:
            raise ValueError(f"Unknown model type: {model_type}")

        return model_creators[model_type](config)

    def _create_clip(self, config: Dict[str, Any]) -> CLIP:
        """Create CLIP model."""
        return create_clip_model(config)

    def _create_coolant(self, config: Dict[str, Any]) -> COOLANT:
        """Create COOLANT model."""
        return COOLANT(config)

    def _create_coolant_official(self, config: Dict[str, Any]) -> COOLANT_Official:
        """Create COOLANT official model."""
        return COOLANT_Official(config)

    def _create_senet(self, config: Dict[str, Any]) -> torch.nn.Module:
        """Create SENet model (placeholder)."""
        # This would need to be implemented when SENet is available
        raise NotImplementedError("SENet model not yet implemented")


class ModelFactory:
    """Factory class for creating models with different configurations."""

    _model_registry = {
        "clip": CLIP,
        "coolant": COOLANT,
        "coolant_official": COOLANT_Official,
    }

    _config_registry = {
        "clip": get_clip_config,
        "coolant": get_coolant_config,
        "coolant_official": get_coolant_config,
        "senet": get_senet_config,
    }

    @classmethod
    def create_model(
        cls,
        model_name: str,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Create a model instance.

        Args:
            model_name: Name of the model to create
            config: Model configuration (optional)
            **kwargs: Additional configuration parameters

        Returns:
            Model instance
        """
        if model_name not in cls._model_registry:
            available_models = list(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {available_models}"
            )

        # Get default config if none provided
        if config is None:
            config = cls._config_registry.get(model_name, lambda: {})()

        # Convert config to dict if it's a config object
        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        else:
            config_dict = config.copy() if isinstance(config, dict) else {}

        # Add kwargs to config
        config_dict.update(kwargs)

        # Create model based on type
        if model_name == "clip":
            return create_clip_model(config_dict)
        elif model_name == "coolant":
            return COOLANT(config_dict)
        elif model_name == "coolant_official":
            return COOLANT_Official(config_dict)
        else:
            # Fallback to direct instantiation
            model_class = cls._model_registry[model_name]
            return model_class(**config_dict)

    @classmethod
    def create_model_from_checkpoint(
        cls, checkpoint_path: str, model_name: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Create a model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model_name: Model name (optional, will try to infer from checkpoint)

        Returns:
            Model instance with loaded weights
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Try to infer model name from checkpoint
        if model_name is None:
            if "model_name" in checkpoint:
                model_name = checkpoint["model_name"]
            elif "config" in checkpoint and "model_type" in checkpoint["config"]:
                model_name = checkpoint["config"]["model_type"]
            else:
                raise ValueError(
                    "Cannot infer model name from checkpoint. Please specify model_name."
                )

        # Get config from checkpoint
        config = checkpoint.get("config", {})

        # Create model
        model = cls.create_model(model_name, config)

        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return model

    @classmethod
    def register_model(cls, name: str, model_class: Type[torch.nn.Module]):
        """Register a new model class."""
        cls._model_registry[name] = model_class

    @classmethod
    def register_config(cls, name: str, config_func):
        """Register a new config function."""
        cls._config_registry[name] = config_func

    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model names."""
        return list(cls._model_registry.keys())

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        if model_name not in cls._model_registry:
            raise ValueError(f"Unknown model: {model_name}")

        model_class = cls._model_registry[model_name]
        config_func = cls._config_registry.get(model_name)

        info = {
            "name": model_name,
            "class": model_class.__name__,
            "module": model_class.__module__,
            "doc": model_class.__doc__,
        }

        if config_func:
            default_config = config_func()
            if hasattr(default_config, "to_dict"):
                info["default_config"] = default_config.to_dict()
            else:
                info["default_config"] = default_config

        return info


# Convenience functions
def create_clip_model(
    config: Optional[Union[Dict[str, Any], CLIPConfig]] = None, **kwargs
) -> CLIP:
    """Create a CLIP model."""
    if config is None:
        config = get_clip_config()

    if isinstance(config, CLIPConfig):
        config_dict = config.to_dict()
    else:
        config_dict = config.copy() if isinstance(config, dict) else {}

    config_dict.update(kwargs)
    return CLIP(**config_dict)


def create_coolant_model(
    config: Optional[Union[Dict[str, Any], COOLANTConfig]] = None, **kwargs
) -> COOLANT:
    """Create a COOLANT model."""
    if config is None:
        config = get_coolant_config()

    if isinstance(config, COOLANTConfig):
        config_dict = config.to_dict()
    else:
        config_dict = config.copy() if isinstance(config, dict) else {}

    config_dict.update(kwargs)
    return COOLANT_Official(config_dict)


def create_coolant_official_model(
    config: Optional[Union[Dict[str, Any], COOLANTConfig]] = None, **kwargs
) -> COOLANT_Official:
    """Create a COOLANT official model."""
    if config is None:
        config = get_coolant_config()

    if isinstance(config, COOLANTConfig):
        config_dict = config.to_dict()
    else:
        config_dict = config.copy() if isinstance(config, dict) else {}

    config_dict.update(kwargs)
    return COOLANT_Official(config_dict)


def create_senet_model(
    config: Optional[Union[Dict[str, Any], SENetConfig]] = None, **kwargs
) -> torch.nn.Module:
    """Create a SENet model (placeholder)."""
    raise NotImplementedError("SENet model not yet implemented")


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a model."""
    return ModelFactory.get_model_info(model_name)


def list_models() -> list:
    """Get list of available models."""
    return ModelFactory.get_available_models()


def create_baseline_models(config: ExperimentConfig) -> Dict[str, torch.nn.Module]:
    """Create baseline models for comparison."""
    models = {}

    # Create CLIP baseline
    clip_config = get_clip_config()
    clip_model = create_clip_model(clip_config)
    models["clip_baseline"] = clip_model

    # Create COOLANT baseline
    coolant_config = get_coolant_config()
    coolant_model = create_coolant_model(coolant_config)
    models["coolant_baseline"] = coolant_model

    return models


def create_experiment_models(config: ExperimentConfig) -> Dict[str, torch.nn.Module]:
    """
    Create multiple models for an experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary of model instances
    """
    models = {}

    # Create main model
    main_model = ModelFactory.create_model(config.model_type, config.model_config)
    models["main"] = main_model

    # Create additional models if specified
    if hasattr(config, "additional_models"):
        for model_config in config.additional_models:
            model = ModelFactory.create_model(
                model_config["type"], model_config.get("config", {})
            )
            models[model_config["name"]] = model

    return models


# Global model builder instance
model_builder = ModelBuilder()
