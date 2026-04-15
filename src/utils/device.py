"""
Device detection and validation utilities for PyTorch training.

Provides centralized device detection that works across:
- Local Mac (MPS backend)
- Google Colab (CUDA backend)
- CPU fallback
"""

import os
import logging
from typing import Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DeviceMismatchError(Exception):
    """Raised when model and tensor devices don't match."""
    pass


def get_device(force_device: Optional[str] = None) -> torch.device:
    """
    Detect and return the appropriate PyTorch device.
    
    Priority order:
    1. force_device parameter (if provided)
    2. FORCE_DEVICE environment variable (if set)
    3. CUDA (if available)
    4. MPS (if available on Mac)
    5. CPU (fallback)
    
    Args:
        force_device: Optional device string to override detection ('cuda', 'mps', 'cpu')
    
    Returns:
        torch.device: The selected device
    
    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cpu')  # Force CPU
        >>> device = get_device()  # With FORCE_DEVICE=mps env var
    """
    # Check explicit parameter first
    if force_device:
        device = torch.device(force_device)
        logger.info(f"Using device: {device} (forced via parameter)")
        return device
    
    # Check environment variable
    env_device = os.getenv("FORCE_DEVICE")
    if env_device:
        device = torch.device(env_device)
        logger.info(f"Using device: {device} (forced via FORCE_DEVICE env var)")
        return device
    
    # Auto-detect: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: cuda (detected CUDA support)")
        return device
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"Using device: mps (detected Apple Silicon MPS support)")
        return device
    
    device = torch.device("cpu")
    logger.info(f"Using device: cpu (no GPU acceleration available)")
    return device


def validate_device_consistency(
    model: nn.Module,
    *tensors: torch.Tensor,
    expected_device: Optional[Union[str, torch.device]] = None
) -> None:
    """
    Validate that model and tensors are on the same device.
    
    Raises DeviceMismatchError with a clear message if devices don't match.
    
    Args:
        model: PyTorch model to validate
        *tensors: Variable number of tensors to validate
        expected_device: Optional expected device (validates model is on this device)
    
    Raises:
        DeviceMismatchError: If model and tensors are on different devices
    
    Examples:
        >>> validate_device_consistency(model, input_tensor, target_tensor)
        >>> validate_device_consistency(model, batch, expected_device='cuda')
    """
    # Get model device from first parameter
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        # Model has no parameters (unlikely but handle gracefully)
        logger.warning("Model has no parameters, skipping device validation")
        return
    
    # Validate against expected device if provided
    if expected_device is not None:
        if isinstance(expected_device, str):
            expected_device = torch.device(expected_device)
        
        if model_device != expected_device:
            raise DeviceMismatchError(
                f"Model is on {model_device} but expected {expected_device}"
            )
    
    # Validate all tensors match model device
    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            continue
            
        if tensor.device != model_device:
            raise DeviceMismatchError(
                f"Device mismatch: model is on {model_device}, "
                f"but tensor {i} is on {tensor.device}. "
                f"Use tensor.to(device) to move tensors to the correct device."
            )
    
    logger.debug(f"Device validation passed: all components on {model_device}")


def get_device_name(device: Union[str, torch.device]) -> str:
    """
    Get a human-readable device name.
    
    Args:
        device: PyTorch device or device string
    
    Returns:
        str: Device name (e.g., 'cuda:0', 'mps', 'cpu')
    """
    if isinstance(device, str):
        device = torch.device(device)
    return str(device)
