"""Utility modules for the fake news detection system."""

from .device import get_device, validate_device_consistency, DeviceMismatchError

__all__ = ['get_device', 'validate_device_consistency', 'DeviceMismatchError']
