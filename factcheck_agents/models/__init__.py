"""Trained-model wrappers used by the Evaluate agent.

Both wrappers are lazy and defensive: importing this package never imports
torch, and a missing/broken checkpoint yields an ``unavailable`` ModelResult
rather than raising — the pipeline keeps running while models are validated.
"""

from .phobert_checker import PhoBERTChecker
from .coolant_checker import CoolantChecker

__all__ = ["PhoBERTChecker", "CoolantChecker"]
