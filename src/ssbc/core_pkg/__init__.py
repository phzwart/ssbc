"""Core API facade.

Provides a stable package path for the core SSBC primitives without moving
the existing `core.py` yet.
"""

from .core import SSBCResult, ssbc_correct

__all__ = [
    "SSBCResult",
    "ssbc_correct",
]
