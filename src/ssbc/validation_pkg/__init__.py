"""Validation API facade.

Provides a stable package path for validation utilities.
"""

from ssbc.validation import (
    print_validation_results,
    validate_pac_bounds,
)

__all__ = [
    "print_validation_results",
    "validate_pac_bounds",
]
