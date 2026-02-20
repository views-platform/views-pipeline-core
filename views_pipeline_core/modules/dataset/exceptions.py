"""
Dataset Exceptions
==================

Custom exceptions for dataset operations with detailed error context.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class DatasetError(Exception):
    """Base exception for all dataset-related errors.
    
    Provides structured error context via the `details` dictionary.
    
    Attributes:
        message: Human-readable error message.
        details: Optional dictionary with additional context.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class ValidationError(DatasetError):
    """Raised when input validation fails.
    
    Examples:
        - Missing required columns
        - Invalid column dtypes
        - Unsupported file formats
    """
    pass


class IntegrityError(DatasetError):
    """Raised when data integrity checks fail.
    
    Examples:
        - Missing grid cells
        - Inconsistent row counts
        - Tensor conversion failures
    """
    pass


class ReconciliationError(DatasetError):
    """Raised when hierarchical reconciliation fails.
    
    Examples:
        - Missing country mappings
        - Invalid grid-to-country links
        - Reconciliation constraint violations
    """
    pass


class MetadataError(DatasetError):
    """Raised when metadata operations fail.
    
    Examples:
        - Metadata file not found
        - Missing required metadata columns
        - Invalid entity IDs
    """
    pass


class TensorConversionError(DatasetError):
    """Raised when tensor conversion fails.
    
    Examples:
        - Shape mismatch during conversion
        - Missing features
        - Invalid array column structures
    """
    pass


__all__ = [
    "DatasetError",
    "ValidationError",
    "IntegrityError",
    "ReconciliationError",
    "MetadataError",
    "TensorConversionError",
]
