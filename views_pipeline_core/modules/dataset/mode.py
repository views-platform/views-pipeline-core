"""
Mode Module
===========

Dataset mode management: Historical (training) vs Forecast (prediction).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, TYPE_CHECKING

import polars as pl

from .exceptions import ValidationError
from .index import _get_columns

if TYPE_CHECKING:
    from .index import IndexModule


class ModeModule:
    """Manages dataset mode: Historical (training) vs Forecast (prediction).
    
    Historical mode:
        - Requires target columns to be specified
        - Used for training ML models
        
    Forecast mode:
        - Targets are predictions (prefixed with 'pred_')
        - Used for inference
    
    Attributes:
        mode: Current mode ('historical' or 'forecast').
        target_cols: List of target column names.
    """
    
    VALID_MODES = ("historical", "forecast")
    PREDICTION_PREFIX = "pred_"
    
    def __init__(self, mode: str, target_cols: Optional[List[str]] = None):
        """Initialize ModeModule.
        
        Args:
            mode: Dataset mode ('historical' or 'forecast').
            target_cols: Target column names (required for historical mode).
            
        Raises:
            ValidationError: If mode is invalid.
        """
        self.mode = mode.lower()
        self.target_cols = target_cols or []
        self._logger = logging.getLogger(f"{__name__}.ModeModule")
        
        if self.mode not in self.VALID_MODES:
            raise ValidationError(
                f"Invalid mode: '{mode}'",
                details={"valid_modes": list(self.VALID_MODES)}
            )
    
    @property
    def is_historical(self) -> bool:
        """Check if in historical (training) mode."""
        return self.mode == "historical"
    
    @property
    def is_forecast(self) -> bool:
        """Check if in forecast (prediction) mode."""
        return self.mode == "forecast"
    
    def validate(self, frame: "Union[pl.DataFrame, pl.LazyFrame]") -> None:
        """Validate mode-specific requirements.

        Args:
            frame: DataFrame or LazyFrame to validate.

        Raises:
            ValidationError: If requirements not met.
        """
        columns = _get_columns(frame)
        if self.mode == "historical":
            if not self.target_cols:
                raise ValidationError(
                    "Historical mode requires target_cols.",
                    details={"hint": "Specify target columns when creating dataset"},
                )
            missing = [t for t in self.target_cols if t not in columns]
            if missing:
                raise ValidationError(
                    f"Target columns not found: {missing}",
                    details={"available": list(columns[:20])},
                )

    def get_targets(self, frame: "Union[pl.DataFrame, pl.LazyFrame]") -> List[str]:
        """Get target column names."""
        if self.mode == "historical":
            return self.target_cols
        return [c for c in _get_columns(frame) if c.startswith(self.PREDICTION_PREFIX)]

    def get_features(
        self,
        frame: "Union[pl.DataFrame, pl.LazyFrame]",
        index_mgr: "IndexModule",
    ) -> List[str]:
        """Get feature columns (excluding indices and targets)."""
        targets = set(self.get_targets(frame))
        ignore = index_mgr.index_cols_set.union(targets)
        return [c for c in _get_columns(frame) if c not in ignore]

    def get_all_data_cols(
        self,
        frame: "Union[pl.DataFrame, pl.LazyFrame]",
        index_mgr: "IndexModule",
    ) -> List[str]:
        """Get all data columns (excluding indices)."""
        return [c for c in _get_columns(frame) if c not in index_mgr.index_cols_set]
    
    def split_features_targets(
        self,
        df: pl.DataFrame,
        index_mgr: "IndexModule",
    ) -> Tuple[List[str], List[str]]:
        """Split columns into features and targets.
        
        Args:
            df: DataFrame to analyze.
            index_mgr: Index configuration.
            
        Returns:
            Tuple of (feature_cols, target_cols).
        """
        targets = self.get_targets(df)
        features = self.get_features(df, index_mgr)
        return features, targets
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ModeModule(mode='{self.mode}', targets={self.target_cols})"


__all__ = ["ModeModule"]
