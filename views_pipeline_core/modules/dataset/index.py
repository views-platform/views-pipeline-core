"""
Index Management Module
=======================

Manages time, entity, and sample index columns for spatiotemporal datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import polars as pl

from .exceptions import ValidationError

if TYPE_CHECKING:
    from .shape import DistributionLayout


@dataclass
class IndexModule:
    """Manages time and entity index columns.
    
    For row-based distributions, also manages sample_col.
    For array-based distributions, the sample dimension is implicit in the arrays.
    
    Attributes:
        time_col: Name of the time index column.
        entity_col: Name of the entity index column.
        sample_col: Name of the sample column (ROW_BASED layout only).
    """
    
    time_col: str
    entity_col: str
    sample_col: Optional[str] = None
    _auto_generated_sample: bool = False
    
    @classmethod
    def create(
        cls,
        df: pl.DataFrame,
        time_col: str,
        entity_col: str,
        sample_col: Optional[str] = None,
        dist_layout: Optional["DistributionLayout"] = None,
    ) -> Tuple["IndexModule", pl.DataFrame]:
        """Factory method to create IndexModule with validated DataFrame.
        
        Validates that required columns exist and sorts the DataFrame
        for deterministic tensor reshaping.
        
        Args:
            df: Input DataFrame.
            time_col: Name of time index column.
            entity_col: Name of entity index column.
            sample_col: Name of sample column (only for row-based distributions).
            dist_layout: Distribution layout (auto-detected if None).
            
        Returns:
            Tuple of (IndexModule, sorted DataFrame).
            
        Raises:
            ValidationError: If required columns are missing.
        """
        # Validate required columns
        if time_col not in df.columns:
            raise ValidationError(
                f"Time column '{time_col}' not found in data.",
                details={"available_columns": list(df.columns[:10])}
            )
        if entity_col not in df.columns:
            raise ValidationError(
                f"Entity column '{entity_col}' not found in data.",
                details={"available_columns": list(df.columns[:10])}
            )
        
        # Determine if we need sample column based on layout
        auto_generated = False
        actual_sample_col = None
        
        if dist_layout and dist_layout.is_row_based:
            if sample_col and sample_col in df.columns:
                actual_sample_col = sample_col
            else:
                raise ValidationError(
                    f"Row-based layout detected but sample_col '{sample_col}' not found."
                )
        elif sample_col and sample_col in df.columns:
            # User explicitly provided sample_col and it exists
            actual_sample_col = sample_col
        
        # Sort DataFrame for deterministic tensor reshaping
        sort_cols = [time_col, entity_col]
        if actual_sample_col:
            sort_cols = [actual_sample_col] + sort_cols
        
        df = df.sort(sort_cols)
        
        return cls(
            time_col=time_col,
            entity_col=entity_col,
            sample_col=actual_sample_col,
            _auto_generated_sample=auto_generated,
        ), df
    
    @property
    def index_cols(self) -> List[str]:
        """Returns ordered list of index column names."""
        if self.sample_col:
            return [self.sample_col, self.time_col, self.entity_col]
        return [self.time_col, self.entity_col]
    
    @property
    def index_cols_set(self) -> set:
        """Returns set of index column names."""
        return set(self.index_cols)
    
    @property
    def has_sample_col(self) -> bool:
        """Check if sample column is being used."""
        return self.sample_col is not None
    
    def get_unique_values(
        self,
        df: pl.DataFrame,
    ) -> Tuple[List[Any], List[Any], Optional[List[Any]]]:
        """Extract unique sorted values for each index dimension.
        
        Args:
            df: DataFrame to extract indices from.
        
        Returns:
            Tuple of (times, entities, samples) where samples is None 
            if not using row-based layout.
        """
        times = df[self.time_col].unique().sort().to_list()
        entities = df[self.entity_col].unique().sort().to_list()
        
        if self.sample_col:
            samples = df[self.sample_col].unique().sort().to_list()
            return times, entities, samples
        
        return times, entities, None
    
    def get_dimensions(
        self,
        df: pl.DataFrame,
        dist_layout: Optional["DistributionLayout"] = None,
    ) -> Tuple[int, int, int]:
        """Get grid dimensions as (n_times, n_entities, n_samples).
        
        Args:
            df: DataFrame to measure.
            dist_layout: Distribution layout for sample dimension.
            
        Returns:
            Tuple of (n_times, n_entities, n_samples).
        """
        times, entities, samples = self.get_unique_values(df)
        n_times = len(times)
        n_entities = len(entities)
        
        if samples is not None:
            n_samples = len(samples)
        elif dist_layout:
            n_samples = dist_layout.sample_size
        else:
            n_samples = 1
        
        return n_times, n_entities, n_samples


__all__ = ["IndexModule"]
