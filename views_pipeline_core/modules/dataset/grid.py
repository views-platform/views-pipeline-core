"""
Grid Consistency Module
=======================

Ensures dense spatiotemporal grids with no missing combinations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .index import IndexModule
    from .shape import DistributionLayout


class GridModule:
    """Ensures dense spatiotemporal grids with no missing combinations.
    
    Handles checking for and filling missing (time, entity, [sample]) 
    combinations in the dataset.
    """
    
    def __init__(self):
        """Initialize GridModule."""
        self._logger = logging.getLogger(f"{__name__}.GridModule")
    
    def check_completeness(
        self,
        df: pl.DataFrame,
        index_mgr: "IndexModule",
        dist_layout: Optional["DistributionLayout"] = None,
    ) -> Tuple[bool, int]:
        """Check if grid is complete (no missing combinations).
        
        Args:
            df: DataFrame to check.
            index_mgr: Index configuration.
            dist_layout: Distribution layout configuration.
            
        Returns:
            Tuple of (is_complete, missing_count).
        """
        n_times, n_entities, n_samples = index_mgr.get_dimensions(df, dist_layout)
        
        # For array-based layout, we only check time × entity
        if dist_layout and dist_layout.is_array_based:
            expected_rows = n_times * n_entities
        else:
            expected_rows = n_times * n_entities * n_samples
        
        actual_rows = len(df)
        is_complete = actual_rows == expected_rows
        missing_count = expected_rows - actual_rows
        
        if not is_complete:
            self._logger.warning(
                f"Grid incomplete: {actual_rows:,} rows, expected {expected_rows:,}"
            )
        
        return is_complete, max(0, missing_count)
    
    def fix_consistency(
        self,
        df: pl.DataFrame,
        index_mgr: "IndexModule",
        dist_layout: Optional["DistributionLayout"] = None,
        fill_value: Optional[Any] = None,
    ) -> pl.DataFrame:
        """Ensure every (time, entity, [sample]) combination exists.
        
        Creates a complete grid and left-joins the original data,
        filling missing values with the specified fill_value.
        
        Args:
            df: DataFrame to fix.
            index_mgr: Index configuration.
            dist_layout: Distribution layout configuration.
            fill_value: Value to fill missing cells (None = keep as null).
            
        Returns:
            DataFrame with complete grid.
        """
        self._logger.info("Fixing space-time consistency...")
        
        times, entities, samples = index_mgr.get_unique_values(df)
        
        # Build grid based on layout
        grid = (
            pl.DataFrame({index_mgr.time_col: times})
            .join(pl.DataFrame({index_mgr.entity_col: entities}), how="cross")
        )
        
        if samples is not None:
            grid = grid.join(pl.DataFrame({index_mgr.sample_col: samples}), how="cross")
        
        # Left join original data
        result = grid.join(df, on=index_mgr.index_cols, how="left")
        
        if fill_value is not None:
            data_cols = [c for c in result.columns if c not in index_mgr.index_cols_set]
            result = result.with_columns([
                pl.col(c).fill_null(fill_value) for c in data_cols
            ])
        
        result = result.sort(index_mgr.index_cols)
        self._logger.info(f"Grid completed: {len(result):,} rows")
        return result


__all__ = ["GridModule"]
