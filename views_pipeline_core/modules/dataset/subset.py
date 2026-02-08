"""
Subset Module
=============

Filtering and subsetting datasets by entity/time ranges.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union, TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .index import IndexModule


class SubsetModule:
    """Handles filtering and subsetting of datasets.
    
    Provides methods to filter by entity IDs, time ranges, and samples.
    """
    
    def __init__(self):
        """Initialize SubsetModule."""
        self._logger = logging.getLogger(f"{__name__}.SubsetModule")
    
    def filter(
        self,
        df: pl.DataFrame,
        index_mgr: "IndexModule",
        entity_ids: Optional[Sequence[int]] = None,
        time_range: Optional[tuple[int, int]] = None,
        sample_ids: Optional[Sequence[int]] = None,
    ) -> pl.DataFrame:
        """Filter DataFrame by entity IDs, time range, and/or sample IDs.
        
        Args:
            df: DataFrame to filter.
            index_mgr: Index configuration.
            entity_ids: Entity IDs to include (None = all).
            time_range: (start, end) inclusive range (None = all).
            sample_ids: Sample IDs to include (None = all).
            
        Returns:
            Filtered DataFrame.
        """
        mask = pl.lit(True)
        
        if entity_ids is not None:
            entity_list = self._cast_ids(entity_ids)
            mask = mask & pl.col(index_mgr.entity_col).is_in(entity_list)
            self._logger.debug(f"Filtering to {len(entity_list)} entities")
        
        if time_range is not None:
            start, end = time_range
            mask = mask & (pl.col(index_mgr.time_col) >= start)
            mask = mask & (pl.col(index_mgr.time_col) <= end)
            self._logger.debug(f"Filtering to time range [{start}, {end}]")
        
        if sample_ids is not None and index_mgr.sample_col is not None:
            sample_list = self._cast_ids(sample_ids)
            mask = mask & pl.col(index_mgr.sample_col).is_in(sample_list)
            self._logger.debug(f"Filtering to {len(sample_list)} samples")
        
        result = df.filter(mask)
        self._logger.info(f"Filtered: {len(df):,} → {len(result):,} rows")
        return result
    
    def _cast_ids(
        self, 
        ids: Union[Sequence[int], List[int], range]
    ) -> List[int]:
        """Cast various ID sequence types to list.
        
        Args:
            ids: Sequence of IDs.
            
        Returns:
            List of integers.
        """
        if isinstance(ids, range):
            return list(ids)
        return list(ids)
    
    def filter_by_time(
        self,
        df: pl.DataFrame,
        index_mgr: "IndexModule",
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> pl.DataFrame:
        """Filter by time range only.
        
        Args:
            df: DataFrame to filter.
            index_mgr: Index configuration.
            start: Start time (inclusive, None = no lower bound).
            end: End time (inclusive, None = no upper bound).
            
        Returns:
            Filtered DataFrame.
        """
        mask = pl.lit(True)
        
        if start is not None:
            mask = mask & (pl.col(index_mgr.time_col) >= start)
        if end is not None:
            mask = mask & (pl.col(index_mgr.time_col) <= end)
        
        return df.filter(mask)
    
    def filter_by_entities(
        self,
        df: pl.DataFrame,
        index_mgr: "IndexModule",
        entity_ids: Sequence[int],
    ) -> pl.DataFrame:
        """Filter by entity IDs only.
        
        Args:
            df: DataFrame to filter.
            index_mgr: Index configuration.
            entity_ids: Entity IDs to include.
            
        Returns:
            Filtered DataFrame.
        """
        entity_list = self._cast_ids(entity_ids)
        return df.filter(pl.col(index_mgr.entity_col).is_in(entity_list))
    
    def sample_entities(
        self,
        df: pl.DataFrame,
        index_mgr: "IndexModule",
        n: int,
        seed: Optional[int] = None,
    ) -> pl.DataFrame:
        """Randomly sample n entities (keeping all their time steps).
        
        Args:
            df: DataFrame to sample from.
            index_mgr: Index configuration.
            n: Number of entities to sample.
            seed: Random seed for reproducibility.
            
        Returns:
            DataFrame with sampled entities.
        """
        unique_entities = df.select(index_mgr.entity_col).unique()
        
        if n >= len(unique_entities):
            return df
        
        sampled = unique_entities.sample(n=n, seed=seed)
        entity_list = sampled[index_mgr.entity_col].to_list()
        
        return df.filter(pl.col(index_mgr.entity_col).is_in(entity_list))


__all__ = ["SubsetModule"]
