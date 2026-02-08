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
        time_ids: Optional[Union[int, Sequence[int]]] = None,
        entity_ids: Optional[Union[int, Sequence[int]]] = None,
        sample_idx: Optional[Union[int, Sequence[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
    ) -> pl.DataFrame:
        """Filter DataFrame by time IDs, entity IDs, samples, and features.
        
        Args:
            df: DataFrame to filter.
            index_mgr: Index configuration.
            time_ids: Time IDs to include (None = all).
            entity_ids: Entity IDs to include (None = all).
            sample_idx: Sample indices to include (None = all).
            features: Feature columns to include (None = all).
            
        Returns:
            Filtered DataFrame.
        """
        result = df
        
        # Filter by time IDs
        if time_ids is not None:
            if isinstance(time_ids, int):
                time_ids = [time_ids]
            time_list = self._cast_ids(time_ids)
            result = result.filter(pl.col(index_mgr.time_col).is_in(time_list))
            self._logger.debug(f"Filtering to {len(time_list)} time IDs")
        
        # Filter by entity IDs
        if entity_ids is not None:
            if isinstance(entity_ids, int):
                entity_ids = [entity_ids]
            entity_list = self._cast_ids(entity_ids)
            result = result.filter(pl.col(index_mgr.entity_col).is_in(entity_list))
            self._logger.debug(f"Filtering to {len(entity_list)} entities")
        
        # Filter by sample indices
        if sample_idx is not None and index_mgr.sample_col is not None:
            if isinstance(sample_idx, int):
                sample_idx = [sample_idx]
            sample_list = self._cast_ids(sample_idx)
            if index_mgr.sample_col in result.columns:
                result = result.filter(pl.col(index_mgr.sample_col).is_in(sample_list))
                self._logger.debug(f"Filtering to {len(sample_list)} samples")
        
        # Select feature columns
        if features is not None:
            if isinstance(features, str):
                features = [features]
            select_cols = [index_mgr.time_col, index_mgr.entity_col]
            if index_mgr.sample_col and index_mgr.sample_col in result.columns:
                select_cols.append(index_mgr.sample_col)
            select_cols.extend([f for f in features if f in result.columns])
            result = result.select(select_cols)
            self._logger.debug(f"Selecting {len(features)} feature columns")
        
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
