"""
Subset Module
=============

Filtering and subsetting datasets by entity/time ranges.
All filter methods work on both ``pl.DataFrame`` and ``pl.LazyFrame``
and return the same type.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union, TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .index import IndexModule

PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]


class SubsetModule:
    """Handles filtering and subsetting of datasets.

    All methods accept *and return* the same frame type so that lazy
    query plans remain lazy until the caller explicitly collects.
    """

    def __init__(self):
        """Initialize SubsetModule."""
        self._logger = logging.getLogger(f"{__name__}.SubsetModule")

    def filter(
        self,
        frame: PolarsFrame,
        index_mgr: "IndexModule",
        time_ids: Optional[Union[int, Sequence[int]]] = None,
        entity_ids: Optional[Union[int, Sequence[int]]] = None,
        sample_idx: Optional[Union[int, Sequence[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
    ) -> PolarsFrame:
        """Filter frame by time IDs, entity IDs, samples, and features.

        Returns the same frame type as the input (lazy stays lazy).
        """
        result = frame

        if time_ids is not None:
            if isinstance(time_ids, int):
                time_ids = [time_ids]
            time_list = self._cast_ids(time_ids)
            result = result.filter(pl.col(index_mgr.time_col).is_in(time_list))
            self._logger.debug(f"Filtering to {len(time_list)} time IDs")

        if entity_ids is not None:
            if isinstance(entity_ids, int):
                entity_ids = [entity_ids]
            entity_list = self._cast_ids(entity_ids)
            result = result.filter(
                pl.col(index_mgr.entity_col).is_in(entity_list)
            )
            self._logger.debug(f"Filtering to {len(entity_list)} entities")

        if sample_idx is not None and index_mgr.sample_col is not None:
            if isinstance(sample_idx, int):
                sample_idx = [sample_idx]
            sample_list = self._cast_ids(sample_idx)
            cols = result.columns
            if index_mgr.sample_col in cols:
                result = result.filter(
                    pl.col(index_mgr.sample_col).is_in(sample_list)
                )
                self._logger.debug(
                    f"Filtering to {len(sample_list)} samples"
                )

        if features is not None:
            if isinstance(features, str):
                features = [features]
            cols = result.columns
            select_cols = [index_mgr.time_col, index_mgr.entity_col]
            if index_mgr.sample_col and index_mgr.sample_col in cols:
                select_cols.append(index_mgr.sample_col)
            select_cols.extend([f for f in features if f in cols])
            result = result.select(select_cols)
            self._logger.debug(f"Selecting {len(features)} feature columns")

        return result

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------

    def _cast_ids(
        self,
        ids: Union[Sequence[int], List[int], range],
    ) -> List[int]:
        """Cast various ID sequence types to list."""
        return list(ids)

    def filter_by_time(
        self,
        frame: PolarsFrame,
        index_mgr: "IndexModule",
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> PolarsFrame:
        """Filter by time range only."""
        mask = pl.lit(True)
        if start is not None:
            mask = mask & (pl.col(index_mgr.time_col) >= start)
        if end is not None:
            mask = mask & (pl.col(index_mgr.time_col) <= end)
        return frame.filter(mask)

    def filter_by_entities(
        self,
        frame: PolarsFrame,
        index_mgr: "IndexModule",
        entity_ids: Sequence[int],
    ) -> PolarsFrame:
        """Filter by entity IDs only."""
        entity_list = self._cast_ids(entity_ids)
        return frame.filter(
            pl.col(index_mgr.entity_col).is_in(entity_list)
        )

    def sample_entities(
        self,
        frame: PolarsFrame,
        index_mgr: "IndexModule",
        n: int,
        seed: Optional[int] = None,
    ) -> PolarsFrame:
        """Randomly sample *n* entities (keeping all their time steps).

        Requires collecting the entity column to discover unique values.
        """
        if isinstance(frame, pl.LazyFrame):
            unique_entities = (
                frame.select(index_mgr.entity_col).unique().collect()
            )
        else:
            unique_entities = frame.select(index_mgr.entity_col).unique()

        if n >= len(unique_entities):
            return frame

        sampled = unique_entities.sample(n=n, seed=seed)
        entity_list = sampled[index_mgr.entity_col].to_list()
        return frame.filter(
            pl.col(index_mgr.entity_col).is_in(entity_list)
        )


__all__ = ["SubsetModule"]
