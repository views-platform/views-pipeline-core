"""
Grid Consistency Module
=======================

Ensures dense spatiotemporal grids with no missing combinations.
Works with both ``pl.DataFrame`` and ``pl.LazyFrame``.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .index import IndexModule
    from .shape import DistributionLayout

PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]


class GridModule:
    """Ensures dense spatiotemporal grids with no missing combinations.

    All methods preserve the input frame type: if you pass a
    ``LazyFrame`` you get a ``LazyFrame`` back (the grid fix is added
    to the lazy plan and executed only when collected).
    """

    def __init__(self):
        """Initialize GridModule."""
        self._logger = logging.getLogger(f"{__name__}.GridModule")

    # -----------------------------------------------------------------
    # Completeness check
    # -----------------------------------------------------------------

    def check_completeness(
        self,
        frame: PolarsFrame,
        index_mgr: "IndexModule",
        dist_layout: Optional["DistributionLayout"] = None,
    ) -> Tuple[bool, int]:
        """Check if grid is complete (no missing combinations).

        Args:
            frame: DataFrame or LazyFrame to check.
            index_mgr: Index configuration.
            dist_layout: Distribution layout configuration.

        Returns:
            Tuple of (is_complete, missing_count).
        """
        n_times, n_entities, n_samples = index_mgr.get_dimensions(
            frame, dist_layout
        )

        if dist_layout and dist_layout.is_array_based:
            expected_rows = n_times * n_entities
        else:
            expected_rows = n_times * n_entities * n_samples

        if isinstance(frame, pl.LazyFrame):
            actual_rows = frame.select(pl.len()).collect().item()
        else:
            actual_rows = len(frame)

        is_complete = actual_rows == expected_rows
        missing_count = expected_rows - actual_rows

        if not is_complete:
            self._logger.warning(
                f"Grid incomplete: {actual_rows:,} rows, "
                f"expected {expected_rows:,}"
            )

        return is_complete, max(0, missing_count)

    # -----------------------------------------------------------------
    # Grid fixing
    # -----------------------------------------------------------------

    def fix_consistency(
        self,
        frame: PolarsFrame,
        index_mgr: "IndexModule",
        dist_layout: Optional["DistributionLayout"] = None,
        fill_value: Optional[Any] = None,
        known_times: Optional[List] = None,
        known_entities: Optional[List] = None,
        known_samples: Optional[List] = None,
    ) -> PolarsFrame:
        """Ensure every (time, entity, [sample]) combination exists.

        Creates a complete grid skeleton and left-joins the original data.
        The return type matches the input: ``LazyFrame`` in → ``LazyFrame``
        out (the join is deferred).

        Args:
            frame: DataFrame or LazyFrame to fix.
            index_mgr: Index configuration.
            dist_layout: Distribution layout configuration.
            fill_value: Value to fill missing cells (``None`` = keep null).
            known_times: Pre-cached unique time values (avoids extra scan).
            known_entities: Pre-cached unique entity values.
            known_samples: Pre-cached unique sample values.

        Returns:
            Frame with complete grid (same type as *frame*).
        """
        self._logger.info("Fixing space-time consistency...")

        is_lazy = isinstance(frame, pl.LazyFrame)

        # Determine unique values — use cached lists if provided
        if known_times is not None and known_entities is not None:
            times = known_times
            entities = known_entities
            samples = known_samples
        else:
            times, entities, samples = index_mgr.get_unique_values(frame)

        # Build skeleton
        grid = (
            pl.DataFrame({index_mgr.time_col: times})
            .join(
                pl.DataFrame({index_mgr.entity_col: entities}),
                how="cross",
            )
        )
        if samples is not None and index_mgr.sample_col:
            grid = grid.join(
                pl.DataFrame({index_mgr.sample_col: samples}),
                how="cross",
            )

        # Left-join data onto skeleton
        grid_frame: PolarsFrame = grid.lazy() if is_lazy else grid
        result = grid_frame.join(
            frame, on=index_mgr.index_cols, how="left"
        )

        # Fill nulls
        if fill_value is not None:
            all_cols = (
                result.collect_schema().names()
                if isinstance(result, pl.LazyFrame)
                else result.columns
            )
            data_cols = [
                c for c in all_cols if c not in index_mgr.index_cols_set
            ]
            result = result.with_columns(
                [pl.col(c).fill_null(fill_value) for c in data_cols]
            )

        result = result.sort(index_mgr.index_cols)
        self._logger.info("Grid consistency fix applied.")
        return result


__all__ = ["GridModule"]
