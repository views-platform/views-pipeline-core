"""
Subset Module
=============

Filtering and subsetting datasets by entity/time ranges.
All filter methods work on both ``pl.DataFrame`` and ``pl.LazyFrame``
and return the same type.

The module provides two filtering strategies:

1. **Standard filter** (``filter``) — appends predicate expressions to
   the lazy plan.  Works well for any frame type and column layout.

2. **Indexed filter** (``filter_indexed``) — a two-phase strategy
   optimised for parquet-backed LazyFrames with *heavy* columns
   (``pl.Array``, ``pl.List``).  Phase 1 scans only lightweight index
   columns (column-projection pushdown skips the heavy data), collects
   matching physical row positions, and decides how clustered the
   matches are.  If the matching rows span a small fraction of the
   file, Phase 2 uses ``.slice()`` to read only the relevant parquet
   row groups — orders of magnitude faster for entity-sorted files.
   Otherwise it falls back to the standard filter.
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

    # -----------------------------------------------------------------
    # Heavy-column detection
    # -----------------------------------------------------------------

    @staticmethod
    def has_heavy_columns(frame: PolarsFrame) -> bool:
        """Check whether the frame schema contains heavy column types.

        Array and List columns (e.g. ``Array(Float32, 256)``) dominate
        parquet I/O because each row stores hundreds or thousands of
        scalar values.  When such columns are present, the two-phase
        :meth:`filter_indexed` strategy can avoid decompressing them
        entirely during the row-selection phase.
        """
        schema = (
            frame.collect_schema()
            if isinstance(frame, pl.LazyFrame)
            else frame.schema
        )
        for name in schema.names():
            if isinstance(schema[name], (pl.Array, pl.List)):
                return True
        return False

    # -----------------------------------------------------------------
    # Standard filter
    # -----------------------------------------------------------------

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
            cols = result.collect_schema().names() if isinstance(result, pl.LazyFrame) else result.columns
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
            cols = result.collect_schema().names() if isinstance(result, pl.LazyFrame) else result.columns
            select_cols = [index_mgr.time_col, index_mgr.entity_col]
            if index_mgr.sample_col and index_mgr.sample_col in cols:
                select_cols.append(index_mgr.sample_col)
            select_cols.extend([f for f in features if f in cols])
            result = result.select(select_cols)
            self._logger.debug(f"Selecting {len(features)} feature columns")

        return result

    # -----------------------------------------------------------------
    # Two-phase optimised filter
    # -----------------------------------------------------------------

    def filter_indexed(
        self,
        frame: PolarsFrame,
        raw_frame: PolarsFrame,
        index_mgr: "IndexModule",
        total_rows: int,
        time_ids: Optional[Union[int, Sequence[int]]] = None,
        entity_ids: Optional[Union[int, Sequence[int]]] = None,
        sample_idx: Optional[Union[int, Sequence[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
        scatter_threshold: float = 0.3,
    ) -> PolarsFrame:
        """Two-phase filter optimised for parquet with heavy columns.

        **Phase 1 — Index-only scan.**  Attaches physical row numbers
        (``with_row_index``) to *raw_frame* (the unsorted
        ``scan_parquet`` plan), projects only lightweight index columns,
        applies entity / time predicates and collects the matching row
        positions.  Column-projection pushdown ensures heavy
        ``Array`` / ``List`` columns are **never** read from disk in
        this phase.

        **Phase 2 — Targeted read.**  Evaluates how clustered the
        matching rows are.  If the span (``max_idx − min_idx + 1``) is
        less than *scatter_threshold* of the file, a ``.slice()`` reads
        only the relevant parquet row groups — orders of magnitude
        faster for entity-sorted files.  Otherwise falls back to the
        standard :meth:`filter` on the sorted *frame*.

        Parameters
        ----------
        frame : PolarsFrame
            Sorted LazyFrame (``self._lf``).
        raw_frame : PolarsFrame
            Unsorted LazyFrame in physical row order
            (``self._raw_lf``).
        index_mgr : IndexModule
            Index column configuration.
        total_rows : int
            Total row count (``self._n_rows``).
        time_ids, entity_ids, sample_idx, features
            Same semantics as :meth:`filter`.
        scatter_threshold : float
            Fraction of the file span below which slice-based reads
            are used (default 0.30 = 30 %).

        Returns
        -------
        PolarsFrame
            Filtered LazyFrame (caller is responsible for
            ``.sort().collect()``).
        """
        has_row_filter = time_ids is not None or entity_ids is not None
        if not has_row_filter or not isinstance(raw_frame, pl.LazyFrame):
            return self.filter(
                frame, index_mgr, time_ids, entity_ids,
                sample_idx, features,
            )

        # ── Phase 1: index-only scan with physical row positions ─────
        index_cols = [index_mgr.time_col, index_mgr.entity_col]
        if index_mgr.sample_col:
            index_cols.append(index_mgr.sample_col)

        # with_row_index BEFORE select → __ridx = physical parquet row.
        # select projects away heavy columns → column projection pushdown
        # ensures Array/List data is never read from disk.
        lightweight = raw_frame.with_row_index("__ridx").select(
            ["__ridx"] + index_cols
        )

        # Normalise filter values (same logic as filter())
        tids = eids = sids = None
        if time_ids is not None:
            tids = [time_ids] if isinstance(time_ids, int) else self._cast_ids(time_ids)
            lightweight = lightweight.filter(
                pl.col(index_mgr.time_col).is_in(tids)
            )
        if entity_ids is not None:
            eids = [entity_ids] if isinstance(entity_ids, int) else self._cast_ids(entity_ids)
            lightweight = lightweight.filter(
                pl.col(index_mgr.entity_col).is_in(eids)
            )
        if sample_idx is not None and index_mgr.sample_col:
            sids = [sample_idx] if isinstance(sample_idx, int) else self._cast_ids(sample_idx)
            lightweight = lightweight.filter(
                pl.col(index_mgr.sample_col).is_in(sids)
            )

        matched = lightweight.collect()
        n_matched = len(matched)

        if n_matched == 0:
            self._logger.info("Pre-filter: 0 matching rows")
            return frame.head(0)

        indices = matched["__ridx"].sort()
        lo = int(indices[0])
        hi = int(indices[-1])
        span = hi - lo + 1

        self._logger.info(
            f"Pre-filter: {n_matched:,} rows, span {lo:,}–{hi:,} "
            f"({span:,} = {span / max(total_rows, 1):.1%} of file)"
        )

        # ── Phase 2: choose read strategy ────────────────────────────
        if total_rows > 0 and span / total_rows < scatter_threshold:
            # Clustered → .slice() reads only the relevant row groups
            self._logger.info(
                f"Slice-based read: offset={lo:,}, length={span:,}"
            )
            result = raw_frame.slice(lo, span)

            # Re-apply exact filters within the slice
            if tids is not None:
                result = result.filter(
                    pl.col(index_mgr.time_col).is_in(tids)
                )
            if eids is not None:
                result = result.filter(
                    pl.col(index_mgr.entity_col).is_in(eids)
                )
            if sids is not None and index_mgr.sample_col:
                result = result.filter(
                    pl.col(index_mgr.sample_col).is_in(sids)
                )
        else:
            # Scattered → fall back to standard filter on sorted frame
            self._logger.info(
                "Scatter exceeds threshold — standard filter fallback"
            )
            return self.filter(
                frame, index_mgr, time_ids, entity_ids,
                sample_idx, features,
            )

        # Apply feature selection (only reached on the slice path;
        # the fallback branch delegates to filter() which handles it)
        if features is not None:
            if isinstance(features, str):
                features = [features]
            cols = (
                result.collect_schema().names()
                if isinstance(result, pl.LazyFrame)
                else result.columns
            )
            select_cols = list(index_mgr.index_cols)
            select_cols.extend(f for f in features if f in cols)
            result = result.select(select_cols)

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
