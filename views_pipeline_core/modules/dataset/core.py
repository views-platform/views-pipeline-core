"""
Core Dataset Module
===================

Main SpatioTemporalDataset class and specialized variants.

This module provides:
    - SpatioTemporalDataset: Base class for spatiotemporal data
    - CountryDataset: Spatial intermediate for country-level data
    - CountryMonthDataset: Country-month level data
    - PriogridDataset: Spatial intermediate for priogrid-level data
    - PriogridMonthDataset: Priogrid-month level data with reconciliation
    - Factory functions for convenient loading

Supports three data layouts:
    - SCALAR: Single value per (time, entity) cell
    - ARRAY_COLUMN: Distributions as pl.Array/pl.List columns (recommended)
    - ROW_BASED: Distributions as repeated rows with sample_col
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import pandas as pd

from .exceptions import (
    DatasetError,
    ValidationError,
    IntegrityError,
    ReconciliationError,
    MetadataError,
    TensorConversionError,
)
from .loader import LoaderModule
from .index import IndexModule
from .grid import GridModule
from .subset import SubsetModule
from .tensor import TensorBundle, TensorConverter, TensorModule
from .mode import ModeModule
from .statistics import StatisticsModule
from .metadata import (
    MetadataModule,
    PriogridMetadata,
    CountryMetadata,
    VIEWSER_AVAILABLE,
)
from .reconciliation import ReconciliationModule
from .shape import DistributionLayout
from .disk import DiskBackedFrame, DiskWorkspace, PatchStore, MmapTensorStore

# Optional PyTorch support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================

BASE_YEAR = 1980
MONTHS_PER_YEAR = 12
DEFAULT_CACHE_SIZE = 100
MAX_WORKERS_RATIO = 0.75

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def month_id_to_date(month_id: int) -> Tuple[int, int]:
    """Convert VIEWS month_id to (year, month).
    
    Args:
        month_id: VIEWS month identifier (1-indexed from Jan 1980).
        
    Returns:
        Tuple of (year, month) where month is 1-12.
        
    Example:
        >>> month_id_to_date(1)
        (1980, 1)
        >>> month_id_to_date(529)
        (2024, 1)
    """
    year = BASE_YEAR + (month_id - 1) // MONTHS_PER_YEAR
    month = ((month_id - 1) % MONTHS_PER_YEAR) + 1
    return year, month


def date_to_month_id(year: int, month: int) -> int:
    """Convert (year, month) to VIEWS month_id.
    
    Args:
        year: Calendar year.
        month: Month (1-12).
        
    Returns:
        VIEWS month_id.
        
    Raises:
        ValueError: If month not in 1-12.
    """
    if not 1 <= month <= 12:
        raise ValueError(f"Month must be 1-12, got {month}")
    return (year - BASE_YEAR) * MONTHS_PER_YEAR + month


def compute_cache_key(*args, **kwargs) -> str:
    """Compute a stable hash key for caching."""
    key_data = str((args, sorted(kwargs.items())))
    return hashlib.md5(key_data.encode()).hexdigest()[:16]


def polars_to_pandas_multiindex(
    df: pl.DataFrame,
    index_cols: List[str],
) -> pd.DataFrame:
    """Convert Polars DataFrame to Pandas with MultiIndex.
    
    Args:
        df: Polars DataFrame.
        index_cols: Columns to use as MultiIndex.
        
    Returns:
        Pandas DataFrame with MultiIndex.
    """
    pdf = df.to_pandas()
    pdf.set_index(index_cols, inplace=True)
    pdf.sort_index(inplace=True)
    return pdf


def get_optimal_workers(max_workers: Optional[int] = None) -> int:
    """Determine optimal number of worker processes."""
    cpu_count = os.cpu_count() or 4
    auto_workers = max(1, int(cpu_count * MAX_WORKERS_RATIO))
    return min(max_workers, auto_workers) if max_workers else auto_workers


def detect_device() -> str:
    """Detect best available compute device."""
    if not TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# Base Dataset Class
# =============================================================================

class SpatioTemporalDataset:
    """Base class for spatiotemporal datasets with tensor operations.
    
    Supports three data layouts:
        - SCALAR: Single value per (time, entity) cell
        - ARRAY_COLUMN: Distributions as pl.Array/pl.List columns (recommended)
        - ROW_BASED: Distributions as repeated rows with sample_col
        
    The layout is auto-detected from the data structure.
    
    Example:
        >>> # Scalar layout
        >>> dataset = SpatioTemporalDataset(
        ...     data="data.parquet",
        ...     time_col="month_id",
        ...     entity_col="priogrid_gid",
        ...     target_cols=["fatalities"],
        ... )
        
        >>> # Array-column layout (distributions)
        >>> dataset = SpatioTemporalDataset(
        ...     data=df_with_arrays,
        ...     time_col="month_id",
        ...     entity_col="priogrid_gid",
        ... )
        
        >>> # Access as tensor
        >>> tensor = dataset.to_tensor()  # (T, E, S, F)
    """
    
    def __init__(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, str, Path],
        time_col: str,
        entity_col: str,
        sample_col: Optional[str] = None,
        target_cols: Optional[List[str]] = None,
        fix_structure: bool = True,
        auto_broadcast: bool = True,
        cache_tensors: bool = True,
        known_time_ids: Optional[List[int]] = None,
        known_entity_ids: Optional[List[int]] = None,
    ):
        """Initialize SpatioTemporalDataset.

        Data is stored internally as a ``pl.LazyFrame`` for deferred
        execution.  Call ``get_subset_dataframe()`` or ``collect()`` to
        materialise data.

        Args:
            data: Data source (DataFrame, LazyFrame, file, dir, or glob).
            time_col: Name of time index column.
            entity_col: Name of entity index column.
            sample_col: Name of sample column for row-based distributions.
                       Leave None for array-column or scalar layouts.
            target_cols: Target columns for historical mode.
            fix_structure: If True, auto-completes grid at query time.
            auto_broadcast: If True, broadcasts scalars to match array dims.
            cache_tensors: If True, enables tensor caching.
            known_time_ids: Pre-computed unique time IDs.  Pass this when
                the underlying LazyFrame is expensive to evaluate (e.g. a
                large cross-join from an extractor) to avoid materialising
                the full plan at init time.
            known_entity_ids: Pre-computed unique entity IDs.  Same
                rationale as *known_time_ids*.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.warning(
            "SpatioTemporalDataset and its subclasses are in early "
            "development. API may change."
        )

        # Initialize disk workspace — all data lives on disk.
        self._workspace = DiskWorkspace()

        # Initialize modules
        self._loader = LoaderModule()
        self._grid = GridModule()
        self._subset = SubsetModule()
        self._tensor_mod = TensorModule(
            auto_broadcast=auto_broadcast,
            cache_enabled=False,  # disk-backed: no in-memory tensor cache
        )
        self._stats = StatisticsModule()
        self._tensors: MmapTensorStore = self._workspace.tensor_store()
        self._fix_structure = fix_structure

        # Load data and spill to disk.
        self._logger.info(f"Loading data from: {type(data).__name__}")
        self._disk: DiskBackedFrame = self._workspace.frame(data)
        self._lf: pl.LazyFrame = self._disk.lazy_frame

        # Detect distribution layout from schema (no collect)
        cols = self._lf.collect_schema().names()
        potential_data_cols = [
            c for c in cols
            if c not in {time_col, entity_col, sample_col}
        ]
        self._dist_layout = DistributionLayout.detect(
            self._lf,
            sample_col=sample_col,
            data_cols=potential_data_cols,
        )
        self._logger.info(f"Distribution layout: {self._dist_layout}")

        # Cache heavy-column flag before any plan mutation.
        self._has_heavy_cols: bool = SubsetModule.has_heavy_columns(self._lf)

        # Preserve the unsorted scan plan for optimised subset reads.
        # IndexModule.create() adds a deferred sort; _raw_lf keeps the
        # original physical row order so filter_indexed() can use
        # with_row_index / slice to skip irrelevant parquet row groups.
        self._raw_lf: pl.LazyFrame = self._lf

        # Initialize indices (validates columns, adds sort to lazy plan)
        self._index, self._lf = IndexModule.create(
            self._lf, time_col, entity_col,
            sample_col=sample_col,
            dist_layout=self._dist_layout,
        )

        # Cache lightweight index metadata.
        # When the caller already knows the unique IDs (e.g. an
        # extractor that built a cross-join panel), accept them
        # directly to avoid evaluating an expensive LazyFrame plan.
        self._cache_index_metadata(
            known_time_ids=known_time_ids,
            known_entity_ids=known_entity_ids,
        )

        # Detect mode
        has_predictions = any(c.startswith("pred_") for c in cols)
        detected_mode = "forecast" if has_predictions else "historical"

        if detected_mode == "historical" and target_cols is None:
            raise ValidationError(
                "Historical mode detected (no 'pred_' columns). "
                "Provide 'target_cols'."
            )

        self._mode = ModeModule(
            mode=detected_mode,
            target_cols=target_cols if detected_mode == "historical" else None,
        )
        self._mode.validate(self._lf)

        # Log summary
        self._logger.info(
            f"Dataset ready: mode={self.mode}, "
            f"layout={self._dist_layout.layout}, "
            f"{len(self._unique_times)} times \u00d7 "
            f"{len(self._unique_entities)} entities \u00d7 "
            f"{self._dist_layout.sample_size} samples"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_index_metadata(
        self,
        known_time_ids: Optional[List[int]] = None,
        known_entity_ids: Optional[List[int]] = None,
    ) -> None:
        """Cache lightweight index metadata.

        When *known_time_ids* / *known_entity_ids* are supplied the
        expensive collect is skipped entirely — the caller already
        knows the panel dimensions (typical for extractor-produced
        LazyFrames backed by a large cross-join).

        Otherwise, per-column unique collects are performed on the
        pre-sort LazyFrame (``_raw_lf``).  These are cheap for
        Parquet-backed scans because Polars only reads the relevant
        column.

        ``_n_rows`` is always *derived* from the product of unique
        counts rather than counting the full frame.  For a complete
        panel this is exact; for sparse data it is a safe
        overestimate (only used for the ``filter_indexed`` scatter
        heuristic and a size warning).
        """
        tc = self._index.time_col
        ec = self._index.entity_col
        sc = self._index.sample_col

        # ── time IDs ────────────────────────────────────────────────
        if known_time_ids is not None:
            self._unique_times: List[int] = sorted(known_time_ids)
        else:
            self._unique_times = (
                self._raw_lf
                .select(pl.col(tc))
                .unique()
                .sort(tc)
                .collect(engine="streaming")
                .to_series()
                .to_list()
            )

        # ── entity IDs ──────────────────────────────────────────────
        if known_entity_ids is not None:
            self._unique_entities: List[int] = sorted(known_entity_ids)
        else:
            self._unique_entities = (
                self._raw_lf
                .select(pl.col(ec))
                .unique()
                .sort(ec)
                .collect(engine="streaming")
                .to_series()
                .to_list()
            )

        # ── sample IDs (always collected; tiny for row-based) ───────
        self._unique_samples: Optional[List[int]] = None
        if sc:
            self._unique_samples = (
                self._raw_lf
                .select(pl.col(sc))
                .unique()
                .sort(sc)
                .collect(engine="streaming")
                .to_series()
                .to_list()
            )

        # ── row count (derived) ─────────────────────────────────────
        n_samples = len(self._unique_samples) if self._unique_samples else 1
        self._n_rows: int = (
            len(self._unique_times) * len(self._unique_entities) * n_samples
        )

    def _ensure_grid_complete(
        self,
        lf: pl.LazyFrame,
        time_ids: Optional[Union[int, List[int]]],
        entity_ids: Optional[Union[int, List[int]]],
        sample_idx: Optional[Union[int, List[int]]],
    ) -> pl.LazyFrame:
        """Auto-complete grid for a subset query."""
        if time_ids is not None:
            t = [time_ids] if isinstance(time_ids, int) else list(time_ids)
        else:
            t = self._unique_times
        if entity_ids is not None:
            e = [entity_ids] if isinstance(entity_ids, int) else list(entity_ids)
        else:
            e = self._unique_entities
        if sample_idx is not None:
            s = [sample_idx] if isinstance(sample_idx, int) else list(sample_idx)
        else:
            s = self._unique_samples

        skeleton_rows = len(t) * len(e) * (len(s) if s else 1)
        if skeleton_rows > 500_000_000:
            self._logger.warning(
                f"Grid skeleton too large ({skeleton_rows:,} rows), "
                "skipping auto-fix"
            )
            return lf

        grid = pl.DataFrame({self._index.time_col: t}).join(
            pl.DataFrame({self._index.entity_col: e}), how="cross",
        )
        if s is not None and self._index.sample_col:
            grid = grid.join(
                pl.DataFrame({self._index.sample_col: s}), how="cross",
            )

        result = grid.lazy().join(lf, on=self._index.index_cols, how="left")
        # Only fill nulls for numeric columns — leave arrays, strings,
        # and other types untouched to avoid data corruption.
        schema = lf.collect_schema()
        numeric_fill = [
            pl.col(c).fill_null(0.0)
            for c in schema.names()
            if c not in self._index.index_cols_set
            and schema[c].is_numeric()
        ]
        if numeric_fill:
            result = result.with_columns(numeric_fill)
        return result

    def _entity_lookup_df(self) -> pl.DataFrame:
        """Minimal DataFrame with entity IDs for metadata lookups."""
        return pl.DataFrame({self.entity_col: self._unique_entities})

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def time_col(self) -> str:
        """Time column name."""
        return self._index.time_col
    
    @property
    def entity_col(self) -> str:
        """Entity column name."""
        return self._index.entity_col
    
    @property
    def sample_col(self) -> Optional[str]:
        """Sample column name (None for array-column or scalar layouts)."""
        return self._index.sample_col
    
    @property
    def target_cols(self) -> List[str]:
        """Target column names."""
        return self._mode.get_targets(self._lf)
    
    @property
    def mode(self) -> str:
        """Dataset mode ('historical' or 'forecast')."""
        return self._mode.mode
    
    @property
    def distribution_layout(self) -> DistributionLayout:
        """Distribution layout configuration."""
        return self._dist_layout
    
    @property
    def sample_size(self) -> int:
        """Number of samples in distribution."""
        return self._dist_layout.sample_size
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Dataset shape as (n_times, n_entities, n_samples, n_features)."""
        n_features = len(self.get_all_data_cols())
        return (
            len(self._unique_times),
            len(self._unique_entities),
            self._dist_layout.sample_size,
            n_features,
        )

    @property
    def lazy_frame(self) -> pl.LazyFrame:
        """Access the underlying LazyFrame."""
        return self._lf

    @property
    def columns(self) -> List[str]:
        """Column names."""
        return self._lf.collect_schema().names()

    def collect(self) -> pl.DataFrame:
        """Materialise the full dataset as a DataFrame.

        .. warning::
            For large datasets (>10M rows) this may exceed available
            RAM.  Prefer ``get_subset_dataframe()`` for targeted access.
        """
        if self._n_rows > 10_000_000:
            self._logger.warning(
                f"Collecting {self._n_rows:,} rows — consider using "
                "get_subset_dataframe() for large datasets."
            )
        return self._lf.collect(engine="streaming")

    # -------------------------------------------------------------------------
    # Column Accessors
    # -------------------------------------------------------------------------

    def get_features(self) -> List[str]:
        """Get feature column names (excluding targets)."""
        return self._mode.get_features(self._lf, self._index)

    def get_all_data_cols(self) -> List[str]:
        """Get all data columns (excluding indices)."""
        return self._mode.get_all_data_cols(self._lf, self._index)

    def get_pred_vars(self) -> List[str]:
        """Get prediction column names (prefixed with 'pred_')."""
        return [c for c in self._lf.collect_schema().names() if c.startswith("pred_")]
    
    # -------------------------------------------------------------------------
    # Data Access
    # -------------------------------------------------------------------------

    def _get_subset_lazy(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
    ) -> pl.LazyFrame:
        """Build a filtered subset as a LazyFrame (no materialisation).

        All filtering and grid-fix logic is deferred in the lazy plan.
        Call ``.collect(engine="streaming")`` on the result when materialisation is needed.
        """
        has_row_filter = time_ids is not None or entity_ids is not None

        if has_row_filter and self._has_heavy_cols:
            result = self._subset.filter_indexed(
                self._lf, self._raw_lf, self._index, self._n_rows,
                time_ids=time_ids, entity_ids=entity_ids,
                sample_idx=sample_idx, features=features,
            )
        else:
            result = self._subset.filter(
                self._lf, self._index,
                time_ids=time_ids, entity_ids=entity_ids,
                sample_idx=sample_idx, features=features,
            )

        # Auto-fix grid for subset queries
        is_subset = time_ids is not None or entity_ids is not None
        if self._fix_structure and is_subset:
            result = self._ensure_grid_complete(
                result, time_ids, entity_ids, sample_idx,
            )

        return result.sort(self._index.index_cols)

    def get_subset_dataframe(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get filtered subset as DataFrame.

        When the dataset contains heavy columns (``Array`` / ``List``)
        and a row-level filter is requested, a two-phase strategy is
        used: an index-only scan collects matching physical row
        positions first (column-projection pushdown skips the heavy
        data), then only the relevant row groups are read.  For
        entity-sorted parquet files this is orders of magnitude faster.

        Args:
            time_ids: Time IDs to include (None = all).
            entity_ids: Entity IDs to include (None = all).
            sample_idx: Sample indices to include (None = all).
            features: Features to include (None = all).
            return_pandas: If True, return pandas DataFrame with MultiIndex.

        Returns:
            Filtered DataFrame (Polars by default, or Pandas with MultiIndex).
        """
        result = self._get_subset_lazy(
            time_ids=time_ids, entity_ids=entity_ids,
            sample_idx=sample_idx, features=features,
        ).collect(engine="streaming")

        if return_pandas:
            index_cols = [self.time_col, self.entity_col]
            if self.sample_col and self.sample_col in result.columns:
                index_cols.append(self.sample_col)
            return polars_to_pandas_multiindex(result, index_cols)
        return result
    
    def get_subset_tensor(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
    ) -> np.ndarray:
        """Get filtered subset as 4D tensor (Time, Entity, Sample, Feature).
        
        Collects the filtered LazyFrame via the streaming engine and
        converts to a numpy array.
        
        Args:
            time_ids: Time IDs to include.
            entity_ids: Entity IDs to include.
            sample_idx: Sample indices to include.
            features: Features to include.
            
        Returns:
            4D numpy array.
        """
        feat_list = self._resolve_features(features)
        if not feat_list:
            return np.array([])

        sub_lf = self._get_subset_lazy(
            time_ids=time_ids, entity_ids=entity_ids,
            sample_idx=sample_idx, features=feat_list,
        )
        sub_df = sub_lf.collect(engine="streaming")
        return self._tensor_mod.convert(
            sub_df, feat_list, self._index, self._dist_layout
        )
    
    def to_tensor(self, include_targets: bool = True) -> np.ndarray:
        """Convert entire dataset to 4D tensor.
        
        Args:
            include_targets: If True, includes target columns.
            
        Returns:
            4D numpy array with shape (T, E, S, F).
        """
        features = self.get_all_data_cols() if include_targets else self.get_features()
        return self.get_subset_tensor(features=features)
    
    def split_data(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split into features (X) and targets (y).
        
        Args:
            time_ids: Time IDs to include.
            entity_ids: Entity IDs to include.
            sample_idx: Sample indices to include.
            features: Feature columns (targets auto-excluded).
            
        Returns:
            Tuple of (X, y) numpy arrays.
        """
        target_cols = self._mode.get_targets(self._lf)
        if not target_cols:
            raise ValidationError("No target columns found")
        
        feat_list = [
            f for f in self._resolve_features(features) 
            if f not in target_cols
        ]
        
        X = self.get_subset_tensor(time_ids, entity_ids, sample_idx, feat_list)
        y = self.get_subset_tensor(time_ids, entity_ids, sample_idx, target_cols)
        
        return X, y
    
    def _resolve_features(
        self,
        features: Optional[Union[str, List[str]]],
    ) -> List[str]:
        """Resolve feature specification to list."""
        if features is None:
            return self.get_all_data_cols()
        if isinstance(features, str):
            return [features]
        return list(features)
    
    # -------------------------------------------------------------------------
    # Grid Operations
    # -------------------------------------------------------------------------
    
    def fix_space_time_consistency(self, fill_value: Optional[Any] = None) -> None:
        """Fill missing grid points.

        Adds the grid-fix join to the lazy plan (does not collect).

        Args:
            fill_value: Value to fill missing cells (None = keep as null).
        """
        self._lf = self._grid.fix_consistency(
            self._lf, self._index, self._dist_layout, fill_value,
            known_times=self._unique_times,
            known_entities=self._unique_entities,
            known_samples=self._unique_samples,
        )

    def check_grid_completeness(self) -> Tuple[bool, int]:
        """Check if grid has no missing combinations.

        Returns:
            Tuple of (is_complete, missing_count).
        """
        return self._grid.check_completeness(
            self._lf, self._index, self._dist_layout
        )
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def _stats_precompute(
        self,
        features: Optional[Union[str, List[str]]],
        time_ids: Optional[Union[int, List[int]]],
        entity_ids: Optional[Union[int, List[int]]],
        default_features_fn: str = "get_all_data_cols",
    ) -> Tuple[pl.LazyFrame, List[str], pl.DataFrame]:
        """Shared pre-computation for all statistics methods.

        Builds the filtered subset as a **LazyFrame** (no collect) and
        derives the index grid from the filter parameters and cached
        metadata.

        Args:
            features: Caller-supplied features (or None).
            time_ids: Caller-supplied time filter.
            entity_ids: Caller-supplied entity filter.
            default_features_fn: Method name to call for default features
                (``"get_pred_vars"`` falls back to ``"get_all_data_cols"``).

        Returns:
            Tuple of (sub_lf, feature_list, index_df) where *sub_lf* is
            a **LazyFrame**, *feature_list* is the resolved feature names,
            and *index_df* has only the index columns.
        """
        if features is None:
            if default_features_fn == "get_pred_vars":
                features = self.get_pred_vars() or self.get_all_data_cols()
            else:
                features = self.get_all_data_cols()
        if isinstance(features, str):
            features = [features]

        # Keep data lazy – no collect
        sub_lf = self._get_subset_lazy(
            time_ids=time_ids, entity_ids=entity_ids, features=features,
        )

        # Build index_df from filter parameters + cached metadata
        # (avoids a collect just to discover unique values)
        if time_ids is not None:
            times = [time_ids] if isinstance(time_ids, int) else list(time_ids)
        else:
            times = self._unique_times
        if entity_ids is not None:
            entities = [entity_ids] if isinstance(entity_ids, int) else list(entity_ids)
        else:
            entities = self._unique_entities

        index_df = pl.DataFrame({
            self.time_col: [t for t in times for _ in entities],
            self.entity_col: [e for _ in times for e in entities],
        })

        return sub_lf, features, index_df

    def _feature_to_tensor(
        self,
        sub_lf: pl.LazyFrame,
        feature: str,
    ) -> Optional[np.ndarray]:
        """Convert a single feature from a LazyFrame to a 3D tensor
        (Time, Entity, Sample).

        Collects only the index columns and the requested feature
        column via the streaming engine.

        Returns ``None`` if the tensor is empty.
        """
        keep = list(self._index.index_cols) + [feature]
        schema_names = sub_lf.collect_schema().names()
        feat_df = sub_lf.select(
            [c for c in keep if c in schema_names]
        ).collect(engine="streaming")

        tensor = self._tensor_mod.convert(
            feat_df, [feature], self._index, self._dist_layout,
        )
        del feat_df
        if tensor.size == 0:
            return None
        # (T, E, S, 1) -> (T, E, S)
        return tensor.squeeze(axis=-1)

    def calculate_hdi(
        self,
        alpha: float = 0.9,
        features: Optional[Union[str, List[str]]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Calculate Highest Density Interval for distributional data.
        
        Args:
            alpha: Credible mass (e.g., 0.9 for 90% HDI).
            features: Features to compute HDI for.
            time_ids: Filter to specific times.
            entity_ids: Filter to specific entities.
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            DataFrame with HDI bounds.
        """
        sub_lf, features, index_df = self._stats_precompute(
            features, time_ids, entity_ids, default_features_fn="get_pred_vars",
        )

        hdi_columns = {}
        for feature in features:
            data = self._feature_to_tensor(sub_lf, feature)
            if data is None:
                continue
            lower, upper = self._stats.calculate_hdi(data, alpha)
            hdi_columns[f"{feature}_hdi_lower"] = lower.flatten()
            hdi_columns[f"{feature}_hdi_upper"] = upper.flatten()
            del data, lower, upper

        result = index_df.with_columns([
            pl.Series(name=k, values=v) for k, v in hdi_columns.items()
        ])

        if return_pandas:
            pdf = result.to_pandas()
            pdf = pdf.set_index([self.time_col, self.entity_col])
            return pdf
        return result
    
    def calculate_map(
        self,
        features: Optional[Union[str, List[str]]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        enforce_non_negative: bool = False,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Calculate Maximum A Posteriori for distributional data.
        
        Args:
            features: Features to compute MAP for.
            time_ids: Filter to specific times.
            entity_ids: Filter to specific entities.
            enforce_non_negative: Clip negative values to 0.
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            DataFrame with MAP values.
        """
        sub_lf, features, index_df = self._stats_precompute(
            features, time_ids, entity_ids, default_features_fn="get_pred_vars",
        )

        map_columns = {}
        for feature in features:
            data = self._feature_to_tensor(sub_lf, feature)
            if data is None:
                continue
            map_vals = self._stats.calculate_map(data, enforce_non_negative)
            map_columns[f"{feature}_map"] = map_vals.flatten()
            del data, map_vals

        result = index_df.with_columns([
            pl.Series(name=k, values=v) for k, v in map_columns.items()
        ])

        if return_pandas:
            pdf = result.to_pandas()
            pdf = pdf.set_index([self.time_col, self.entity_col])
            return pdf
        return result
    
    def calculate_hdi_map(
        self,
        alpha: float = 0.9,
        features: Optional[Union[str, List[str]]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        enforce_non_negative: bool = False,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Calculate HDI and MAP in a single efficient pass.
        
        More efficient than calling calculate_hdi and calculate_map
        separately as it processes each feature once.
        
        Args:
            alpha: Credible mass (e.g., 0.9 for 90% HDI).
            features: Features to compute for.
            time_ids: Filter to specific times.
            entity_ids: Filter to specific entities.
            enforce_non_negative: Clip negative MAP values to 0.
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            DataFrame with HDI bounds and MAP values.
        """
        sub_lf, features, index_df = self._stats_precompute(
            features, time_ids, entity_ids, default_features_fn="get_pred_vars",
        )

        all_columns = {}
        for feature in features:
            data = self._feature_to_tensor(sub_lf, feature)
            if data is None:
                continue
            lower, upper, map_vals = self._stats.calculate_hdi_map(
                data, alpha, enforce_non_negative
            )
            all_columns[f"{feature}_hdi_lower"] = lower.flatten()
            all_columns[f"{feature}_hdi_upper"] = upper.flatten()
            all_columns[f"{feature}_map"] = map_vals.flatten()
            del data, lower, upper, map_vals

        result = index_df.with_columns([
            pl.Series(name=k, values=v) for k, v in all_columns.items()
        ])

        if return_pandas:
            pdf = result.to_pandas()
            pdf = pdf.set_index([self.time_col, self.entity_col])
            return pdf
        return result
    
    def compute_statistics(
        self,
        features: Optional[Union[str, List[str]]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Compute comprehensive summary statistics.
        
        Args:
            features: Features to compute stats for.
            time_ids: Filter to specific times.
            entity_ids: Filter to specific entities.
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            DataFrame with summary statistics.
        """
        sub_lf, features, index_df = self._stats_precompute(
            features, time_ids, entity_ids,
        )

        stat_columns = {}
        for feature in features:
            data = self._feature_to_tensor(sub_lf, feature)
            if data is None:
                continue
            stats = self._stats.compute_summary_statistics(data)
            for metric_name, metric_arr in stats.items():
                stat_columns[f"{feature}_{metric_name}"] = metric_arr.flatten()
            del data, stats

        if not stat_columns:
            if return_pandas:
                empty_df = pd.DataFrame()
                empty_df.index = pd.MultiIndex.from_tuples(
                    [], names=[self.time_col, self.entity_col]
                )
                return empty_df
            return index_df.head(0)

        result = index_df.with_columns([
            pl.Series(name=k, values=v) for k, v in stat_columns.items()
        ])

        if return_pandas:
            pdf = result.to_pandas()
            pdf = pdf.set_index([self.time_col, self.entity_col])
            return pdf
        return result
    
    # -------------------------------------------------------------------------
    # Integrity
    # -------------------------------------------------------------------------
    
    def check_integrity(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
        max_features: int = 5,
    ) -> bool:
        """Verify tensor conversion works without errors for a subset.
        
        Only tests a small time slice (first 2 months) to validate
        schema and conversion logic without full materialisation.
        
        Args:
            time_ids: Time indices to verify.
            entity_ids: Entity indices to verify.
            sample_idx: Sample indices to verify.
            features: Features to verify.
            max_features: Max features to test when features=None.
        
        Returns:
            True if tensor conversion succeeds.
        """
        try:
            feat_list = self._resolve_features(features)
            if features is None:
                feat_list = feat_list[:max_features]
            
            # Only test a small time window to keep it bounded
            probe_times = time_ids if time_ids is not None else self._unique_times[:2]

            sub_lf = self._get_subset_lazy(
                time_ids=probe_times,
                entity_ids=entity_ids,
                sample_idx=sample_idx,
                features=feat_list,
            )
            sub_df = sub_lf.collect(engine="streaming")
            
            if sub_df.is_empty():
                return True
            
            tensor = self._tensor_mod.convert(
                sub_df, feat_list, self._index, self._dist_layout
            )
            del sub_df
            
            self._logger.debug(f"Integrity check passed: shape {tensor.shape}")
            del tensor
            return True
            
        except Exception as e:
            self._logger.error(f"Integrity check failed: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Multi-Dimensional Tensor Bundle
    # -------------------------------------------------------------------------
    
    def to_tensor_bundle(
        self,
        columns: Optional[List[str]] = None,
        shape_hints: Optional[Dict[str, Tuple[int, ...]]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        dtype: str = "float32",
    ) -> TensorBundle:
        """Convert dataset to TensorBundle for multi-dimensional array support.
        
        Args:
            columns: Columns to convert (None = all data columns).
            shape_hints: Explicit shape overrides {col_name: (dim1, dim2, ...)}.
            time_ids: Filter to specific time indices.
            entity_ids: Filter to specific entity indices.
            dtype: Output numpy dtype.
            
        Returns:
            TensorBundle with properly shaped tensors.
        """
        df = self._get_subset_lazy(
            time_ids=time_ids, entity_ids=entity_ids, features=columns,
        ).collect(engine="streaming")

        converter = TensorConverter(
            time_col=self.time_col,
            entity_col=self.entity_col,
            shape_hints=shape_hints,
            dtype=dtype,
        )
        return converter.convert(df, columns)

    def to_tensor_mmap(
        self,
        features: Optional[List[str]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        dtype: str = "float32",
        name: Optional[str] = None,
    ) -> np.ndarray:
        """Convert dataset to a memory-mapped 4D tensor (T, E, S, F).

        Builds the tensor feature-by-feature so peak RAM never exceeds
        one feature's worth of data (T * E * S). The result is a
        read-only np.memmap backed by a file on disk.

        Args:
            features: Feature columns to include (None = all data columns).
            time_ids: Filter to specific time indices.
            entity_ids: Filter to specific entity indices.
            dtype: Numpy dtype string.
            name: Identifier for the stored tensor (default: auto-generated).

        Returns:
            Read-only np.memmap with shape (T, E, S, F).
        """
        feat_list = self._resolve_features(features)
        if not feat_list:
            raise ValidationError("No features to convert")

        if time_ids is not None:
            times = [time_ids] if isinstance(time_ids, int) else list(time_ids)
        else:
            times = self._unique_times
        if entity_ids is not None:
            entities = [entity_ids] if isinstance(entity_ids, int) else list(entity_ids)
        else:
            entities = self._unique_entities

        n_times = len(times)
        n_entities = len(entities)
        n_samples = self._dist_layout.sample_size
        n_features = len(feat_list)

        tensor_name = name or f"tensor_{n_times}x{n_entities}x{n_samples}x{n_features}"
        shape = (n_times, n_entities, n_samples, n_features)

        logger.info(
            f"Creating mmap tensor '{tensor_name}': shape={shape}, "
            f"dtype={dtype}, ~{np.prod(shape) * np.dtype(dtype).itemsize / 1e9:.2f} GB on disk"
        )

        handle = self._tensors.create(tensor_name, shape=shape, dtype=dtype)

        # Build base lazy plan once — select only index + all requested features.
        # No .sort() overhead: self._lf is already sorted.
        index_cols = list(self._index.index_cols)
        base_lf = self._lf.select(index_cols + feat_list)

        # Apply row filters directly (bypass _get_subset_lazy overhead)
        if time_ids is not None:
            t_list = [time_ids] if isinstance(time_ids, int) else list(time_ids)
            base_lf = base_lf.filter(pl.col(self.time_col).is_in(t_list))
        if entity_ids is not None:
            e_list = [entity_ids] if isinstance(entity_ids, int) else list(entity_ids)
            base_lf = base_lf.filter(pl.col(self.entity_col).is_in(e_list))

        for f_idx, feat in enumerate(feat_list):
            # Collect only index + this feature via streaming engine
            feat_df = base_lf.select(index_cols + [feat]).collect(engine="streaming")
            data = self._tensor_mod.convert(
                feat_df, [feat], self._index, self._dist_layout
            )
            del feat_df
            if data.size == 0:
                logger.warning(f"Feature '{feat}' produced empty tensor, skipping")
                continue
            # data shape: (T, E, S, 1) -> write into feature slice
            handle.write_block(
                data.squeeze(axis=-1),
                (slice(None), slice(None), slice(None), f_idx),
            )
            del data

        return handle.read()

    def get_ml_inputs(
        self,
        feature_cols: List[str],
        target_cols: Optional[List[str]] = None,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
        shape_hints: Optional[Dict[str, Tuple[int, ...]]] = None,
        batch_format: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """Get ML-ready inputs with proper tensor shapes.
        
        Args:
            feature_cols: Feature column names.
            target_cols: Target column names (optional).
            time_ids: Filter to specific time indices.
            entity_ids: Filter to specific entity indices.
            shape_hints: Explicit shape overrides.
            batch_format: If True, flatten T×E to batch dimension.
            
        Returns:
            Tuple of (X_dict, y_dict).
        """
        all_cols = list(feature_cols)
        if target_cols:
            all_cols.extend(target_cols)
        
        bundle = self.to_tensor_bundle(
            columns=all_cols,
            shape_hints=shape_hints,
            time_ids=time_ids,
            entity_ids=entity_ids,
        )
        
        X = {}
        for col in feature_cols:
            if col in bundle.columns:
                X[col] = bundle.get_batch(col) if batch_format else bundle.get(col)
        
        y = None
        if target_cols:
            y = {}
            for col in target_cols:
                if col in bundle.columns:
                    y[col] = bundle.get_batch(col) if batch_format else bundle.get(col)
        
        return X, y
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"mode='{self.mode}', layout='{self._dist_layout.layout}', "
            f"times={len(self._unique_times)}, "
            f"entities={len(self._unique_entities)}, "
            f"samples={self._dist_layout.sample_size})"
        )

    def __len__(self) -> int:
        """Number of rows in dataset."""
        return self._n_rows


# =============================================================================
# Country Dataset (Spatial Intermediate)
# =============================================================================

class CountryDataset(SpatioTemporalDataset):
    """Spatial intermediate for country-level datasets.
    
    Encapsulates all country-specific spatial concerns: entity column
    default, metadata loading, and metadata accessor methods.
    
    Temporal subclasses (CountryMonthDataset, CountryDayDataset, etc.)
    add time-resolution-specific utilities.
    """
    
    DEFAULT_ENTITY_COL = "country_id"
    
    def __init__(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, str, Path],
        time_col: str,
        entity_col: str = DEFAULT_ENTITY_COL,
        sample_col: Optional[str] = None,
        target_cols: Optional[List[str]] = None,
        fix_structure: bool = False,
        auto_broadcast: bool = True,
        cache_tensors: bool = True,
        metadata_path: Optional[Union[str, Path]] = None,
        fetch_metadata: bool = False,
        known_time_ids: Optional[List[int]] = None,
        known_entity_ids: Optional[List[int]] = None,
    ):
        """Initialize CountryDataset.

        Args:
            data: Data source.
            time_col: Time column name (set by temporal subclass).
            entity_col: Entity column name.
            sample_col: Sample column for row-based distributions.
            target_cols: Target columns for historical mode.
            fix_structure: Auto-complete grid at query time.
            auto_broadcast: Broadcast scalars to match arrays.
            cache_tensors: Enable tensor caching.
            metadata_path: Path to country metadata file.
            fetch_metadata: If True, fetch metadata via viewser Queryset.
            known_time_ids: Pre-computed unique time IDs.
            known_entity_ids: Pre-computed unique entity IDs.
        """
        super().__init__(
            data=data, time_col=time_col, entity_col=entity_col,
            sample_col=sample_col, target_cols=target_cols,
            fix_structure=fix_structure, auto_broadcast=auto_broadcast,
            cache_tensors=cache_tensors,
            known_time_ids=known_time_ids,
            known_entity_ids=known_entity_ids,
        )
        
        # Initialize metadata handler
        self._country_meta = CountryMetadata(time_col=time_col, entity_col=entity_col)
        
        if metadata_path:
            self._country_meta.load_from_file(metadata_path)
        elif fetch_metadata:
            self._country_meta.fetch()
    
    # -------------------------------------------------------------------------
    # Metadata Accessors (delegated to CountryMetadata)
    # -------------------------------------------------------------------------
    
    def get_isoab(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get ISO country code (2-letter)."""
        return self._country_meta.get_isoab(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_name(
        self,
        with_id: bool = False,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get country names."""
        return self._country_meta.get_name(self._entity_lookup_df(), with_id=with_id, return_pandas=return_pandas)
    
    def get_gwcode(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get Gleditsch-Ward country code."""
        return self._country_meta.get_gwcode(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_isonum(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get ISO numeric country code."""
        return self._country_meta.get_isonum(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_capname(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get capital city name."""
        return self._country_meta.get_capname(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_cap_lat_lon(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get capital city coordinates."""
        return self._country_meta.get_cap_lat_lon(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_region(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get region information (in_africa, in_me flags)."""
        return self._country_meta.get_region(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_region_name(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get region classification based on GW codes."""
        return self._country_meta.get_region_name(self._entity_lookup_df(), return_pandas=return_pandas)

    # -------------------------------------------------------------------------
    # Reconciliation Support
    # -------------------------------------------------------------------------

    def to_reconciler(
        self,
        feature: str,
        time_id: int,
        country_id: Optional[int] = None,
    ) -> np.ndarray:
        """Extract country values for ForecastReconciler.

        Returns values in natural scale (exp applied to ln_ features)
        with shape (n_samples, n_entities).

        Args:
            feature: Prediction feature to extract.
            time_id: Time ID to extract.
            country_id: Specific country (None = all entities).

        Returns:
            Array of shape (n_samples, n_entities) in natural scale.
        """
        entity_ids = [country_id] if country_id is not None else None

        tensor = self.get_subset_tensor(
            time_ids=[time_id], entity_ids=entity_ids, features=[feature]
        )
        # Shape: (1, n_entities, n_samples, 1) -> (n_samples, n_entities)
        values = tensor.squeeze(axis=(0, 3)).T

        values = ReconciliationModule._transform_for_reconciliation(values, feature)
        return values


# =============================================================================
# Country-Month Dataset
# =============================================================================

class CountryMonthDataset(CountryDataset):
    """Dataset specialized for Country-Month (CM) level data.
    
    Inherits country-level spatial concerns from CountryDataset and
    adds month-resolution date utilities.
    """
    
    DEFAULT_TIME_COL = "month_id"
    
    def __init__(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, str, Path],
        time_col: str = DEFAULT_TIME_COL,
        entity_col: str = CountryDataset.DEFAULT_ENTITY_COL,
        sample_col: Optional[str] = None,
        target_cols: Optional[List[str]] = None,
        fix_structure: bool = False,
        auto_broadcast: bool = True,
        cache_tensors: bool = True,
        metadata_path: Optional[Union[str, Path]] = None,
        fetch_metadata: bool = False,
        known_time_ids: Optional[List[int]] = None,
        known_entity_ids: Optional[List[int]] = None,
    ):
        """Initialize CountryMonthDataset.

        Args:
            data: Data source.
            time_col: Time column name.
            entity_col: Entity column name.
            sample_col: Sample column for row-based distributions.
            target_cols: Target columns for historical mode.
            fix_structure: Auto-complete grid at query time.
            auto_broadcast: Broadcast scalars to match arrays.
            cache_tensors: Enable tensor caching.
            metadata_path: Path to country metadata file.
            fetch_metadata: If True, fetch metadata via viewser Queryset.
            known_time_ids: Pre-computed unique time IDs.
            known_entity_ids: Pre-computed unique entity IDs.
        """
        super().__init__(
            data=data, time_col=time_col, entity_col=entity_col,
            sample_col=sample_col, target_cols=target_cols,
            fix_structure=fix_structure, auto_broadcast=auto_broadcast,
            cache_tensors=cache_tensors,
            metadata_path=metadata_path, fetch_metadata=fetch_metadata,
            known_time_ids=known_time_ids,
            known_entity_ids=known_entity_ids,
        )
    
    # -------------------------------------------------------------------------
    # Date Utilities
    # -------------------------------------------------------------------------
    
    def get_year(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get year for each time ID."""
        times = pl.Series(self.time_col, self._unique_times)
        years = [month_id_to_date(int(t))[0] for t in self._unique_times]
        result = pl.DataFrame({self.time_col: times, "year": years})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_month(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get month-of-year for each time ID."""
        times = pl.Series(self.time_col, self._unique_times)
        months = [month_id_to_date(int(t))[1] for t in self._unique_times]
        result = pl.DataFrame({self.time_col: times, "month": months})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_date(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get full date information (year, month, date string)."""
        data = []
        for t in self._unique_times:
            year, month = month_id_to_date(int(t))
            data.append((t, year, month, f"{year}-{month:02d}-01"))
        result = pl.DataFrame(data, schema=[self.time_col, "year", "month", "date"])
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_quarter(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get quarter for each time ID."""
        times = pl.Series(self.time_col, self._unique_times)
        quarters = [
            (month_id_to_date(int(t))[1] - 1) // 3 + 1 
            for t in self._unique_times
        ]
        result = pl.DataFrame({self.time_col: times, "quarter": quarters})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result


# =============================================================================
# Priogrid Dataset (Spatial Intermediate)
# =============================================================================

class PriogridDataset(SpatioTemporalDataset):
    """Spatial intermediate for priogrid-level datasets.
    
    Encapsulates all priogrid-specific spatial concerns: entity column
    default, metadata/country-mapping loading, metadata accessors,
    country-based subsetting, and hierarchical reconciliation.
    
    Temporal subclasses (PriogridMonthDataset, PriogridDayDataset, etc.)
    add time-resolution-specific utilities.
    """
    
    DEFAULT_ENTITY_COL = "priogrid_id"
    
    def __init__(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, str, Path],
        time_col: str,
        entity_col: str = DEFAULT_ENTITY_COL,
        sample_col: Optional[str] = None,
        target_cols: Optional[List[str]] = None,
        fix_structure: bool = False,
        auto_broadcast: bool = True,
        cache_tensors: bool = True,
        country_mapping: Optional[Union[str, Path, pl.DataFrame]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        fetch_metadata: bool = False,
        known_time_ids: Optional[List[int]] = None,
        known_entity_ids: Optional[List[int]] = None,
    ):
        """Initialize PriogridDataset.

        Args:
            data: Data source.
            time_col: Time column name (set by temporal subclass).
            entity_col: Entity column name.
            sample_col: Sample column for row-based distributions.
            target_cols: Target columns for historical mode.
            fix_structure: Auto-complete grid at query time.
            auto_broadcast: Broadcast scalars to match arrays.
            cache_tensors: Enable tensor caching.
            country_mapping: Grid-to-country mapping (file or DataFrame).
            metadata_path: Path to grid metadata file.
            fetch_metadata: If True, fetch metadata via viewser Queryset.
            known_time_ids: Pre-computed unique time IDs.
            known_entity_ids: Pre-computed unique entity IDs.
        """
        super().__init__(
            data=data, time_col=time_col, entity_col=entity_col,
            sample_col=sample_col, target_cols=target_cols,
            fix_structure=fix_structure, auto_broadcast=auto_broadcast,
            cache_tensors=cache_tensors,
            known_time_ids=known_time_ids,
            known_entity_ids=known_entity_ids,
        )
        
        # Initialize metadata handler
        self._pg_meta = PriogridMetadata(time_col=time_col, entity_col=entity_col)
        
        self._metadata: Optional[MetadataModule] = None
        self._reconciler: Optional[ReconciliationModule] = None
        
        # Patch-based reconciliation store — accumulates small updates
        # on disk without ever collecting the full frame.
        self._patches: Optional[PatchStore] = None
        
        if country_mapping is not None:
            self._load_country_mapping(country_mapping)
        elif metadata_path:
            self._pg_meta.load_from_file(metadata_path)
        elif fetch_metadata:
            self._pg_meta.fetch()
    
    # -------------------------------------------------------------------------
    # Reconciled data (public accessor)
    # -------------------------------------------------------------------------

    @property
    def reconciled_dataframe(self) -> Optional[pl.DataFrame]:
        """Materialise and return the reconciled DataFrame, or None.

        Applies all accumulated patches lazily via anti-join + concat,
        then collects with streaming engine.
        """
        if self._patches is None or not self._patches.has_patches:
            return None
        return self._patches.apply(self._lf).sort(
            self._index.index_cols
        ).collect(engine="streaming")

    @property
    def reconciled_lazy_frame(self) -> Optional[pl.LazyFrame]:
        """Return the reconciled LazyFrame without collecting, or None."""
        if self._patches is None or not self._patches.has_patches:
            return None
        return self._patches.apply(self._lf).sort(self._index.index_cols)

    @reconciled_dataframe.setter
    def reconciled_dataframe(self, value: Optional[pl.DataFrame]) -> None:
        """Accept an eager DataFrame and store as a full patch."""
        if value is None:
            if self._patches is not None:
                self._patches.clear()
            return
        # Replace all existing patches with this full frame as patch
        if self._patches is None:
            self._patches = self._workspace.patch_store(
                [self.time_col, self.entity_col]
            )
        else:
            self._patches.clear()
        if isinstance(value, pl.LazyFrame):
            value = value.collect(engine="streaming")
        self._patches.add_patch(value)
    
    def _load_country_mapping(
        self,
        mapping: Union[str, Path, pl.DataFrame],
    ) -> None:
        """Load grid-to-country mapping.

        The mapping is small (~65k rows × 2 cols) so collecting it
        is acceptable.  We only collect the two required columns.
        """
        if isinstance(mapping, (str, Path)):
            path = Path(mapping)
            if path.suffix == ".parquet":
                mapping_lf = pl.scan_parquet(path)
            else:
                mapping_lf = pl.scan_csv(path)
        elif isinstance(mapping, pl.LazyFrame):
            mapping_lf = mapping
        else:
            mapping_lf = mapping.clone().lazy()
        
        # Handle column name variations
        entity_col = self.entity_col
        mapping_cols = mapping_lf.collect_schema().names()
        if "priogrid_id" in mapping_cols and entity_col not in mapping_cols:
            mapping_lf = mapping_lf.rename({"priogrid_id": entity_col})
            mapping_cols = mapping_lf.collect_schema().names()
        
        required = {entity_col, "country_id"}
        missing = required - set(mapping_cols)
        if missing:
            raise ValidationError(f"Country mapping missing columns: {missing}")
        
        self._metadata = MetadataModule(entity_col)
        # Collect only the columns needed for the mapping (bounded: ~65k rows)
        mapping_df = mapping_lf.select(
            [c for c in mapping_cols if c in required or c == entity_col]
        ).collect(engine="streaming")
        self._metadata.load_from_dataframe(mapping_df, entity_col)
        
        if self._metadata._country_to_entities:
            self._reconciler = ReconciliationModule(self._metadata)
        
        self._logger.info(
            f"Country mapping loaded: {len(self._metadata._entity_to_country)} grids"
        )
    
    # -------------------------------------------------------------------------
    # Metadata Accessors (delegated to PriogridMetadata)
    # -------------------------------------------------------------------------
    
    def get_lat_lon(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get geographic coordinates for grids."""
        return self._pg_meta.get_lat_lon(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_row_col(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get row and column indices for each priogrid."""
        return self._pg_meta.get_row_col(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_country_id(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get country ID for each grid."""
        return self._pg_meta.get_country_id(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_isoab(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get ISO code for the country of each priogrid."""
        return self._pg_meta.get_isoab(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_name(
        self,
        with_id: bool = False,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get country names for each priogrid."""
        return self._pg_meta.get_name(self._entity_lookup_df(), with_id=with_id, return_pandas=return_pandas)
    
    def get_gwcode(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get Gleditsch-Ward country code for each priogrid."""
        return self._pg_meta.get_gwcode(self._entity_lookup_df(), return_pandas=return_pandas)
    
    def get_region(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get region classification based on GW codes."""
        return self._pg_meta.get_region(self._entity_lookup_df(), return_pandas=return_pandas)
    
    # -------------------------------------------------------------------------
    # Country-based Operations
    # -------------------------------------------------------------------------
    
    def get_subset_by_country_id(
        self,
        country_ids: Union[int, List[int]],
        time_ids: Optional[Union[int, List[int]]] = None,
        sample_idx: Optional[Union[int, List[int]]] = None,
        features: Optional[Union[str, List[str]]] = None,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Filter dataset to grids belonging to specified countries.
        
        Args:
            country_ids: Country IDs to include.
            time_ids: Time IDs to include.
            sample_idx: Sample indices to include.
            features: Features to include.
            return_pandas: If True, return pandas DataFrame with MultiIndex.
            
        Returns:
            Filtered DataFrame.
        """
        target_grids = self._pg_meta.get_grids_for_countries(country_ids)
        
        if not target_grids:
            self._logger.warning(f"No grids found for countries: {country_ids}")
            if return_pandas:
                return pd.DataFrame()
            return pl.DataFrame()
        
        return self.get_subset_dataframe(
            time_ids=time_ids, entity_ids=target_grids,
            sample_idx=sample_idx, features=features,
            return_pandas=return_pandas,
        )
    
    # -------------------------------------------------------------------------
    # Reconciliation
    # -------------------------------------------------------------------------
    
    def to_reconciler(
        self,
        feature: str,
        time_id: int,
        country_id: int,
    ) -> Tuple[np.ndarray, List[int]]:
        """Extract grid values for ForecastReconciler.
        
        Extracts values in natural scale (untransformed) for reconciliation.
        
        Args:
            feature: Prediction feature to extract.
            time_id: Time ID to extract.
            country_id: Country ID whose grids to extract.
            
        Returns:
            Tuple of (values, entity_ids) where values has shape (n_samples, n_grids)
            in natural scale (exp applied to ln_ features).
        """
        if self._metadata is None:
            raise ReconciliationError("Country mapping not loaded.")
        
        entity_ids = self._metadata.get_entities_for_country(country_id)
        if not entity_ids:
            raise ReconciliationError(f"No grids for country {country_id}")
        
        tensor = self.get_subset_tensor(
            time_ids=[time_id], entity_ids=entity_ids, features=[feature]
        )
        
        # Shape: (1, n_grids, n_samples, 1) -> (n_samples, n_grids)
        values = tensor.squeeze(axis=(0, 3)).T
        
        # Transform to natural scale for reconciliation
        values = ReconciliationModule._transform_for_reconciliation(values, feature)
        
        return values, entity_ids
    
    def reconcile(
        self,
        country_id: int,
        feature: str,
        reconciled_values: np.ndarray,
        time_id: int,
    ) -> None:
        """Update reconciled_dataframe with reconciled values.
        
        Writes a small patch to disk (only the affected rows) without
        collecting the full dataset.  All patches are applied lazily
        when ``reconciled_dataframe`` or ``reconciled_lazy_frame`` is
        accessed.
        
        Args:
            country_id: Country ID whose grid cells to update.
            feature: Prediction feature to update.
            reconciled_values: Array of reconciled values (n_samples, n_grids)
                              in natural scale (will be log-transformed if needed).
            time_id: Time ID to update.
        """
        if self._metadata is None:
            raise ReconciliationError("Country mapping not loaded.")
        
        # Initialize patch store on first call
        if self._patches is None:
            self._patches = self._workspace.patch_store(
                [self.time_col, self.entity_col]
            )
        
        entity_ids = self._metadata.get_entities_for_country(country_id)
        if not entity_ids:
            raise ReconciliationError(f"No grids for country {country_id}")
        
        if reconciled_values.shape[1] != len(entity_ids):
            raise ReconciliationError(
                f"Values shape {reconciled_values.shape} doesn't match "
                f"{len(entity_ids)} grid cells in country {country_id}"
            )
        
        # Transform back to original scale (log for ln_ features)
        reconciled_values = ReconciliationModule._inverse_transform(
            reconciled_values, feature
        )
        
        # Build a small patch DataFrame — only the affected rows.
        # Vectorized construction: no per-entity loop.
        n_entities = len(entity_ids)
        patch_df = pl.DataFrame({
            self.time_col: [time_id] * n_entities,
            self.entity_col: entity_ids,
            feature: [
                reconciled_values[:, idx].tolist()
                for idx in range(n_entities)
            ],
        })

        # Write patch to disk (tiny: a few hundred rows)
        self._patches.add_patch(patch_df)
        
        self._logger.info(
            f"Reconciled {feature} for country {country_id} at time {time_id} "
            f"({n_entities} grids, patch #{self._patches.n_patches})"
        )


# =============================================================================
# Priogrid-Month Dataset
# =============================================================================

class PriogridMonthDataset(PriogridDataset):
    """Dataset specialized for Priogrid-Month (PGM) level data.
    
    Inherits priogrid-level spatial concerns from PriogridDataset
    (metadata, reconciliation, country-based operations) and adds
    month-resolution date utilities.
    """
    
    DEFAULT_TIME_COL = "month_id"
    
    def __init__(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, str, Path],
        time_col: str = DEFAULT_TIME_COL,
        entity_col: str = PriogridDataset.DEFAULT_ENTITY_COL,
        sample_col: Optional[str] = None,
        target_cols: Optional[List[str]] = None,
        fix_structure: bool = False,
        auto_broadcast: bool = True,
        cache_tensors: bool = True,
        country_mapping: Optional[Union[str, Path, pl.DataFrame]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        fetch_metadata: bool = False,
        known_time_ids: Optional[List[int]] = None,
        known_entity_ids: Optional[List[int]] = None,
    ):
        """Initialize PriogridMonthDataset.

        Args:
            data: Data source.
            time_col: Time column name.
            entity_col: Entity column name.
            sample_col: Sample column for row-based distributions.
            target_cols: Target columns for historical mode.
            fix_structure: Auto-complete grid at query time.
            auto_broadcast: Broadcast scalars to match arrays.
            cache_tensors: Enable tensor caching.
            country_mapping: Grid-to-country mapping (file or DataFrame).
            metadata_path: Path to grid metadata file.
            fetch_metadata: If True, fetch metadata via viewser Queryset.
            known_time_ids: Pre-computed unique time IDs.
            known_entity_ids: Pre-computed unique entity IDs.
        """
        super().__init__(
            data=data, time_col=time_col, entity_col=entity_col,
            sample_col=sample_col, target_cols=target_cols,
            fix_structure=fix_structure, auto_broadcast=auto_broadcast,
            cache_tensors=cache_tensors,
            country_mapping=country_mapping,
            metadata_path=metadata_path, fetch_metadata=fetch_metadata,
            known_time_ids=known_time_ids,
            known_entity_ids=known_entity_ids,
        )
    
    # -------------------------------------------------------------------------
    # Date Utilities
    # -------------------------------------------------------------------------
    
    def get_year(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get year for each time ID."""
        times = pl.Series(self.time_col, self._unique_times)
        years = [month_id_to_date(int(t))[0] for t in self._unique_times]
        result = pl.DataFrame({self.time_col: times, "year": years})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_month(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get month-of-year for each time ID."""
        times = pl.Series(self.time_col, self._unique_times)
        months = [month_id_to_date(int(t))[1] for t in self._unique_times]
        result = pl.DataFrame({self.time_col: times, "month": months})
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result
    
    def get_date(
        self,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get full date information (year, month, date string)."""
        data = []
        for t in self._unique_times:
            year, month = month_id_to_date(int(t))
            data.append((t, year, month, f"{year}-{month:02d}-01"))
        result = pl.DataFrame(data, schema=[self.time_col, "year", "month", "date"])
        if return_pandas:
            return polars_to_pandas_multiindex(result, [self.time_col])
        return result


# =============================================================================
# Factory Functions
# =============================================================================

def load_pgm_dataset(
    path: Union[str, Path],
    target_cols: Optional[List[str]] = None,
    country_mapping: Optional[Union[str, Path]] = None,
    **kwargs,
) -> PriogridMonthDataset:
    """Convenience factory for PriogridMonthDataset.
    
    Args:
        path: Path to data file.
        target_cols: Target column names.
        country_mapping: Grid-to-country mapping file.
        **kwargs: Additional arguments to PriogridMonthDataset.
        
    Returns:
        Configured PriogridMonthDataset.
    """
    return PriogridMonthDataset(
        data=path, target_cols=target_cols,
        country_mapping=country_mapping, **kwargs,
    )


def load_cm_dataset(
    path: Union[str, Path],
    target_cols: Optional[List[str]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> CountryMonthDataset:
    """Convenience factory for CountryMonthDataset.
    
    Args:
        path: Path to data file.
        target_cols: Target column names.
        metadata_path: Path to country metadata file.
        **kwargs: Additional arguments to CountryMonthDataset.
        
    Returns:
        Configured CountryMonthDataset.
    """
    return CountryMonthDataset(
        data=path, target_cols=target_cols,
        metadata_path=metadata_path, **kwargs,
    )


__all__ = [
    # Dataset Classes
    "SpatioTemporalDataset",
    "CountryDataset",
    "CountryMonthDataset", 
    "PriogridDataset",
    "PriogridMonthDataset",
    # Factory Functions
    "load_pgm_dataset",
    "load_cm_dataset",
    # Utilities
    "month_id_to_date",
    "date_to_month_id",
    "compute_cache_key",
    "get_optimal_workers",
    "detect_device",
    "polars_to_pandas_multiindex",
    # Constants
    "BASE_YEAR",
    "MONTHS_PER_YEAR",
    "TORCH_AVAILABLE",
    "VIEWSER_AVAILABLE",
]
