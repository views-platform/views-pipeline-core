"""
Index Management Module
=======================

Manages time, entity, and sample index columns for spatiotemporal datasets.
Works with both ``pl.DataFrame`` and ``pl.LazyFrame``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING

import polars as pl

from .exceptions import ValidationError

if TYPE_CHECKING:
    from .shape import DistributionLayout

# Type alias for either eager or lazy frame
PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]


def _get_columns(frame: PolarsFrame) -> List[str]:
    """Get column names from a DataFrame or LazyFrame."""
    return frame.columns


def _get_schema(frame: PolarsFrame) -> dict:
    """Get schema dict from a DataFrame or LazyFrame."""
    if isinstance(frame, pl.LazyFrame):
        return frame.collect_schema()
    return frame.schema


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
        frame: PolarsFrame,
        time_col: str,
        entity_col: str,
        sample_col: Optional[str] = None,
        dist_layout: Optional["DistributionLayout"] = None,
    ) -> Tuple["IndexModule", PolarsFrame]:
        """Factory method to create IndexModule with validated frame.

        Validates that required columns exist and adds a deferred sort.

        Args:
            frame: Input DataFrame or LazyFrame.
            time_col: Name of time index column.
            entity_col: Name of entity index column.
            sample_col: Name of sample column (only for row-based distributions).
            dist_layout: Distribution layout (auto-detected if None).

        Returns:
            Tuple of (IndexModule, sorted frame — same type as input).

        Raises:
            ValidationError: If required columns are missing.
        """
        schema = _get_schema(frame)

        if time_col not in schema:
            raise ValidationError(
                f"Time column '{time_col}' not found in data.",
                details={"available_columns": list(schema)[:10]},
            )
        if entity_col not in schema:
            raise ValidationError(
                f"Entity column '{entity_col}' not found in data.",
                details={"available_columns": list(schema)[:10]},
            )

        # Determine sample column
        actual_sample_col: Optional[str] = None
        if dist_layout and dist_layout.is_row_based:
            if sample_col and sample_col in schema:
                actual_sample_col = sample_col
            else:
                raise ValidationError(
                    f"Row-based layout detected but sample_col '{sample_col}' "
                    f"not found in data."
                )
        elif sample_col and sample_col in schema:
            actual_sample_col = sample_col

        # Sort — deferred for LazyFrames, immediate for DataFrames
        sort_cols = [time_col, entity_col]
        if actual_sample_col:
            sort_cols = [actual_sample_col] + sort_cols

        frame = frame.sort(sort_cols)

        return (
            cls(
                time_col=time_col,
                entity_col=entity_col,
                sample_col=actual_sample_col,
                _auto_generated_sample=False,
            ),
            frame,
        )

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Index queries (collect when needed)
    # -----------------------------------------------------------------

    def get_unique_values(
        self,
        frame: PolarsFrame,
    ) -> Tuple[List[Any], List[Any], Optional[List[Any]]]:
        """Extract unique sorted values for each index dimension.

        For LazyFrames this triggers a lightweight collect of only the
        index columns.

        Returns:
            Tuple of (times, entities, samples) where *samples* is ``None``
            if the dataset has no sample column.
        """
        cols = [self.time_col, self.entity_col]
        if self.sample_col:
            cols.append(self.sample_col)

        if isinstance(frame, pl.LazyFrame):
            meta = frame.select(cols).unique().collect()
        else:
            meta = frame.select(cols).unique()

        times = meta[self.time_col].unique().sort().to_list()
        entities = meta[self.entity_col].unique().sort().to_list()

        if self.sample_col and self.sample_col in meta.columns:
            samples = meta[self.sample_col].unique().sort().to_list()
            return times, entities, samples

        return times, entities, None

    def get_dimensions(
        self,
        frame: PolarsFrame,
        dist_layout: Optional["DistributionLayout"] = None,
    ) -> Tuple[int, int, int]:
        """Get grid dimensions as (n_times, n_entities, n_samples).

        Args:
            frame: DataFrame or LazyFrame to measure.
            dist_layout: Distribution layout for the sample dimension.

        Returns:
            Tuple of (n_times, n_entities, n_samples).
        """
        times, entities, samples = self.get_unique_values(frame)
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
