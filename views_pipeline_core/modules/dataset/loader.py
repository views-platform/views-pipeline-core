"""
Data Loading Module
===================

Handles data loading from various sources into Polars LazyFrames.
Supports: Polars DataFrames/LazyFrames, Pandas DataFrames, Parquet
files/directories/globs, CSV files.

All inputs are converted to LazyFrame for deferred execution, enabling
predicate pushdown and memory-efficient processing of 100GB+ datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import polars as pl
import pandas as pd

from .exceptions import ValidationError


class LoaderModule:
    """Loads data from any source as a Polars LazyFrame.

    Every input type is normalised to ``pl.LazyFrame`` so that downstream
    modules can build lazy query plans with filter/projection pushdown
    before materialising data.

    Supported sources:
        - ``pl.DataFrame`` → ``.lazy()``
        - ``pl.LazyFrame`` → returned directly
        - ``pd.DataFrame`` → converted to Polars then ``.lazy()``
        - File path (``.parquet``, ``.csv``) → ``scan_parquet`` / ``scan_csv``
        - Directory of parquet files → ``scan_parquet("dir/**/*.parquet")``
        - Glob pattern (``"data/*.parquet"``) → ``scan_parquet(glob)``
    """

    def __init__(self):
        """Initialize LoaderModule."""
        self._logger = logging.getLogger(f"{__name__}.LoaderModule")

    def load(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, str, Path],
    ) -> pl.LazyFrame:
        """Load *data* as a Polars LazyFrame.

        Args:
            data: Any supported data source (see class docstring).

        Returns:
            ``pl.LazyFrame`` ready for lazy query planning.

        Raises:
            FileNotFoundError: If a concrete file/directory path does not exist.
            ValidationError: If the data type or file format is unsupported.
        """
        if isinstance(data, pl.DataFrame):
            self._logger.debug("Wrapping Polars DataFrame as LazyFrame.")
            return data.lazy()

        if isinstance(data, pl.LazyFrame):
            self._logger.debug("Data already a LazyFrame.")
            return data

        if isinstance(data, pd.DataFrame):
            self._logger.debug("Converting Pandas → Polars → LazyFrame.")
            if isinstance(data.index, pd.MultiIndex) or data.index.name is not None:
                data = data.reset_index()
            return pl.from_pandas(data).lazy()

        if isinstance(data, (str, Path)):
            str_path = str(data)
            has_glob = any(c in str_path for c in "*?[]")

            if has_glob:
                self._logger.info(f"Scanning parquet glob: {str_path}")
                return pl.scan_parquet(str_path)

            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Data path not found: {path}")

            if path.is_dir():
                pattern = str(path / "**/*.parquet")
                self._logger.info(f"Scanning parquet directory: {path}")
                return pl.scan_parquet(pattern)

            self._logger.info(f"Scanning file: {path}")
            if path.suffix == ".parquet":
                return pl.scan_parquet(path)
            if path.suffix == ".csv":
                return pl.scan_csv(path)

            raise ValidationError(f"Unsupported file format: {path.suffix}")

        raise ValidationError(f"Unsupported data type: {type(data).__name__}")


__all__ = ["LoaderModule"]
