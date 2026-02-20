"""
Data Loading Module
===================

Handles data loading from various sources with validation.
Supports: Polars/Pandas DataFrames, Parquet files, CSV files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import polars as pl
import pandas as pd

from .exceptions import ValidationError


class LoaderModule:
    """Handles data loading from various sources with validation.
    
    Supports:
        - Polars DataFrames (direct or cloned)
        - Polars LazyFrames (with optional collection)
        - Pandas DataFrames (converted to Polars)
        - Parquet files (with lazy loading option)
        - CSV files
    
    Attributes:
        use_lazy: If True, uses scan_parquet for large files.
    """
    
    def __init__(self, use_lazy: bool = False):
        """Initialize LoaderModule.
        
        Args:
            use_lazy: If True, use lazy loading for parquet files.
        """
        self.use_lazy = use_lazy
        self._logger = logging.getLogger(f"{__name__}.LoaderModule")
    
    def load(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame, str, Path],
        collect: bool = True,
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Load data from various sources into a Polars DataFrame or LazyFrame.
        
        Args:
            data: Data source - can be:
                - pl.DataFrame: Cloned directly
                - pl.LazyFrame: Collected if collect=True, otherwise returned as-is
                - pd.DataFrame: Converted to Polars
                - str/Path: File path (.parquet or .csv)
            collect: If True, collect LazyFrames immediately. If False, may return
                a LazyFrame for pl.LazyFrame inputs or parquet files with use_lazy=True.
            
        Returns:
            Polars DataFrame if collect=True or input is eager.
            Polars LazyFrame if collect=False and input is lazy or use_lazy=True for parquet.
            
        Raises:
            FileNotFoundError: If file path doesn't exist.
            ValidationError: If data type or file format is unsupported.
        """
        if isinstance(data, pl.DataFrame):
            self._logger.debug("Data provided as Polars DataFrame, cloning.")
            return data.clone()
        
        elif isinstance(data, pl.LazyFrame):
            self._logger.debug("Data provided as Polars LazyFrame.")
            return data.collect() if collect else data
        
        elif isinstance(data, pd.DataFrame):
            self._logger.debug("Converting Pandas DataFrame to Polars.")
            return pl.from_pandas(data)
        
        elif isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            
            self._logger.info(f"Loading data from: {path}")
            
            if path.suffix == ".parquet":
                if self.use_lazy:
                    lf = pl.scan_parquet(path)
                    return lf.collect() if collect else lf
                return pl.read_parquet(path)
            elif path.suffix == ".csv":
                return pl.read_csv(path)
            else:
                raise ValidationError(f"Unsupported file format: {path.suffix}")
        else:
            raise ValidationError(f"Unsupported data type: {type(data).__name__}")


__all__ = ["LoaderModule"]
