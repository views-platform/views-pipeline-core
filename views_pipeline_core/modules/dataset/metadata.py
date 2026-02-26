"""
Metadata Module
===============

Entity metadata management: coordinates, names, country mappings.

Supports fetching metadata via viewser Queryset or loading from files.
All operations use Polars for performance; pandas conversion on request.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import polars as pl
import pandas as pd

from .exceptions import MetadataError, ValidationError

# Optional viewser support
try:
    from viewser import Queryset, Column
    VIEWSER_AVAILABLE = True
except ImportError:
    Queryset = None
    Column = None
    VIEWSER_AVAILABLE = False


def _to_pandas_multiindex(
    df: pl.DataFrame,
    index_cols: List[str],
) -> pd.DataFrame:
    """Convert Polars DataFrame to pandas with MultiIndex.
    
    Args:
        df: Polars DataFrame.
        index_cols: Columns to use as MultiIndex.
        
    Returns:
        Pandas DataFrame with MultiIndex.
    """
    pdf = df.to_pandas()
    existing_cols = [c for c in index_cols if c in pdf.columns]
    if existing_cols:
        pdf.set_index(existing_cols, inplace=True)
        pdf.sort_index(inplace=True)
    return pdf


# =============================================================================
# Region Classification
# =============================================================================

# GW code ranges for geographic regions (Gleditsch & Ward)
REGION_GW_RANGES = {
    "Americas": (2, 165),
    "Europe": (200, 399),
    "Africa": (400, 626),
    "Middle East": (630, 698),
    "Asia": (700, 899),
    "Oceania": (900, 990),
}


def classify_region_from_gwcode(gwcode_col: pl.Expr) -> pl.Expr:
    """Create Polars expression to classify region from GW code.
    
    Args:
        gwcode_col: Polars expression for GW code column.
        
    Returns:
        Polars expression with region classification.
    """
    expr = pl.lit("Other")
    for region, (low, high) in REGION_GW_RANGES.items():
        expr = pl.when(
            (gwcode_col >= low) & (gwcode_col <= high)
        ).then(pl.lit(region)).otherwise(expr)
    return expr


# =============================================================================
# Priogrid Metadata Module
# =============================================================================

class PriogridMetadata:
    """Metadata manager for PRIO-GRID entities.
    
    Fetches and caches metadata including coordinates, country mappings,
    and geographic attributes. Uses Polars for all internal operations.
    
    Example:
        >>> meta = PriogridMetadata(time_col="month_id", entity_col="priogrid_gid")
        >>> meta.fetch()  # Fetch via Queryset
        >>> lat_lon = meta.get_lat_lon(entity_ids=[123, 456])
    """
    
    def __init__(
        self,
        time_col: str = "month_id",
        entity_col: str = "priogrid_gid",
    ):
        """Initialize PriogridMetadata.
        
        Args:
            time_col: Name of time column.
            entity_col: Name of entity column.
        """
        self.time_col = time_col
        self.entity_col = entity_col
        self._cache: Optional[pl.DataFrame] = None
        self._country_to_entities: Optional[Dict[int, List[int]]] = None
        self._logger = logging.getLogger(f"{__name__}.PriogridMetadata")
    
    @property
    def is_loaded(self) -> bool:
        """Check if metadata is loaded."""
        return self._cache is not None
    
    def fetch(self) -> None:
        """Fetch metadata via viewser Queryset.
        
        Raises:
            ImportError: If viewser not installed.
            MetadataError: If fetch fails.
        """
        if self._cache is not None:
            return
        
        if not VIEWSER_AVAILABLE:
            raise ImportError(
                "viewser not installed. Install with: pip install viewser"
            )
        
        self._logger.info("Fetching priogrid metadata via Queryset...")
        
        try:
            pdf = (
                Queryset("pg_metadata", "priogrid_month")
                .with_column(Column("lat", from_loa="priogrid", from_column="latitude"))
                .with_column(Column("long", from_loa="priogrid", from_column="longitude"))
                .with_column(Column("gwcode", from_loa="country", from_column="gwcode"))
                .with_column(Column("row", from_loa="priogrid", from_column="row"))
                .with_column(Column("col", from_loa="priogrid", from_column="col"))
                .with_column(Column("year_id", from_loa="priogrid_year", from_column="year_id"))
                .with_column(Column("isoab", from_loa="country", from_column="isoab"))
                .with_column(Column("name", from_loa="country", from_column="name"))
                .with_column(Column("country_id", from_loa="country_month", from_column="country_id"))
                .publish()
                .fetch()
                .reset_index()
            )
            
            # Normalize column names
            if "priogrid_gid" in pdf.columns and self.entity_col != "priogrid_gid":
                pdf.rename(columns={"priogrid_gid": self.entity_col}, inplace=True)
            
            # Convert to Polars
            self._cache = pl.from_pandas(pdf)
            self._build_country_mappings()
            
            self._logger.info(f"Metadata fetched: {len(self._cache)} records")
            
        except Exception as e:
            raise MetadataError(f"Failed to fetch metadata: {e}")
    
    def load_from_file(self, path: Union[str, Path]) -> None:
        """Load metadata from file.
        
        Args:
            path: Path to parquet or CSV file.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValidationError: If format unsupported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        
        self._logger.info(f"Loading metadata from: {path}")
        
        if path.suffix == ".parquet":
            self._cache = pl.read_parquet(path)
        elif path.suffix == ".csv":
            self._cache = pl.read_csv(path)
        else:
            raise ValidationError(f"Unsupported format: {path.suffix}")
        
        # Normalize column names
        if "priogrid_gid" in self._cache.columns and self.entity_col != "priogrid_gid":
            self._cache = self._cache.rename({"priogrid_gid": self.entity_col})
        
        self._build_country_mappings()
    
    def load_from_dataframe(self, df: Union[pl.DataFrame, pd.DataFrame]) -> None:
        """Load metadata from DataFrame.
        
        Args:
            df: Polars or pandas DataFrame with metadata.
        """
        if isinstance(df, pd.DataFrame):
            self._cache = pl.from_pandas(df.reset_index())
        else:
            self._cache = df.clone()
        
        # Normalize column names
        if "priogrid_gid" in self._cache.columns and self.entity_col != "priogrid_gid":
            self._cache = self._cache.rename({"priogrid_gid": self.entity_col})
        
        self._build_country_mappings()
    
    def _build_country_mappings(self) -> None:
        """Build country-to-entity mapping from cache."""
        if self._cache is None or "country_id" not in self._cache.columns:
            return
        
        # Get unique entity-country pairs (use first occurrence per entity)
        mapping = (
            self._cache
            .select([self.entity_col, "country_id"])
            .unique(subset=[self.entity_col])
        )
        
        self._country_to_entities = {}
        for row in mapping.iter_rows(named=True):
            cid = row["country_id"]
            eid = row[self.entity_col]
            if cid is not None:
                if cid not in self._country_to_entities:
                    self._country_to_entities[cid] = []
                self._country_to_entities[cid].append(eid)
        
        self._logger.debug(
            f"Built mappings: {len(self._country_to_entities)} countries"
        )
    
    def _ensure_loaded(self) -> None:
        """Ensure metadata is loaded, fetching if needed."""
        if self._cache is None:
            self.fetch()
    
    def _join_with_data(
        self,
        data_df: pl.DataFrame,
        columns: List[str],
    ) -> pl.DataFrame:
        """Join metadata columns with data DataFrame.
        
        Args:
            data_df: Data DataFrame with time and entity columns.
            columns: Metadata columns to join.
            
        Returns:
            DataFrame with joined metadata.
        """
        self._ensure_loaded()
        
        # Select needed columns from cache
        select_cols = [self.time_col, self.entity_col] + columns
        available_cols = [c for c in select_cols if c in self._cache.columns]
        
        meta_subset = self._cache.select(available_cols).unique()
        
        # Join on time and entity
        join_cols = [self.time_col, self.entity_col]
        if self.time_col not in meta_subset.columns:
            # Metadata may not have time column; join only on entity
            join_cols = [self.entity_col]
            meta_subset = meta_subset.unique(subset=[self.entity_col])
        
        result = data_df.select(join_cols).unique().join(
            meta_subset,
            on=join_cols,
            how="left",
        )
        
        return result
    
    # -------------------------------------------------------------------------
    # Accessor Methods
    # -------------------------------------------------------------------------
    
    def get_lat_lon(
        self,
        data_df: pl.DataFrame,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get latitude/longitude for entities in data.
        
        Args:
            data_df: Data DataFrame to get coordinates for.
            return_pandas: If True, return pandas with MultiIndex.
            
        Returns:
            DataFrame with lat/lon columns.
        """
        result = self._join_with_data(data_df, ["lat", "long"])
        result = result.rename({"long": "lon"})
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_row_col(
        self,
        data_df: pl.DataFrame,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get PRIO-GRID row/col indices for entities.
        
        Args:
            data_df: Data DataFrame.
            return_pandas: If True, return pandas with MultiIndex.
            
        Returns:
            DataFrame with row/col columns.
        """
        result = self._join_with_data(data_df, ["row", "col"])
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_country_id(
        self,
        data_df: pl.DataFrame,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get country ID for each grid cell.
        
        Args:
            data_df: Data DataFrame.
            return_pandas: If True, return pandas with MultiIndex.
            
        Returns:
            DataFrame with country_id column.
        """
        result = self._join_with_data(data_df, ["country_id"])
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_isoab(
        self,
        data_df: pl.DataFrame,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get ISO alpha-2 country code for each grid.
        
        Args:
            data_df: Data DataFrame.
            return_pandas: If True, return pandas with MultiIndex.
            
        Returns:
            DataFrame with isoab column.
        """
        result = self._join_with_data(data_df, ["isoab"])
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_name(
        self,
        data_df: pl.DataFrame,
        with_id: bool = False,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get country name for each grid.
        
        Args:
            data_df: Data DataFrame.
            with_id: If True, prepend country_id to name.
            return_pandas: If True, return pandas with MultiIndex.
            
        Returns:
            DataFrame with name column.
        """
        if with_id:
            result = self._join_with_data(data_df, ["country_id", "name"])
            result = result.with_columns(
                (pl.col("country_id").cast(pl.Utf8) + " - " + pl.col("name")).alias("name")
            ).drop("country_id")
        else:
            result = self._join_with_data(data_df, ["name"])
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_gwcode(
        self,
        data_df: pl.DataFrame,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get Gleditsch-Ward country code for each grid.
        
        Args:
            data_df: Data DataFrame.
            return_pandas: If True, return pandas with MultiIndex.
            
        Returns:
            DataFrame with gwcode column.
        """
        result = self._join_with_data(data_df, ["gwcode"])
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_region(
        self,
        data_df: pl.DataFrame,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get region classification based on GW codes.
        
        Args:
            data_df: Data DataFrame.
            return_pandas: If True, return pandas with MultiIndex.
            
        Returns:
            DataFrame with region column.
        """
        result = self._join_with_data(data_df, ["gwcode"])
        result = result.with_columns(
            classify_region_from_gwcode(pl.col("gwcode")).alias("region")
        ).drop("gwcode")
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_entities_for_country(self, country_id: int) -> List[int]:
        """Get all entity IDs belonging to a country.
        
        Args:
            country_id: Country ID.
            
        Returns:
            List of entity IDs.
        """
        self._ensure_loaded()
        if self._country_to_entities is None:
            return []
        return self._country_to_entities.get(country_id, [])
    
    def get_grids_for_countries(
        self,
        country_ids: Union[int, List[int]],
    ) -> List[int]:
        """Get all grid IDs belonging to specified countries.
        
        Args:
            country_ids: Single country ID or list of country IDs.
            
        Returns:
            List of grid entity IDs.
        """
        self._ensure_loaded()
        
        if isinstance(country_ids, int):
            country_ids = [country_ids]
        
        if self._country_to_entities is None:
            return []
        
        grid_ids = []
        for cid in country_ids:
            grid_ids.extend(self._country_to_entities.get(cid, []))
        
        return grid_ids
    
    def get_all_countries(self) -> List[int]:
        """Get all country IDs in metadata."""
        self._ensure_loaded()
        if self._country_to_entities is None:
            return []
        return list(self._country_to_entities.keys())
    
    def __repr__(self) -> str:
        if self._cache is None:
            return "PriogridMetadata(not loaded)"
        n_records = len(self._cache)
        n_countries = len(self._country_to_entities) if self._country_to_entities else 0
        return f"PriogridMetadata(records={n_records}, countries={n_countries})"


# =============================================================================
# Country Metadata Module
# =============================================================================

class CountryMetadata:
    """Metadata manager for Country entities.
    
    Fetches and caches country metadata including names, codes,
    capitals, and regional attributes. Uses Polars for all operations.
    
    Example:
        >>> meta = CountryMetadata(time_col="month_id", entity_col="country_id")
        >>> meta.fetch()  # Fetch via Queryset
        >>> names = meta.get_name(data_df)
    """
    
    def __init__(
        self,
        time_col: str = "month_id",
        entity_col: str = "country_id",
    ):
        """Initialize CountryMetadata.
        
        Args:
            time_col: Name of time column.
            entity_col: Name of entity column.
        """
        self.time_col = time_col
        self.entity_col = entity_col
        self._cache: Optional[pl.DataFrame] = None
        self._logger = logging.getLogger(f"{__name__}.CountryMetadata")
    
    @property
    def is_loaded(self) -> bool:
        """Check if metadata is loaded."""
        return self._cache is not None
    
    def fetch(self) -> None:
        """Fetch metadata via viewser Queryset.
        
        Raises:
            ImportError: If viewser not installed.
            MetadataError: If fetch fails.
        """
        if self._cache is not None:
            return
        
        if not VIEWSER_AVAILABLE:
            raise ImportError(
                "viewser not installed. Install with: pip install viewser"
            )
        
        self._logger.info("Fetching country metadata via Queryset...")
        
        try:
            pdf = (
                Queryset("country_metadata", "country_month")
                .with_column(Column("isoab", from_loa="country", from_column="isoab"))
                .with_column(Column("name", from_loa="country", from_column="name"))
                .with_column(Column("gwcode", from_loa="country", from_column="gwcode"))
                .with_column(Column("isonum", from_loa="country", from_column="isonum"))
                .with_column(Column("capname", from_loa="country", from_column="capname"))
                .with_column(Column("caplat", from_loa="country", from_column="caplat"))
                .with_column(Column("caplong", from_loa="country", from_column="caplong"))
                .with_column(Column("in_africa", from_loa="country", from_column="in_africa"))
                .with_column(Column("in_me", from_loa="country", from_column="in_me"))
                .with_column(Column("year_id", from_loa="country_year", from_column="year_id"))
                .publish()
                .fetch()
                .reset_index()
            )
            
            # Convert to Polars and deduplicate — all columns are static
            # per entity, so we only need one row per country_id.
            full = pl.from_pandas(pdf)
            static_cols = [c for c in full.columns if c != self.time_col]
            self._cache = full.select(static_cols).unique(subset=[self.entity_col])
            
            self._logger.info(
                f"Metadata fetched: {len(full)} raw records → {len(self._cache)} unique entities"
            )
            
        except Exception as e:
            raise MetadataError(f"Failed to fetch metadata: {e}")
    
    def load_from_file(self, path: Union[str, Path]) -> None:
        """Load metadata from file.
        
        Args:
            path: Path to parquet or CSV file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        
        self._logger.info(f"Loading metadata from: {path}")
        
        if path.suffix == ".parquet":
            self._cache = pl.read_parquet(path)
        elif path.suffix == ".csv":
            self._cache = pl.read_csv(path)
        else:
            raise ValidationError(f"Unsupported format: {path.suffix}")
    
    def load_from_dataframe(self, df: Union[pl.DataFrame, pd.DataFrame]) -> None:
        """Load metadata from DataFrame."""
        if isinstance(df, pd.DataFrame):
            self._cache = pl.from_pandas(df.reset_index())
        else:
            self._cache = df.clone()
    
    def _ensure_loaded(self) -> None:
        """Ensure metadata is loaded, fetching if needed."""
        if self._cache is None:
            self.fetch()
    
    def _join_with_data(
        self,
        data_df: pl.DataFrame,
        columns: List[str],
    ) -> pl.DataFrame:
        """Join metadata columns with data DataFrame."""
        self._ensure_loaded()
        
        select_cols = [self.time_col, self.entity_col] + columns
        available_cols = [c for c in select_cols if c in self._cache.columns]
        
        meta_subset = self._cache.select(available_cols).unique()
        
        join_cols = [self.time_col, self.entity_col]
        if self.time_col not in meta_subset.columns:
            join_cols = [self.entity_col]
            meta_subset = meta_subset.unique(subset=[self.entity_col])
        
        # Also filter join_cols to what data_df actually has
        join_cols = [c for c in join_cols if c in data_df.columns]
        if not join_cols:
            raise ValueError("No common join columns between data and metadata")
        
        result = data_df.select(join_cols).unique().join(
            meta_subset,
            on=join_cols,
            how="left",
        )
        
        return result
    
    # -------------------------------------------------------------------------
    # Accessor Methods
    # -------------------------------------------------------------------------
    
    def get_isoab(
        self,
        data_df: pl.DataFrame,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get ISO alpha-2 country code."""
        result = self._join_with_data(data_df, ["isoab"])
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_name(
        self,
        data_df: pl.DataFrame,
        with_id: bool = False,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get country name."""
        if with_id:
            result = self._join_with_data(data_df, ["name"])
            result = result.with_columns(
                (pl.col(self.entity_col).cast(pl.Utf8) + " - " + pl.col("name")).alias("name")
            )
        else:
            result = self._join_with_data(data_df, ["name"])
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_gwcode(
        self,
        data_df: pl.DataFrame,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get Gleditsch-Ward country code."""
        result = self._join_with_data(data_df, ["gwcode"])
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_isonum(
        self,
        data_df: pl.DataFrame,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get ISO numeric country code."""
        result = self._join_with_data(data_df, ["isonum"])
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_capname(
        self,
        data_df: pl.DataFrame,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get capital city name."""
        result = self._join_with_data(data_df, ["capname"])
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_cap_lat_lon(
        self,
        data_df: pl.DataFrame,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get capital city coordinates."""
        result = self._join_with_data(data_df, ["caplat", "caplong"])
        result = result.rename({"caplat": "cap_lat", "caplong": "cap_lon"})
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_region(
        self,
        data_df: pl.DataFrame,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get region flags (in_africa, in_me)."""
        result = self._join_with_data(data_df, ["in_africa", "in_me"])
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def get_region_name(
        self,
        data_df: pl.DataFrame,
        return_pandas: bool = False,
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """Get region name based on GW codes."""
        result = self._join_with_data(data_df, ["gwcode"])
        result = result.with_columns(
            classify_region_from_gwcode(pl.col("gwcode")).alias("region")
        ).drop("gwcode")
        
        if return_pandas:
            return _to_pandas_multiindex(result, [self.time_col, self.entity_col])
        return result
    
    def __repr__(self) -> str:
        if self._cache is None:
            return "CountryMetadata(not loaded)"
        return f"CountryMetadata(records={len(self._cache)})"


# =============================================================================
# Legacy MetadataModule (for backward compatibility)
# =============================================================================

class MetadataModule:
    """Legacy metadata module for backward compatibility.
    
    Use PriogridMetadata or CountryMetadata for new code.
    """
    
    REGION_RANGES = REGION_GW_RANGES
    
    def __init__(self, entity_col: str):
        self.entity_col = entity_col
        self._cache: Optional[pl.DataFrame] = None
        self._country_to_entities: Optional[Dict[int, List[int]]] = None
        self._entity_to_country: Optional[Dict[int, int]] = None
        self._logger = logging.getLogger(f"{__name__}.MetadataModule")
    
    @property
    def is_loaded(self) -> bool:
        return self._cache is not None
    
    def load_from_file(self, path: Union[str, Path], entity_col: Optional[str] = None) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        
        if path.suffix == ".parquet":
            self._cache = pl.read_parquet(path)
        elif path.suffix == ".csv":
            self._cache = pl.read_csv(path)
        else:
            raise ValidationError(f"Unsupported format: {path.suffix}")
        
        entity_col = entity_col or self.entity_col
        if entity_col not in self._cache.columns:
            raise ValidationError(f"Entity column '{entity_col}' not found")
        
        if "country_id" in self._cache.columns:
            self._build_country_mappings()
    
    def load_from_dataframe(self, df: pl.DataFrame, entity_col: Optional[str] = None) -> None:
        entity_col = entity_col or self.entity_col
        if entity_col not in df.columns:
            raise ValidationError(f"Entity column '{entity_col}' not found")
        
        self._cache = df.clone()
        if "country_id" in self._cache.columns:
            self._build_country_mappings()
    
    def _build_country_mappings(self) -> None:
        if self._cache is None:
            return
        
        self._entity_to_country = dict(zip(
            self._cache[self.entity_col].to_list(),
            self._cache["country_id"].to_list()
        ))
        
        self._country_to_entities = {}
        for eid, cid in self._entity_to_country.items():
            if cid not in self._country_to_entities:
                self._country_to_entities[cid] = []
            self._country_to_entities[cid].append(eid)
    
    def get_entities_for_country(self, country_id: int) -> List[int]:
        if self._country_to_entities is None:
            raise MetadataError("Country mappings not loaded.")
        return self._country_to_entities.get(country_id, [])
    
    def get_country_for_entity(self, entity_id: int) -> Optional[int]:
        if self._entity_to_country is None:
            return None
        return self._entity_to_country.get(entity_id)
    
    def get_coordinates(
        self,
        entity_ids: Optional[List[int]] = None,
        lat_col: str = "lat",
        lon_col: str = "long",
    ) -> pl.DataFrame:
        if self._cache is None:
            raise MetadataError("Metadata not loaded")
        
        result = self._cache.select([self.entity_col, lat_col, lon_col])
        if entity_ids is not None:
            result = result.filter(pl.col(self.entity_col).is_in(entity_ids))
        return result
    
    def get_region(
        self,
        entity_ids: Optional[List[int]] = None,
        gwcode_col: str = "gwcode",
    ) -> pl.DataFrame:
        if self._cache is None:
            raise MetadataError("Metadata not loaded")
        
        if gwcode_col not in self._cache.columns:
            raise MetadataError(f"GW code column '{gwcode_col}' not found")
        
        result = self._cache.with_columns(
            classify_region_from_gwcode(pl.col(gwcode_col)).alias("region")
        ).select([self.entity_col, "region"])
        
        if entity_ids is not None:
            result = result.filter(pl.col(self.entity_col).is_in(entity_ids))
        return result
    
    def get_all_countries(self) -> List[int]:
        if self._country_to_entities is None:
            return []
        return list(self._country_to_entities.keys())
    
    def get_all_entities(self) -> List[int]:
        if self._cache is None:
            return []
        return self._cache[self.entity_col].unique().to_list()
    
    def __repr__(self) -> str:
        if self._cache is None:
            return "MetadataModule(not loaded)"
        n_entities = len(self._cache)
        n_countries = len(self._country_to_entities) if self._country_to_entities else 0
        return f"MetadataModule(entities={n_entities}, countries={n_countries})"


__all__ = [
    "PriogridMetadata",
    "CountryMetadata",
    "MetadataModule",
    "classify_region_from_gwcode",
    "REGION_GW_RANGES",
    "VIEWSER_AVAILABLE",
]
