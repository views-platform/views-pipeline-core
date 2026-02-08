"""
Metadata Module
===============

Entity metadata management: coordinates, names, country mappings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

from .exceptions import MetadataError, ValidationError


class MetadataModule:
    """Manages entity metadata (coordinates, names, mappings).
    
    Provides geographic information and hierarchical relationships
    between spatial entities (e.g., grids to countries).
    
    Attributes:
        entity_col: Name of the entity ID column.
        REGION_RANGES: GWCode ranges for geographic regions.
    """
    
    REGION_RANGES = {
        "Africa": (400, 626),
        "Middle East": (630, 698),
        "Asia": (700, 999),
        "Europe": (200, 395),
        "Americas": (2, 199),
    }
    
    def __init__(self, entity_col: str):
        """Initialize MetadataModule.
        
        Args:
            entity_col: Name of the entity ID column.
        """
        self.entity_col = entity_col
        self._cache: Optional[pl.DataFrame] = None
        self._country_to_entities: Optional[Dict[int, List[int]]] = None
        self._entity_to_country: Optional[Dict[int, int]] = None
        self._logger = logging.getLogger(f"{__name__}.MetadataModule")
    
    @property
    def is_loaded(self) -> bool:
        """Check if metadata has been loaded."""
        return self._cache is not None
    
    def load_from_file(
        self,
        path: Union[str, Path],
        entity_col: Optional[str] = None,
    ) -> None:
        """Load metadata from file.
        
        Args:
            path: Path to metadata file (parquet or csv).
            entity_col: Override entity column name (optional).
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValidationError: If format unsupported or entity column missing.
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
        
        entity_col = entity_col or self.entity_col
        if entity_col not in self._cache.columns:
            raise ValidationError(
                f"Entity column '{entity_col}' not found",
                details={"available": list(self._cache.columns)}
            )
        
        if "country_id" in self._cache.columns:
            self._build_country_mappings()
    
    def load_from_dataframe(
        self,
        df: pl.DataFrame,
        entity_col: Optional[str] = None,
    ) -> None:
        """Load metadata from DataFrame.
        
        Args:
            df: DataFrame containing metadata.
            entity_col: Override entity column name (optional).
            
        Raises:
            ValidationError: If entity column missing.
        """
        entity_col = entity_col or self.entity_col
        if entity_col not in df.columns:
            raise ValidationError(
                f"Entity column '{entity_col}' not found",
                details={"available": list(df.columns)}
            )
        
        self._cache = df.clone()
        if "country_id" in self._cache.columns:
            self._build_country_mappings()
    
    def _build_country_mappings(self) -> None:
        """Build bidirectional entity-country mappings."""
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
        
        self._logger.debug(
            f"Built mappings: {len(self._entity_to_country)} entities, "
            f"{len(self._country_to_entities)} countries"
        )
    
    def get_entities_for_country(self, country_id: int) -> List[int]:
        """Get all entity IDs belonging to a country.
        
        Args:
            country_id: Country ID.
            
        Returns:
            List of entity IDs.
            
        Raises:
            MetadataError: If country mappings not loaded.
        """
        if self._country_to_entities is None:
            raise MetadataError("Country mappings not loaded.")
        return self._country_to_entities.get(country_id, [])
    
    def get_country_for_entity(self, entity_id: int) -> Optional[int]:
        """Get country ID for an entity.
        
        Args:
            entity_id: Entity ID.
            
        Returns:
            Country ID or None if not found.
        """
        if self._entity_to_country is None:
            return None
        return self._entity_to_country.get(entity_id)
    
    def get_coordinates(
        self,
        entity_ids: Optional[List[int]] = None,
        lat_col: str = "lat",
        lon_col: str = "long",
    ) -> pl.DataFrame:
        """Get geographic coordinates for entities.
        
        Args:
            entity_ids: Filter to specific entities (None = all).
            lat_col: Latitude column name.
            lon_col: Longitude column name.
            
        Returns:
            DataFrame with entity ID and coordinates.
            
        Raises:
            MetadataError: If metadata not loaded.
        """
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
        """Get region classification based on GW codes.
        
        Args:
            entity_ids: Filter to specific entities (None = all).
            gwcode_col: GWCode column name.
            
        Returns:
            DataFrame with entity ID and region.
            
        Raises:
            MetadataError: If metadata not loaded or gwcode column missing.
        """
        if self._cache is None:
            raise MetadataError("Metadata not loaded")
        
        if gwcode_col not in self._cache.columns:
            raise MetadataError(f"GW code column '{gwcode_col}' not found")
        
        # Build region expression
        region_expr = pl.lit("Other")
        for name, (low, high) in self.REGION_RANGES.items():
            region_expr = pl.when(
                (pl.col(gwcode_col) >= low) & (pl.col(gwcode_col) <= high)
            ).then(pl.lit(name)).otherwise(region_expr)
        
        result = self._cache.select([self.entity_col, region_expr.alias("region")])
        if entity_ids is not None:
            result = result.filter(pl.col(self.entity_col).is_in(entity_ids))
        return result
    
    def get_all_countries(self) -> List[int]:
        """Get list of all country IDs.
        
        Returns:
            List of country IDs.
        """
        if self._country_to_entities is None:
            return []
        return list(self._country_to_entities.keys())
    
    def get_all_entities(self) -> List[int]:
        """Get list of all entity IDs.
        
        Returns:
            List of entity IDs.
        """
        if self._cache is None:
            return []
        return self._cache[self.entity_col].unique().to_list()
    
    def __repr__(self) -> str:
        """String representation."""
        if self._cache is None:
            return "MetadataModule(not loaded)"
        
        n_entities = len(self._cache)
        n_countries = len(self._country_to_entities) if self._country_to_entities else 0
        return f"MetadataModule(entities={n_entities}, countries={n_countries})"


__all__ = ["MetadataModule"]
