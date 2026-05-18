"""Spatial level domain value object.

Encodes the two spatial resolutions used in VIEWS conflict forecasting:
country-month (CM) and PRIO-GRID-month (PGM).  Each level carries its
canonical MultiIndex layout and entity ID column name.
"""
from enum import Enum


class SpatialLevel(Enum):
    """Spatial resolution of a forecast or dataset."""

    CM = "cm"
    PGM = "pgm"

    @property
    def index_names(self) -> tuple:
        """Canonical pd.MultiIndex names for this level."""
        return _INDEX_NAMES[self]

    @property
    def entity_column(self) -> str:
        """Entity ID column name used in parquet delivery format."""
        return _ENTITY_COLS[self]

    @classmethod
    def from_str(cls, value: str) -> "SpatialLevel":
        """Parse from a config string.

        Raises:
            ValueError: If *value* is not a recognised spatial level.
        """
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(
            f"Unsupported spatial level '{value}'. "
            f"Must be one of: {[m.value for m in cls]}"
        )


_INDEX_NAMES = {
    SpatialLevel.CM: ("country_id", "month_id"),
    SpatialLevel.PGM: ("priogrid_gid", "month_id"),
}

_ENTITY_COLS = {
    SpatialLevel.CM: "country_id",
    SpatialLevel.PGM: "priogrid_id",
}
