"""Level-specific dataset subclasses with index-name validation.

Each subclass adds exactly one invariant on the store's dimension names — the
priogrid vs. country entity axis and the month vs. year time axis — layered
through inheritance so a ``PGMDataset`` enforces both the priogrid entity and
the month time in one ``validate_indices`` chain.
"""

from __future__ import annotations

from views_pipeline_core.modules.dataset.base import ViewsDataset


class PGDataset(ViewsDataset):
    """PRIO-GRID dataset: entity dimension must be ``priogrid_id``."""

    def validate_indices(self) -> None:
        super().validate_indices()
        if self._entity_id != "priogrid_id":
            raise ValueError(
                f"PGDataset requires entity dimension 'priogrid_id', got '{self._entity_id}'"
            )


class PGMDataset(PGDataset):
    """PRIO-GRID-month dataset: time dimension must be ``month_id``."""

    def validate_indices(self) -> None:
        super().validate_indices()
        if self._time_id != "month_id":
            raise ValueError(
                f"PGMDataset requires time dimension 'month_id', got '{self._time_id}'"
            )


class PGYDataset(ViewsDataset):
    """PRIO-GRID-year dataset: time ``year_id``, entity ``priogrid_id``."""

    def validate_indices(self) -> None:
        super().validate_indices()
        if self._time_id != "year_id":
            raise ValueError(
                f"PGYDataset requires time dimension 'year_id', got '{self._time_id}'"
            )
        if self._entity_id != "priogrid_id":
            raise ValueError(
                f"PGYDataset requires entity dimension 'priogrid_id', got '{self._entity_id}'"
            )


class CDataset(ViewsDataset):
    """Country dataset: entity dimension must be ``country_id``."""

    def validate_indices(self) -> None:
        super().validate_indices()
        if self._entity_id != "country_id":
            raise ValueError(
                f"CDataset requires entity dimension 'country_id', got '{self._entity_id}'"
            )


class CMDataset(CDataset):
    """Country-month dataset: time dimension must be ``month_id``."""

    def validate_indices(self) -> None:
        super().validate_indices()
        if self._time_id != "month_id":
            raise ValueError(
                f"CMDataset requires time dimension 'month_id', got '{self._time_id}'"
            )


class CYDataset(CDataset):
    """Country-year dataset: time dimension must be ``year_id``."""

    def validate_indices(self) -> None:
        super().validate_indices()
        if self._time_id != "year_id":
            raise ValueError(
                f"CYDataset requires time dimension 'year_id', got '{self._time_id}'"
            )
