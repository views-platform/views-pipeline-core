"""Tests for SpatialLevel domain value object."""
import pytest

from views_pipeline_core.domain.spatial import SpatialLevel


class TestFromStr:
    def test_cm(self):
        assert SpatialLevel.from_str("cm") is SpatialLevel.CM

    def test_pgm(self):
        assert SpatialLevel.from_str("pgm") is SpatialLevel.PGM

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unsupported spatial level"):
            SpatialLevel.from_str("invalid")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            SpatialLevel.from_str("")


class TestIndexNames:
    def test_cm_index_names(self):
        assert SpatialLevel.CM.index_names == ("country_id", "month_id")

    def test_pgm_index_names(self):
        assert SpatialLevel.PGM.index_names == ("priogrid_gid", "month_id")


class TestEntityColumn:
    def test_cm_entity_column(self):
        assert SpatialLevel.CM.entity_column == "country_id"

    def test_pgm_entity_column(self):
        assert SpatialLevel.PGM.entity_column == "priogrid_id"


class TestEnumCompleteness:
    def test_exactly_two_members(self):
        assert len(SpatialLevel) == 2

    def test_values(self):
        assert {m.value for m in SpatialLevel} == {"cm", "pgm"}
