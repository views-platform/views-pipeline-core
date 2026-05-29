"""Characterization tests for MappingModule.

These tests verify MappingModule's public contract using mocks to avoid
shapefile I/O. Written BEFORE extraction to views-reporting so they
validate behavior through the re-export shim post-move.
"""

import matplotlib

matplotlib.use("Agg")

import geopandas
import pandas as pd
import plotly
import pytest
from shapely.geometry import box
from unittest.mock import MagicMock, patch

from views_pipeline_core.data.handlers import _CDataset, _PGDataset
from views_pipeline_core.modules.mapping import MappingModule


@pytest.fixture(autouse=True)
def _patch_metadata_functions(monkeypatch):
    """Redirect standalone metadata functions to read mock test data."""
    monkeypatch.setattr(
        "views_reporting.mapping.mapping.get_isoab",
        lambda dataset: dataset._mock_iso_data,
    )
    monkeypatch.setattr(
        "views_reporting.mapping.mapping.get_name",
        lambda dataset, **kwargs: dataset._mock_name_data,
    )


def _make_country_shapefile_gdf():
    """Minimal GeoDataFrame mimicking Natural Earth country shapefile."""
    return geopandas.GeoDataFrame(
        {
            "ADM0_A3": ["NOR", "SWE", "FIN"],
            "geometry": [
                box(5, 58, 15, 71),
                box(11, 55, 24, 69),
                box(20, 59, 32, 70),
            ],
        },
        crs="EPSG:4326",
    )


def _make_priogrid_shapefile_gdf():
    """Minimal GeoDataFrame mimicking PRIO-GRID shapefile."""
    return geopandas.GeoDataFrame(
        {
            "gid": [100, 200, 300],
            "row": [10, 20, 30],
            "col": [5, 6, 7],
            "country_name": ["Norway", "Sweden", "Finland"],
            "isoab": ["NOR", "SWE", "FIN"],
            "xcoord": [10.0, 15.0, 25.0],
            "ycoord": [60.0, 62.0, 64.0],
            "geometry": [
                box(9.5, 59.5, 10.5, 60.5),
                box(14.5, 61.5, 15.5, 62.5),
                box(24.5, 63.5, 25.5, 64.5),
            ],
        },
        crs="EPSG:4326",
    )


def _make_mock_country_dataset():
    """Mock _CDataset with properties MappingModule reads."""
    ds = MagicMock(spec=_CDataset)
    ds.__class__ = _CDataset
    ds._entity_id = "country_id"
    ds._time_id = "month_id"
    ds.targets = ["pred_ged_sb"]
    ds.features = []
    idx = pd.MultiIndex.from_tuples(
        [(528, 1), (528, 2), (528, 3)],
        names=["month_id", "country_id"],
    )
    df = pd.DataFrame({"pred_ged_sb": [0.5, 0.8, 0.3]}, index=idx)
    ds.dataframe = df

    iso_df = pd.DataFrame(
        {
            "month_id": [528, 528, 528],
            "country_id": [1, 2, 3],
            "isoab": ["NOR", "SWE", "FIN"],
        }
    )
    ds._mock_iso_data = iso_df.set_index(["month_id", "country_id"])

    name_df = pd.DataFrame(
        {
            "month_id": [528, 528, 528],
            "country_id": [1, 2, 3],
            "name": ["Norway", "Sweden", "Finland"],
        }
    )
    ds._mock_name_data = name_df.set_index(["month_id", "country_id"])
    ds.get_subset_dataframe.return_value = df
    return ds


def _make_mock_priogrid_dataset():
    """Mock _PGDataset with properties MappingModule reads."""
    ds = MagicMock(spec=_PGDataset)
    ds.__class__ = _PGDataset
    ds._entity_id = "priogrid_gid"
    ds._time_id = "month_id"
    ds.targets = ["pred_ged_sb"]
    ds.features = []
    idx = pd.MultiIndex.from_tuples(
        [(528, 100), (528, 200), (528, 300)],
        names=["month_id", "priogrid_gid"],
    )
    df = pd.DataFrame({"pred_ged_sb": [0.1, 0.5, 0.9]}, index=idx)
    ds.dataframe = df

    iso_df = pd.DataFrame(
        {
            "month_id": [528, 528, 528],
            "priogrid_gid": [100, 200, 300],
            "isoab": ["NOR", "SWE", "FIN"],
        }
    )
    ds._mock_iso_data = iso_df.set_index(["month_id", "priogrid_gid"])

    name_df = pd.DataFrame(
        {
            "month_id": [528, 528, 528],
            "priogrid_gid": [100, 200, 300],
            "name": ["Norway", "Sweden", "Finland"],
        }
    )
    ds._mock_name_data = name_df.set_index(["month_id", "priogrid_gid"])
    ds.get_subset_dataframe.return_value = df
    return ds


def _create_mapping_module_country():
    """Instantiate MappingModule for a country dataset, bypassing shapefile I/O."""
    ds = _make_mock_country_dataset()
    shapefile_gdf = _make_country_shapefile_gdf()
    with (
        patch.object(
            MappingModule,
            "_MappingModule__get_country_shapefile",
            return_value=shapefile_gdf,
        ),
        patch.object(
            MappingModule,
            "_MappingModule__get_priogrid_shapefile",
            return_value=_make_priogrid_shapefile_gdf(),
        ),
    ):
        mapper = MappingModule(ds)
    return mapper, ds


def _create_mapping_module_priogrid():
    """Instantiate MappingModule for a priogrid dataset, bypassing shapefile I/O."""
    ds = _make_mock_priogrid_dataset()
    shapefile_gdf = _make_priogrid_shapefile_gdf()
    with (
        patch.object(
            MappingModule,
            "_MappingModule__get_country_shapefile",
            return_value=_make_country_shapefile_gdf(),
        ),
        patch.object(
            MappingModule,
            "_MappingModule__get_priogrid_shapefile",
            return_value=shapefile_gdf,
        ),
    ):
        mapper = MappingModule(ds)
    return mapper, ds


class TestMappingModuleInitCountry:
    """Verify __init__ behavior for country datasets."""

    def test_location_col(self):
        mapper, _ = _create_mapping_module_country()
        assert mapper._location_col == "ADM0_A3"

    def test_featureidkey(self):
        mapper, _ = _create_mapping_module_country()
        assert mapper._featureidkey == "properties.ADM0_A3"

    def test_hover_columns(self):
        mapper, _ = _create_mapping_module_country()
        assert mapper._hover_columns == ["country_name"]

    def test_country_attributes_exclude_geometry(self):
        mapper, _ = _create_mapping_module_country()
        assert "geometry" not in mapper._country_attributes
        assert "ADM0_A3" in mapper._country_attributes

    def test_base_geojson_created(self):
        mapper, _ = _create_mapping_module_country()
        assert mapper._base_geojson is not None
        assert "features" in mapper._base_geojson


class TestMappingModuleInitPriogrid:
    """Verify __init__ behavior for priogrid datasets."""

    def test_location_col(self):
        mapper, _ = _create_mapping_module_priogrid()
        assert mapper._location_col == "gid"

    def test_featureidkey(self):
        mapper, _ = _create_mapping_module_priogrid()
        assert mapper._featureidkey == "properties.gid"

    def test_hover_columns(self):
        mapper, _ = _create_mapping_module_priogrid()
        assert mapper._hover_columns == MappingModule._PRIOGRID_HOVER_COLS

    def test_priogrid_attributes_exclude_geometry(self):
        mapper, _ = _create_mapping_module_priogrid()
        assert "geometry" not in mapper._priogrid_attributes
        assert "gid" in mapper._priogrid_attributes

    def test_base_geojson_created(self):
        mapper, _ = _create_mapping_module_priogrid()
        assert mapper._base_geojson is not None
        assert "features" in mapper._base_geojson


class TestMappingModuleInitInvalid:
    """Verify rejection of invalid dataset types."""

    def test_non_dataset_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            MappingModule("not a dataset")


class TestPrepareBaseGeojson:
    """Verify GeoJSON preparation."""

    def test_country_geojson_has_adm0_a3(self):
        mapper, _ = _create_mapping_module_country()
        feature_props = [f["properties"] for f in mapper._base_geojson["features"]]
        assert all("ADM0_A3" in p for p in feature_props)

    def test_priogrid_geojson_has_gid(self):
        mapper, _ = _create_mapping_module_priogrid()
        feature_props = [f["properties"] for f in mapper._base_geojson["features"]]
        assert all("gid" in p for p in feature_props)

    def test_geojson_has_geometry(self):
        mapper, _ = _create_mapping_module_country()
        for feature in mapper._base_geojson["features"]:
            assert "geometry" in feature
            assert feature["geometry"] is not None


class TestGetSubsetMappingDataframe:
    """Verify get_subset_mapping_dataframe delegates correctly."""

    def test_calls_get_subset_dataframe(self):
        mapper, ds = _create_mapping_module_country()
        mapper.get_subset_mapping_dataframe(time_ids=528, entity_ids=[1, 2])
        ds.get_subset_dataframe.assert_called_once_with(
            time_ids=528, entity_ids=[1, 2]
        )

    def test_returns_geodataframe(self):
        mapper, _ = _create_mapping_module_country()
        result = mapper.get_subset_mapping_dataframe()
        assert isinstance(result, geopandas.GeoDataFrame)

    def test_result_has_geometry_column(self):
        mapper, _ = _create_mapping_module_country()
        result = mapper.get_subset_mapping_dataframe()
        assert "geometry" in result.columns

    def test_priogrid_returns_geodataframe(self):
        mapper, _ = _create_mapping_module_priogrid()
        result = mapper.get_subset_mapping_dataframe()
        assert isinstance(result, geopandas.GeoDataFrame)


class TestCheckMissingGeometries:
    """Verify geometry validation and cleanup."""

    def test_drops_na_geometries(self):
        mapper, _ = _create_mapping_module_country()
        gdf = geopandas.GeoDataFrame(
            {
                "country_id": [1, 2, 3],
                "isoab": ["NOR", "SWE", "FIN"],
                "geometry": [box(0, 0, 1, 1), None, box(2, 2, 3, 3)],
            },
            crs="EPSG:4326",
        )
        result = mapper._MappingModule__check_missing_geometries(gdf)
        assert len(result) == 2
        assert 2 not in result["country_id"].values

    def test_keeps_all_when_no_missing(self):
        mapper, _ = _create_mapping_module_country()
        gdf = geopandas.GeoDataFrame(
            {
                "country_id": [1, 2],
                "isoab": ["NOR", "SWE"],
                "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)],
            },
            crs="EPSG:4326",
        )
        result = mapper._MappingModule__check_missing_geometries(gdf)
        assert len(result) == 2


class TestPlotMapValidation:
    """Verify plot_map input validation."""

    def test_rejects_invalid_target(self):
        mapper, _ = _create_mapping_module_country()
        gdf = mapper.get_subset_mapping_dataframe()
        with pytest.raises(ValueError, match="Target must be a dependent variable"):
            mapper.plot_map(gdf, target="nonexistent_column")

    def test_static_rejects_multiple_time_periods(self):
        mapper, ds = _create_mapping_module_country()
        idx = pd.MultiIndex.from_tuples(
            [(528, 1), (528, 2), (529, 1), (529, 2)],
            names=["month_id", "country_id"],
        )
        df = pd.DataFrame({"pred_ged_sb": [0.5, 0.8, 0.3, 0.6]}, index=idx)
        ds.get_subset_dataframe.return_value = df

        iso_df = pd.DataFrame(
            {
                "month_id": [528, 528, 529, 529],
                "country_id": [1, 2, 1, 2],
                "isoab": ["NOR", "SWE", "NOR", "SWE"],
            }
        )
        ds._mock_iso_data = iso_df.set_index(["month_id", "country_id"])

        name_df = pd.DataFrame(
            {
                "month_id": [528, 528, 529, 529],
                "country_id": [1, 2, 1, 2],
                "name": ["Norway", "Sweden", "Norway", "Sweden"],
            }
        )
        ds._mock_name_data = name_df.set_index(["month_id", "country_id"])

        gdf = mapper.get_subset_mapping_dataframe()
        with pytest.raises(ValueError, match="Static plots require single time unit"):
            mapper.plot_map(gdf, target="pred_ged_sb", interactive=False)


class TestPlotMapInteractive:
    """Verify interactive (Plotly) map output."""

    def test_returns_plotly_figure(self):
        mapper, _ = _create_mapping_module_country()
        gdf = mapper.get_subset_mapping_dataframe()
        fig = mapper.plot_map(
            gdf, target="pred_ged_sb", interactive=True, as_html=False
        )
        assert isinstance(fig, plotly.graph_objs.Figure)

    def test_returns_html_string(self):
        mapper, _ = _create_mapping_module_country()
        gdf = mapper.get_subset_mapping_dataframe()
        html = mapper.plot_map(
            gdf, target="pred_ged_sb", interactive=True, as_html=True
        )
        assert isinstance(html, str)
        assert "<div" in html.lower()

    def test_figure_has_choropleth_data(self):
        mapper, _ = _create_mapping_module_country()
        gdf = mapper.get_subset_mapping_dataframe()
        fig = mapper.plot_map(
            gdf, target="pred_ged_sb", interactive=True, as_html=False
        )
        assert len(fig.data) > 0
        assert fig.data[0].type == "choropleth"


class TestPlotMapInteractivePriogrid:
    """Verify interactive map for priogrid datasets."""

    def test_returns_plotly_figure(self):
        mapper, _ = _create_mapping_module_priogrid()
        gdf = mapper.get_subset_mapping_dataframe()
        fig = mapper.plot_map(
            gdf, target="pred_ged_sb", interactive=True, as_html=False
        )
        assert isinstance(fig, plotly.graph_objs.Figure)

    def test_returns_html_string(self):
        mapper, _ = _create_mapping_module_priogrid()
        gdf = mapper.get_subset_mapping_dataframe()
        html = mapper.plot_map(
            gdf, target="pred_ged_sb", interactive=True, as_html=True
        )
        assert isinstance(html, str)
        assert "<div" in html.lower()
