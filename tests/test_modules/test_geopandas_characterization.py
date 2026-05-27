"""Characterization tests for geopandas usage in MappingModule.

These tests verify that the geopandas API operations used by
views_pipeline_core/modules/mapping/mapping.py work correctly after
dependency version bumps. The mapping module uses:
  - gpd.GeoDataFrame() constructor with geometry column and CRS
  - gpd.read_file() for shapefile loading
  - GeoDataFrame.merge() for spatial joins
  - GeoDataFrame.geometry attribute access (isna, is_empty)

Written as part of C-101 (Dependabot vulnerability remediation).
"""

import numpy as np
import pandas as pd
import pytest

geopandas = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")


class TestGeoDataFrameConstruction:
    """Verify GeoDataFrame construction patterns used in mapping.py."""

    def test_construct_from_dataframe_with_geometry_column(self):
        """mapping.py:302 constructs GeoDataFrame from merged DataFrame."""
        from shapely.geometry import Point

        df = pd.DataFrame(
            {
                "entity_id": [1, 2, 3],
                "value": [0.5, 0.8, 0.3],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            }
        )
        gdf = geopandas.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        assert isinstance(gdf, geopandas.GeoDataFrame)
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326
        assert len(gdf) == 3
        assert list(gdf.columns) == ["entity_id", "value", "geometry"]

    def test_construct_with_polygon_geometry(self):
        """mapping.py uses polygon geometries (country borders, grid cells)."""
        from shapely.geometry import box

        df = pd.DataFrame(
            {
                "gid": [100, 200],
                "geometry": [box(0, 0, 0.5, 0.5), box(0.5, 0, 1.0, 0.5)],
            }
        )
        gdf = geopandas.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        assert gdf.geometry.geom_type.tolist() == ["Polygon", "Polygon"]
        assert not gdf.geometry.isna().any()
        assert not gdf.geometry.is_empty.any()

    def test_merge_preserves_geodataframe_type(self):
        """mapping.py:296-306 merges data with shapefile, result must be GeoDataFrame."""
        from shapely.geometry import Point

        data_df = pd.DataFrame(
            {"entity_id": [1, 2, 3], "pred_ged_sb": [0.1, 0.5, 0.9]}
        )
        geo_df = geopandas.GeoDataFrame(
            {
                "entity_id": [1, 2, 3],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
            },
            crs="EPSG:4326",
        )

        merged = data_df.merge(geo_df, on="entity_id", how="left")
        result = geopandas.GeoDataFrame(merged, geometry="geometry", crs=geo_df.crs)

        assert isinstance(result, geopandas.GeoDataFrame)
        assert "pred_ged_sb" in result.columns
        assert result.crs.to_epsg() == 4326

    def test_geometry_isna_detection(self):
        """mapping.py:248 checks for missing geometries after merge."""
        from shapely.geometry import Point

        df = pd.DataFrame(
            {
                "entity_id": [1, 2, 3],
                "geometry": [Point(0, 0), None, Point(2, 2)],
            }
        )
        gdf = geopandas.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        assert gdf.geometry.isna().sum() == 1
        assert gdf.geometry.isna().tolist() == [False, True, False]

    def test_float32_columns_survive_geodataframe_construction(self):
        """mapping.py:290 casts numeric columns to float32 before merge."""
        from shapely.geometry import Point

        df = pd.DataFrame(
            {
                "entity_id": [1, 2],
                "value": np.array([0.5, 0.8], dtype=np.float32),
                "geometry": [Point(0, 0), Point(1, 1)],
            }
        )
        gdf = geopandas.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        assert gdf["value"].dtype == np.float32


class TestGeoDataFrameReadFile:
    """Verify gpd.read_file() works (requires shapefile on disk)."""

    def test_read_file_raises_on_missing_path(self):
        """Sanity: gpd.read_file raises for non-existent paths."""
        with pytest.raises(Exception):
            geopandas.read_file("/nonexistent/path/to/shapefile.shp")

    def test_country_shapefile_readable(self):
        """mapping.py:160-167 reads Natural Earth country boundaries."""
        from pathlib import Path

        shapefile_path = (
            Path(__file__).parent.parent.parent
            / "views_pipeline_core"
            / "assets"
            / "shapefiles"
            / "country"
            / "ne_110m_admin_0_countries.shp"
        )
        if not shapefile_path.exists():
            pytest.skip("Country shapefile not available in test environment")

        gdf = geopandas.read_file(shapefile_path)
        assert isinstance(gdf, geopandas.GeoDataFrame)
        assert "geometry" in gdf.columns
        assert "ADM0_A3" in gdf.columns
        assert len(gdf) > 100  # ~177 countries in Natural Earth

    def test_priogrid_shapefile_readable(self):
        """mapping.py:194-201 reads PRIO-GRID cell boundaries."""
        from pathlib import Path

        shapefile_path = (
            Path(__file__).parent.parent.parent
            / "views_pipeline_core"
            / "assets"
            / "shapefiles"
            / "priogrid"
            / "priogrid_cell.shp"
        )
        if not shapefile_path.exists():
            pytest.skip("PRIO-GRID shapefile not available in test environment")

        gdf = geopandas.read_file(shapefile_path)
        assert isinstance(gdf, geopandas.GeoDataFrame)
        assert "geometry" in gdf.columns
        assert "gid" in gdf.columns
        assert len(gdf) > 1000  # ~260k cells but reading full file is slow
