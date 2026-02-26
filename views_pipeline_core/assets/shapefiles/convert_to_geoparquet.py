"""
One-time conversion script: Shapefiles → GeoParquet.

Reads country and priogrid shapefiles, simplifies geometries,
reprojects to EPSG:4326 (WGS84), and writes GeoParquet files
for efficient loading with PyArrow/GeoArrow.

Usage:
    python convert_to_geoparquet.py
"""

from pathlib import Path
import geopandas as gpd

SCRIPT_DIR = Path(__file__).parent


def convert_country():
    """Convert Natural Earth country shapefile to GeoParquet."""
    shp_path = SCRIPT_DIR / "country" / "ne_110m_admin_0_countries.shp"
    out_path = SCRIPT_DIR / "country" / "ne_110m_admin_0_countries.parquet"

    print(f"Reading {shp_path} ...")
    gdf = gpd.read_file(shp_path)

    # Reproject to WGS84
    gdf = gdf.to_crs(epsg=4326)

    # Keep only essential columns
    gdf = gdf[["ADM0_A3", "geometry"]]

    # Simplify geometries
    gdf["geometry"] = gdf.geometry.simplify(tolerance=0.01, preserve_topology=True)

    print(f"  {len(gdf)} features → writing {out_path}")
    gdf.to_parquet(out_path, index=False)
    print("  Done.")


def convert_priogrid():
    """Convert PRIO-GRID cell shapefile to GeoParquet."""
    shp_path = SCRIPT_DIR / "priogrid" / "priogrid_cell.shp"
    out_path = SCRIPT_DIR / "priogrid" / "priogrid_cell.parquet"

    print(f"Reading {shp_path} ...")
    gdf = gpd.read_file(shp_path)

    # Reproject to WGS84
    gdf = gdf.to_crs(epsg=4326)

    # Keep essential columns
    keep_cols = [c for c in ["gid", "row", "col", "xcoord", "ycoord"] if c in gdf.columns]
    gdf = gdf[keep_cols + ["geometry"]]

    # Simplify geometries (lighter tolerance for small cells)
    gdf["geometry"] = gdf.geometry.simplify(tolerance=0.005, preserve_topology=True)

    print(f"  {len(gdf)} features → writing {out_path}")
    gdf.to_parquet(out_path, index=False)
    print("  Done.")


if __name__ == "__main__":
    convert_country()
    convert_priogrid()
    print("\nAll conversions complete.")
