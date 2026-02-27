import polars as pl
import numpy as np
import geopandas as gpd
from views_pipeline_core.modules.dataset.core import (
    PriogridDataset,
    CountryDataset,
)
import logging
from typing import Union, Optional, List, Dict, Any
from pathlib import Path
import json
import gc

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# OrRd colormap LUT (9 stops, matching matplotlib's OrRd)
# Used for mapping normalized [0, 1] values to RGBA.
# ──────────────────────────────────────────────────────────────
_ORRD_STOPS = np.array(
    [
        [255, 247, 236],  # 0.000
        [254, 232, 200],  # 0.125
        [253, 212, 158],  # 0.250
        [253, 187, 132],  # 0.375
        [252, 141, 89],   # 0.500
        [239, 101, 72],   # 0.625
        [215, 48, 31],    # 0.750
        [179, 0, 0],      # 0.875
        [127, 0, 0],      # 1.000
    ],
    dtype=np.float64,
)
_ORRD_POSITIONS = np.linspace(0, 1, len(_ORRD_STOPS))


def _apply_orrd(values: np.ndarray, alpha: int = 200) -> np.ndarray:
    """
    Map normalised values [0, 1] → RGBA uint8 via the OrRd colormap.

    Args:
        values: 1-D array of float64 in [0, 1]. NaN → transparent.
        alpha: Default alpha channel value (0-255).

    Returns:
        (N, 4) uint8 array of [R, G, B, A].
    """
    out = np.zeros((len(values), 4), dtype=np.uint8)
    valid = ~np.isnan(values)
    v = np.clip(values[valid], 0.0, 1.0)
    r = np.interp(v, _ORRD_POSITIONS, _ORRD_STOPS[:, 0])
    g = np.interp(v, _ORRD_POSITIONS, _ORRD_STOPS[:, 1])
    b = np.interp(v, _ORRD_POSITIONS, _ORRD_STOPS[:, 2])
    out[valid, 0] = np.round(r).astype(np.uint8)
    out[valid, 1] = np.round(g).astype(np.uint8)
    out[valid, 2] = np.round(b).astype(np.uint8)
    out[valid, 3] = alpha
    # NaN → fully transparent
    out[~valid, 3] = 0
    return out


class MappingModule:
    """
    Geographic visualization module for VIEWS datasets.

    Renders GPU-accelerated choropleth maps using deck.gl via standalone HTML.
    Supports both country-level and priogrid-level datasets with automatic
    shapefile handling, temporal animation, and optimized GeoParquet geometry
    storage.
    """

    _COUNTRY_HOVER_COLS = ["country_name"]
    _PRIOGRID_HOVER_COLS = [
        "gid",
        "row",
        "col",
        "country_name",
        "isoab",
        "xcoord",
        "ycoord",
    ]

    # ------------------------------------------------------------------ init
    def __init__(
        self, views_dataset: Union[PriogridDataset, CountryDataset]
    ):
        """
        Initialize mapping module.

        Loads geometry from pre-built GeoParquet files, converts to GeoJSON
        FeatureCollection, and caches attribute tables for fast Polars joins.

        Args:
            views_dataset: ``PriogridDataset`` or ``CountryDataset``.

        Raises:
            ValueError: If dataset is not a valid type.
            FileNotFoundError: If the GeoParquet file is missing.
        """
        self._dataset = views_dataset
        self._entity_id = self._dataset.entity_col
        self._time_id = self._dataset.time_col

        if isinstance(views_dataset, PriogridDataset):
            self._location_col = "gid"
            self._hover_columns = self._PRIOGRID_HOVER_COLS
        elif isinstance(views_dataset, CountryDataset):
            self._location_col = "ADM0_A3"
            self._hover_columns = self._COUNTRY_HOVER_COLS
        else:
            raise ValueError(
                "Invalid dataset type. Must be PriogridDataset or CountryDataset."
            )

        # Load GeoParquet → GeoJSON + attribute table
        self._geojson: Dict[str, Any] = {}          # FeatureCollection dict
        self._attribute_table: pl.DataFrame = None   # non-geometry columns
        self._prepare_geometry()

    # ---------------------------------------------------------- geometry prep
    def _prepare_geometry(self):
        """
        Load GeoParquet, build GeoJSON FeatureCollection and attribute table.

        The GeoJSON is a Python dict kept in memory (serialised to JS at
        render time).  The attribute table is a lightweight Polars DataFrame
        used for joins (no geometry column).
        """
        assets = Path(__file__).parent.parent.parent / "assets" / "shapefiles"

        if isinstance(self._dataset, PriogridDataset):
            parquet_path = assets / "priogrid" / "priogrid_cell.parquet"
            shp_fallback = assets / "priogrid" / "priogrid_cell.shp"
        else:
            parquet_path = assets / "country" / "ne_110m_admin_0_countries.parquet"
            shp_fallback = assets / "country" / "ne_110m_admin_0_countries.shp"

        # Prefer GeoParquet; fall back to shapefile + geopandas
        if parquet_path.exists():
            gdf = gpd.read_parquet(parquet_path)
        else:
            logger.warning(
                "GeoParquet not found at %s — falling back to shapefile. "
                "Run convert_to_geoparquet.py to create it.",
                parquet_path,
            )
            gdf = gpd.read_file(shp_fallback).to_crs(epsg=4326)
            if isinstance(self._dataset, PriogridDataset):
                keep = [c for c in ["gid", "row", "col", "xcoord", "ycoord"] if c in gdf.columns]
                gdf = gdf[keep + ["geometry"]]
                gdf["geometry"] = gdf.geometry.simplify(0.005, preserve_topology=True)
            else:
                gdf = gdf[["ADM0_A3", "geometry"]]
                gdf["geometry"] = gdf.geometry.simplify(0.01, preserve_topology=True)

        # Build attribute table (Polars, no geometry)
        attr_cols = [c for c in gdf.columns if c != "geometry"]
        self._attribute_table = pl.from_pandas(gdf[attr_cols])

        # Build GeoJSON FeatureCollection with `properties.{location_col}` set
        features = []
        for idx, row in gdf.iterrows():
            props = {c: _to_json_safe(row[c]) for c in attr_cols}
            geom = row.geometry.__geo_interface__
            features.append(
                {"type": "Feature", "properties": props, "geometry": geom}
            )
        self._geojson = {"type": "FeatureCollection", "features": features}

        # Compute bounding box for default view state
        total_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        self._center_lon = float((total_bounds[0] + total_bounds[2]) / 2)
        self._center_lat = float((total_bounds[1] + total_bounds[3]) / 2)
        lon_span = total_bounds[2] - total_bounds[0]
        lat_span = total_bounds[3] - total_bounds[1]
        span = max(lon_span, lat_span)
        # rough zoom: 360/span → log2
        self._default_zoom = float(
            max(0.5, np.log2(360.0 / max(span, 1e-6)) - 0.5)
        )

        del gdf
        gc.collect()

    # ------------------------------------------------------- data preparation
    def _prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Join raw dataset with attribute table and metadata (isoab, name).

        Takes a *Polars-native* DataFrame straight from the dataset's
        ``get_subset_dataframe()``, enriches it with shapefile attributes
        and country metadata, and returns a flat Polars DataFrame (no
        geometry column).

        Args:
            df: Polars DataFrame with ``time_col``, ``entity_col``, data cols.

        Returns:
            Enriched ``pl.DataFrame`` with added ``isoab``, ``country_name``,
            and shapefile attribute columns.
        """
        # Select only needed columns (target + index)
        keep = list(self._dataset.target_cols) + [self._entity_id, self._time_id]
        keep = [c for c in keep if c in df.columns]
        df = df.select(keep)

        # Cast numeric to Float32 for memory
        numeric = [
            c
            for c in df.columns
            if c not in (self._entity_id, self._time_id)
            and df.schema[c] in (pl.Float64, pl.Int64, pl.Int32)
        ]
        if numeric:
            df = df.with_columns([pl.col(c).cast(pl.Float32) for c in numeric])

        # Add isoab + country_name from metadata
        iso_df = self._dataset.get_isoab()   # pl.DataFrame [entity_col, isoab]
        name_df = self._dataset.get_name(with_id=True)  # pl.DataFrame [entity_col, name]
        df = df.join(iso_df, on=self._entity_id, how="left")
        df = df.join(name_df, on=self._entity_id, how="left")
        df = df.rename({"name": "country_name"})

        # Join with shapefile attribute table
        if isinstance(self._dataset, CountryDataset):
            # shapefile keyed by ADM0_A3 == isoab
            df = df.join(
                self._attribute_table,
                left_on="isoab",
                right_on="ADM0_A3",
                how="left",
            )
            # Polars drops the right key column; restore it for GeoJSON matching
            if "ADM0_A3" not in df.columns:
                df = df.with_columns(pl.col("isoab").alias("ADM0_A3"))
        elif isinstance(self._dataset, PriogridDataset):
            df = df.join(
                self._attribute_table,
                left_on=self._entity_id,
                right_on="gid",
                how="left",
            )
            # Polars drops the right key column; restore it for GeoJSON matching
            if "gid" not in df.columns:
                df = df.with_columns(pl.col(self._entity_id).alias("gid"))

        return df

    # ------------------------------------------------ public data extraction
    def get_subset_mapping_dataframe(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
    ) -> pl.DataFrame:
        """
        Extract enriched subset of data for visualization (Polars native).

        Args:
            time_ids: Time period(s) to include (``None`` → all).
            entity_ids: Entities to include (``None`` → all).

        Returns:
            ``pl.DataFrame`` with target columns, index columns, metadata,
            and shapefile attribute columns (no geometry — geometry lives in
            the cached GeoJSON).
        """
        df = self._dataset.get_subset_dataframe(
            time_ids=time_ids, entity_ids=entity_ids
        )
        return self._prepare_data(df)

    # --------------------------------------------------------- color helpers
    @staticmethod
    def _compute_color_range(
        values: np.ndarray,
    ) -> tuple[float, float]:
        """0th–98th quantile range on raw values, ignoring NaN."""
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return 0.0, 1.0
        z_min = float(np.quantile(valid, 0.0))
        z_max = float(np.quantile(valid, 0.98))
        if z_max - z_min < 1e-9:
            z_max = z_min + 1.0
        return z_min, z_max

    @staticmethod
    def _normalize(values: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
        """Clip and normalise raw values to [0, 1]."""
        return np.clip((values - z_min) / (z_max - z_min), 0.0, 1.0)

    def _build_color_dict(
        self,
        location_ids: np.ndarray,
        values: np.ndarray,
        z_min: float,
        z_max: float,
    ) -> Dict[str, List[int]]:
        """Map locations → [R,G,B,A] lists given raw values."""
        normed = self._normalize(values, z_min, z_max)
        rgba = _apply_orrd(normed)
        return {
            str(lid): rgba[i].tolist()
            for i, lid in enumerate(location_ids)
        }

    # ------------------------------------------------------------ templates
    @staticmethod
    def _read_template(name: str) -> str:
        """Read an HTML template from the templates/ directory."""
        path = Path(__file__).parent / "templates" / name
        return path.read_text(encoding="utf-8")

    # -------------------------------------------------------- view state
    def _view_state_for_data(
        self, df: pl.DataFrame
    ) -> Dict[str, float]:
        """Compute zoom/center from data extent or fall back to defaults."""
        if (
            isinstance(self._dataset, PriogridDataset)
            and "xcoord" in df.columns
            and "ycoord" in df.columns
        ):
            xs = df["xcoord"].drop_nulls().to_numpy()
            ys = df["ycoord"].drop_nulls().to_numpy()
            if len(xs) > 0:
                lon = float((xs.min() + xs.max()) / 2)
                lat = float((ys.min() + ys.max()) / 2)
                span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
                zoom = float(max(0.5, np.log2(360.0 / span) - 0.5))
                return {"longitude": lon, "latitude": lat, "zoom": zoom}

        return {
            "longitude": self._center_lon,
            "latitude": self._center_lat,
            "zoom": self._default_zoom,
        }

    # ------------------------------------------------------ hover helpers
    def _build_hover_data(self, df: pl.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Build per-location hover tooltip data.

        Returns ``{location_id_str: {attr: value, ...}}`` for every entity
        in the DataFrame.
        """
        # Deduplicate to one row per location
        deduped = df.unique(subset=[self._location_col]) if self._location_col in df.columns else df.unique(subset=[self._entity_id])

        exclude = {"geometry", self._time_id, self._entity_id}
        hover_cols = [
            c
            for c in self._hover_columns
            if c in deduped.columns and c not in exclude
        ]

        # Use location_col if available, else entity_id
        key_col = self._location_col if self._location_col in deduped.columns else self._entity_id

        result: Dict[str, Dict[str, Any]] = {}
        # Materialise as dicts for speed
        for row in deduped.select([key_col] + hover_cols).iter_rows(named=True):
            lid = str(row[key_col])
            result[lid] = {c: _to_json_safe(row[c]) for c in hover_cols}

        return result

    # ----------------------------------------------- interactive (animated)
    def _plot_interactive_map(
        self, mapping_dataframe: pl.DataFrame, target: str
    ) -> str:
        """
        Build animated deck.gl choropleth HTML with time slider.

        Returns a self-contained HTML string.
        """
        loc_col = self._location_col if self._location_col in mapping_dataframe.columns else self._entity_id
        all_times = sorted(mapping_dataframe[self._time_id].unique().to_list())

        # Pivot data: one value per (location, time)
        pivot = (
            mapping_dataframe
            .select([loc_col, self._time_id, target])
            .group_by([loc_col, self._time_id])
            .agg(pl.col(target).first())
        )

        all_locations = pivot[loc_col].unique().sort().to_numpy()

        # Collect all values for global color range
        all_vals = pivot[target].to_numpy().astype(np.float64)
        # Replace None with NaN
        all_vals = np.where(all_vals is None, np.nan, all_vals).astype(np.float64)
        z_min, z_max = self._compute_color_range(all_vals)

        # Build color_data: {time_id_str: {loc_id_str: [r,g,b,a]}}
        # Build value_data: {time_id_str: {loc_id_str: float}}
        color_data: Dict[str, Dict[str, List[int]]] = {}
        value_data: Dict[str, Dict[str, Any]] = {}

        for tid in all_times:
            frame = pivot.filter(pl.col(self._time_id) == tid)
            locs = frame[loc_col].to_numpy()
            vals = frame[target].to_numpy().astype(np.float64)
            color_data[str(tid)] = self._build_color_dict(locs, vals, z_min, z_max)
            value_data[str(tid)] = {
                str(lid): _to_json_safe(float(v)) if not np.isnan(v) else None
                for lid, v in zip(locs, vals)
            }

        hover_data = self._build_hover_data(mapping_dataframe)
        view_state = self._view_state_for_data(mapping_dataframe)

        template = self._read_template("deckgl_animated_map.html")
        html = template.format(
            title=f"{target} — animated map",
            map_height=900,
            geojson_data=json.dumps(self._geojson),
            color_data=json.dumps(color_data),
            hover_data=json.dumps(hover_data),
            value_data=json.dumps(value_data),
            time_ids=json.dumps([_to_json_safe(t) for t in all_times]),
            location_col=self._location_col,
            time_col=self._time_id,
            target=target,
            z_min=z_min,
            z_max=z_max,
            view_state=json.dumps(view_state),
        )

        del pivot, color_data, value_data
        gc.collect()
        return html

    # --------------------------------------------------- static (single t)
    def _plot_static_map(
        self, mapping_dataframe: pl.DataFrame, target: str, time_value: int
    ) -> str:
        """
        Build static deck.gl choropleth HTML for a single time period.

        Returns a self-contained HTML string.
        """
        loc_col = self._location_col if self._location_col in mapping_dataframe.columns else self._entity_id

        frame = mapping_dataframe.filter(pl.col(self._time_id) == time_value)
        locs = frame[loc_col].to_numpy()
        vals = frame[target].to_numpy().astype(np.float64)
        z_min, z_max = self._compute_color_range(vals)

        color_dict = self._build_color_dict(locs, vals, z_min, z_max)
        value_dict = {
            str(lid): _to_json_safe(float(v)) if not np.isnan(v) else None
            for lid, v in zip(locs, vals)
        }

        hover_data = self._build_hover_data(mapping_dataframe)
        view_state = self._view_state_for_data(mapping_dataframe)

        template = self._read_template("deckgl_static_map.html")
        html = template.format(
            title=f"{target} — {self._time_id} {time_value}",
            map_height=900,
            geojson_data=json.dumps(self._geojson),
            color_data=json.dumps(color_dict),
            hover_data=json.dumps(hover_data),
            value_data=json.dumps(value_dict),
            location_col=self._location_col,
            time_col=self._time_id,
            target=target,
            time_value=int(time_value),
            z_min=z_min,
            z_max=z_max,
            view_state=json.dumps(view_state),
        )
        return html

    # -------------------------------------------------------- public API
    def plot_map(
        self,
        mapping_dataframe: pl.DataFrame,
        target: str,
        interactive: bool = False,
        as_html: bool = False,
    ) -> str:
        """
        Generate choropleth map visualization for a given target variable.

        Produces GPU-accelerated deck.gl maps rendered as self-contained HTML.
        Interactive maps include a time slider with play/pause animation.
        Static maps show a single time period.

        Args:
            mapping_dataframe: Data from ``get_subset_mapping_dataframe()``.
            target: Column name to visualise. Must be in dataset targets or features.
            interactive: If ``True``, creates an animated map with time slider.
                If ``False``, creates a single-frame map. Default: ``False``.
            as_html: Kept for API compatibility. The return is always an HTML
                string regardless of this flag.

        Returns:
            Self-contained HTML string suitable for embedding in reports.

        Raises:
            ValueError: If *target* is not in the dataset's targets or features.
            ValueError: If static mode requested with multiple time periods.
        """
        target_options = set(self._dataset.target_cols) | set(
            self._dataset.get_features()
        )
        if target not in target_options:
            raise ValueError(
                f"Target must be a dependent variable or feature. Choose from {target_options}"
            )

        # Unwrap single-element arrays (e.g. from sample distributions)
        if target in mapping_dataframe.columns:
            col = mapping_dataframe[target]
            if col.dtype == pl.List:
                mapping_dataframe = mapping_dataframe.with_columns(
                    pl.col(target).list.first().alias(target)
                )

        if interactive:
            return self._plot_interactive_map(mapping_dataframe, target)
        else:
            time_ids = mapping_dataframe[self._time_id].unique().to_list()
            if len(time_ids) > 1:
                raise ValueError("Static plots require a single time unit")
            return self._plot_static_map(mapping_dataframe, target, time_ids[0])


# ────────────────────────────────────────────────────────────── helpers

def _to_json_safe(v: Any) -> Any:
    """Convert numpy/polars scalars to JSON-serialisable Python types."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if isinstance(v, float):
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    return v
