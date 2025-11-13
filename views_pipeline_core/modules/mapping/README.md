# VIEWS Pipeline Core: Geographic Mapping Module

File: `views_pipeline_core/modules/mapping/mapping.py`  
Primary class: `MappingModule`  
Scope: Country-month (CM) and Priogrid-month (PGM) geographic visualization of targets and features.

## Overview

`MappingModule` provides static (Matplotlib) and interactive (Plotly) choropleth maps for VIEWS datasets. It auto-loads the appropriate shapefile (Natural Earth for countries, PRIO-GRID for cells), merges data with geometry, optimizes memory, and returns ready-to-plot objects or embeddable HTML. Designed for use in reporting, exploratory analysis, and publication.

Supports:
- Time-slice maps (static)
- Temporal animations (interactive)
- Automatic geometry simplification (performance)
- Hover metadata (country/grid attributes)
- ISO enrichment and country name resolution
- Safe handling of array-valued probabilistic predictions

## Core Class: MappingModule

### Initialization

```python
from views_pipeline_core.modules.mapping.mapping import MappingModule
mapper = MappingModule(views_dataset)
```

Args:
- `views_dataset`: Instance of `_PGDataset` or `_CDataset`.

Behavior:
- Detects dataset type.
- Loads matching shapefile.
- Simplifies geometries and builds cached GeoJSON.
- Sets internal location key (`gid` for priogrid, `ADM0_A3` for country).

Raises:
- `ValueError` if dataset type invalid.
- `FileNotFoundError` if shapefile missing.

### Internal Data Model

| Attribute | Description |
|-----------|-------------|
| `_dataset` | Original VIEWS dataset wrapper |
| `_dataframe` | Underlying pandas DataFrame |
| `_entity_id`, `_time_id` | Column names for entity/time indices |
| `_world` | Loaded shapefile (GeoDataFrame) |
| `_base_geojson` | Light GeoJSON prepared for Plotly rendering |
| `_location_col` | Identifier column used for mapping |
| `_hover_columns` | Metadata fields displayed in interactive hover |
| `_priogrid_attributes` / `_country_attributes` | Non-geometry shapefile properties |

### Method Reference

#### `_prepare_base_geojson()`
Converts full-resolution shapefile to EPSG:4326 and simplifies geometry (tolerance=0.01) for fast interactive rendering. Stores minimal property set (location id + geometry).

#### `__get_country_shapefile()`
Loads Natural Earth 1:110m country boundaries. Returns GeoDataFrame with `ADM0_A3` codes.

#### `__get_priogrid_shapefile()`
Loads PRIO-GRID shapefile (~260k cells). Returns GeoDataFrame with grid cell attributes: `gid`, `row`, `col`, coordinates.

#### `__check_missing_geometries(mapping_dataframe, drop_missing_geometries=True)`
Logs and optionally drops rows with missing or empty geometry. Returns cleaned GeoDataFrame.

#### `__init_mapping_dataframe(dataframe)`
Prepares merged GeoDataFrame for visualization:
- Resets index
- Selects target + id columns
- Casts numerics to `float32`
- Adds ISO (`isoab`) and country names
- Merges with shapefile on appropriate key (`isoab` or `gid`)
- Validates geometry

Returns: GeoDataFrame ready for plotting.

#### `__add_isoab(dataframe)`
Enriches provided DataFrame with `isoab` and `country_name` via dataset metadata accessors. Returns updated DataFrame.

#### `get_subset_mapping_dataframe(time_ids=None, entity_ids=None)`
Filters dataset for provided time/entity selection; returns geometry-enriched GeoDataFrame.

Args:
- `time_ids`: int | list | None
- `entity_ids`: int | list | None

Returns: GeoDataFrame (subset + geometry + metadata).

#### `_plot_interactive_map(mapping_dataframe, target)`
Builds Plotly animated choropleth:
- Fixed color scale (global 50–95th quantiles)
- Play/pause controls + slider
- Hover template with location + metadata + time + value

Returns: `plotly.graph_objects.Figure`

#### `_plot_static_map(mapping_dataframe, target, time_unit)`
Builds publication-quality static Matplotlib map for a single time slice:
- Edge outlines
- OrRd colormap
- Horizontal colorbar
- Title includes time unit

Returns: `matplotlib.figure.Figure`

#### `plot_map(mapping_dataframe, target, interactive=False, as_html=False)`
High-level entry point.

Args:
- `mapping_dataframe`: Output from `get_subset_mapping_dataframe`
- `target`: Column to visualize (must be in `dataset.targets` or `dataset.features`)
- `interactive`: If True → Plotly animation; else Matplotlib static
- `as_html`: If True returns HTML string (Plotly full document or base64 PNG)

Returns:
- HTML string OR Matplotlib Figure OR Plotly Figure

Raises:
- `ValueError` if target invalid
- `ValueError` if static plot requested with multiple time periods

### Usage Examples

#### Static Country Map (Single Month)
```python
gdf = mapper.get_subset_mapping_dataframe(time_ids=528)
fig = mapper.plot_map(gdf, target="pred_ged_sb", interactive=False)
fig.savefig("country_map.png", dpi=300)
```

#### Interactive Priogrid Animation
```python
gdf_pg = mapper.get_subset_mapping_dataframe(time_ids=[520, 521, 522, 523])
html = mapper.plot_map(gdf_pg, target="pred_ged_sb", interactive=True, as_html=True)
with open("priogrid_animation.html", "w") as f:
    f.write(html)
```

#### Embedding Plot in Report
```python
from views_pipeline_core.modules.reports import ReportModule
report = ReportModule()
gdf = mapper.get_subset_mapping_dataframe(time_ids=[530, 531, 532])
map_html = mapper.plot_map(gdf, "pred_ged_ns", interactive=True, as_html=True)
report.add_html(map_html)
report.export_as_html("report.html")
```

### Best Practices

| Goal | Recommendation |
|------|----------------|
| Animation | Use ≤ 24 time steps to keep file size reasonable |
| Large priogrid mapping | Filter entities (regions) to reduce geometry load |
| Memory pressure | Let module simplify geometries (default) |
| Probabilistic predictions | If arrays per cell, ensure length=1 for collapse; module handles single-element extraction |
| Publication | Use static map; adjust DPI manually |
| Performance | Avoid repeated initialization—reuse one `MappingModule` instance per dataset |

### Performance Notes

- Geometries simplified early to reduce per-frame overhead.
- Numeric columns cast to `float32`.
- Plotly uses a single base GeoJSON; only z-values & hover data updated per frame.
- Garbage collection invoked after large intermediate structure use.

### Error Handling

| Condition | Response |
|-----------|----------|
| Invalid dataset type | Raises `ValueError` |
| Missing target variable | Raises `ValueError` |
| Static plot with multiple months | Raises `ValueError` |
| Missing geometries after merge | Warns; optionally drops |
| Non-existent shapefile | Underlying `FileNotFoundError` / `OSError` |

### FAQ

| Question | Answer |
|----------|--------|
| Can I map a feature, not a target? | Yes—features are accepted. |
| Why do some countries disappear? | Likely dropped due to missing geometry merge. |
| Are interactive maps offline-capable? | Yes, Plotly JS included inline. |
| Does it support clipping to region? | Not yet—subset by entity IDs first. |
| Multi-target animation? | Only one target per call; create separate maps. |

### Integration Points

| Component | Interaction |
|-----------|-------------|
| Reports Module | Embeds HTML or base64 PNG |
| Evaluation | Spatial QA of prediction ranges |
| Reconciliation Module | Visualize before/after reconciliation values |
| WandB | Export HTML as artifact manually |

### Suggested Workflow (Priogrid Forecasts)

1. Generate predictions (manager).
2. Reconcile to country totals (optional).
3. Initialize `MappingModule` with priogrid dataset.
4. Subset time range.
5. Produce interactive animation.
6. Embed into model report.