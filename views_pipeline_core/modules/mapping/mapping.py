import pandas as pd
import numpy as np
from views_pipeline_core.data.handlers import (
    _CDataset,
    _PGDataset,
)
import logging
from typing import Union, Optional, List, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import gc

logger = logging.getLogger(__name__)


class MappingModule:
    """
    Geographic visualization module for VIEWS datasets.
    
    Provides interactive and static choropleth mapping for both country-level
    and priogrid-level datasets with automatic shapefile handling and optimized
    rendering.
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

    def __init__(self, views_dataset: Union[_PGDataset, _CDataset]):
        """
        Initialize mapping module with VIEWS dataset and load appropriate shapefiles.

        Sets up geographic infrastructure including shapefile loading, coordinate
        reference system configuration, and GeoJSON preparation for efficient
        rendering.

        Args:
            views_dataset: Dataset to visualize. Either:
                - _PGDataset: Priogrid-level data with cell-based geography
                - _CDataset: Country-level data with national boundaries

        Raises:
            ValueError: If dataset is not _PGDataset or _CDataset instance
            FileNotFoundError: If required shapefile is missing

        Example:
            >>> from views_pipeline_core.data.handlers import PGMDataset
            >>> dataset = PGMDataset(predictions_df)
            >>> mapper = MappingModule(dataset)
            >>> print(mapper._location_col)
            'gid'

        Note:
            - Automatically detects dataset type and loads correct shapefile
            - Simplifies geometries to reduce file size
            - Prepares base GeoJSON for faster subsequent renders
            - For PGM: Uses priogrid_cell.shp with ~260k cells
            - For CM: Uses Natural Earth 1:110m country boundaries
        """
        self._dataset = views_dataset
        self._dataframe = self._dataset.dataframe
        self._entity_id = self._dataset._entity_id
        self._time_id = self._dataset._time_id

        if isinstance(views_dataset, _PGDataset):
            self._world = self.__get_priogrid_shapefile()
            self._location_col = "gid"
            self._featureidkey = "properties.gid"
            # Get all available priogrid attributes (excluding geometry)
            self._priogrid_attributes = [
                col for col in self._world.columns if col != "geometry"
            ]
            self._hover_columns = self._PRIOGRID_HOVER_COLS
        elif isinstance(views_dataset, _CDataset):
            self._world = self.__get_country_shapefile()
            self._location_col = "ADM0_A3"
            self._featureidkey = "properties.ADM0_A3"
            # Get all available country attributes (excluding geometry)
            self._country_attributes = [
                col for col in self._world.columns if col != "geometry"
            ]
            self._hover_columns = self._COUNTRY_HOVER_COLS
        else:
            raise ValueError("Invalid dataset type. Must be a _PGDataset or _CDataset.")

        self._mapping_dataframe = None
        self._base_geojson = None
        self._prepare_base_geojson()  # Initialize base GeoJSON

    def _prepare_base_geojson(self):
        """
        Create optimized GeoJSON representation for efficient map rendering.

        Converts shapefile to WGS84 projection, retains only essential properties,
        and simplifies geometries to reduce file size while preserving topology.

        Internal Use:
            Called by __init__() during module initialization.

        Note:
            - Converts to EPSG:4326 (WGS84) for web compatibility
            - Simplifies geometries with 0.01 degree tolerance
            - Memory freed immediately after processing
        """
        base_gdf = self._world.to_crs(epsg=4326).copy()

        # Keep only essential properties to reduce size
        if isinstance(self._dataset, _PGDataset):
            base_gdf = base_gdf[["gid", "geometry"]]
        elif isinstance(self._dataset, _CDataset):
            # For country datasets, keep ADM0_A3 (which matches isoab) and geometry
            base_gdf = base_gdf[["ADM0_A3", "geometry"]]
        else:
            raise ValueError("Invalid dataset type. Must be a _PGDataset or _CDataset.")

        # Simplify geometries to reduce file size
        base_gdf["geometry"] = base_gdf.geometry.simplify(
            tolerance=0.01, preserve_topology=True
        )

        self._base_geojson = base_gdf.__geo_interface__

        # Free memory
        del base_gdf
        gc.collect()

    def __get_country_shapefile(self):
        """
        Load Natural Earth country boundaries shapefile.

        Internal Use:
            Called by __init__() for country-level datasets.

        Returns:
            geopandas.GeoDataFrame: Country boundaries with attributes:
                - ADM0_A3: ISO 3-letter country code
                - geometry: Country polygon/multipolygon
                - Additional Natural Earth attributes

        Raises:
            FileNotFoundError: If shapefile doesn't exist at expected path
            OSError: If shapefile cannot be read

        Note:
            - Uses Natural Earth 1:110m resolution (simplified)
            - Suitable for global-scale visualization
            - Path: assets/shapefiles/country/ne_110m_admin_0_countries.shp
        """
        path = (
            Path(__file__).parent.parent.parent
            / "assets"
            / "shapefiles"
            / "country"
            / "ne_110m_admin_0_countries.shp"
        )
        world = gpd.read_file(path)

        # Ensure ADM0_A3 column exists and is properly formatted
        # if 'ADM0_A3' not in world.columns:
        #     # Try to find the ISO code column with a different name
        #     iso_cols = [col for col in world.columns if 'iso' in col.lower() or 'a3' in col.lower()]
        #     if iso_cols:
        #         world = world.rename(columns={iso_cols[0]: 'ADM0_A3'})
        #     else:
        #         # If no ISO code column found, create one from the index
        #         world['ADM0_A3'] = world.index.astype(str)

        return world

    def __get_priogrid_shapefile(self):
        """
        Load PRIO-GRID cell boundaries shapefile.

        Internal Use:
            Called by __init__() for priogrid-level datasets.

        Returns:
            geopandas.GeoDataFrame: Grid cell boundaries with attributes:
                - gid: Grid cell identifier
                - row, col: Grid coordinates
                - geometry: Cell polygon (0.5° × 0.5°)
                - Additional PRIO-GRID attributes

        Raises:
            FileNotFoundError: If shapefile doesn't exist at expected path
            OSError: If shapefile cannot be read

        Note:
            - PRIO-GRID cells are 0.5° × 0.5° (~55km at equator)
            - Global coverage with ~260,000 cells
            - Path: assets/shapefiles/priogrid/priogrid_cell.shp
        """
        path = (
            Path(__file__).parent.parent.parent
            / "assets"
            / "shapefiles"
            / "priogrid"
            / "priogrid_cell.shp"
        )
        return gpd.read_file(path)

    def __check_missing_geometries(
        self, mapping_dataframe: pd.DataFrame, drop_missing_geometries: bool = True
    ):
        """
        Validate geometries and optionally remove invalid rows.

        Identifies rows with missing or empty geometries and either removes them
        or logs warnings about their presence.

        Internal Use:
            Called by __init_mapping_dataframe() during data preparation.

        Args:
            mapping_dataframe: GeoDataFrame to validate
            drop_missing_geometries: If True, removes rows with invalid geometries.
                If False, only logs warnings.

        Returns:
            pd.DataFrame: Cleaned GeoDataFrame (or original if drop=False)

        Note:
            - Logs unique ISO codes for missing geometries
            - Reports number of dropped rows and their IDs
            - Missing geometries typically indicate:
              - Data outside shapefile coverage
              - Mismatched entity IDs
              - Corrupt geometry data
        """
        missing = mapping_dataframe[
            mapping_dataframe.geometry.is_empty | mapping_dataframe.geometry.isna()
        ]
        if not missing.empty:
            logger.warning(f"Missing geometries for: {missing['isoab'].unique()}")
        if drop_missing_geometries:
            initial_count = len(mapping_dataframe)
            cleaned_gdf = mapping_dataframe[
                (~mapping_dataframe.geometry.is_empty)
                & (~mapping_dataframe.geometry.isna())
            ].copy()

            dropped_count = initial_count - len(cleaned_gdf)
            if dropped_count > 0:
                logger.warning(
                    f"Dropped {dropped_count} rows with missing geometries. "
                    f"Remaining: {len(cleaned_gdf)} rows. "
                    f"Missing IDs: {mapping_dataframe[self._entity_id][mapping_dataframe.geometry.isna()].unique().tolist()}"
                )
            return cleaned_gdf
        return mapping_dataframe

    def __init_mapping_dataframe(self, dataframe: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Prepare GeoDataFrame by merging data with geometries and metadata.

        Processes input DataFrame by selecting relevant columns, adding geographic
        identifiers (ISO codes, country names), merging with shapefiles, and
        validating geometries.

        Internal Use:
            Called by get_subset_mapping_dataframe() to prepare visualization data.

        Args:
            dataframe: Input DataFrame with predictions/data to visualize

        Returns:
            gpd.GeoDataFrame: Visualization-ready GeoDataFrame with:
                - Original target/feature columns
                - geometry: Polygon/MultiPolygon
                - isoab: ISO country code
                - country_name: Country name
                - Additional shapefile attributes

        Raises:
            KeyError: If required merge columns missing
            ValueError: If geometries missing after merge

        Note:
            - Converts numeric columns to float32 for memory efficiency
            - Filters to entities present in last time period
            - For PGM: Merges on priogrid_id
            - For CM: Merges on ISO code (isoab)
        """
        _dataframe = dataframe.reset_index()[
            self._dataset.targets + [self._entity_id, self._time_id]
        ]

        numeric_cols = _dataframe.select_dtypes(include=np.number).columns
        _dataframe[numeric_cols] = _dataframe[numeric_cols].astype(np.float32)

        if isinstance(self._dataset, _CDataset):
            _dataframe = self.__add_isoab(dataframe=_dataframe)

            # _dataframe['isoab'] = _dataframe['isoab'].str.upper()
            # self._world['ADM0_A3'] = self._world['ADM0_A3'].str.upper()

            # Include all country attributes in the merge
            _dataframe = _dataframe.merge(
                self._world,
                left_on="isoab",
                right_on="ADM0_A3",
                how="left",
            )
            merged_gdf = gpd.GeoDataFrame(
                _dataframe,
                geometry="geometry",
                crs=self._world.crs,
            )
            return self.__check_missing_geometries(merged_gdf)

        elif isinstance(self._dataset, _PGDataset):
            # Include all priogrid attributes in the merge
            _dataframe = self.__add_isoab(dataframe=_dataframe)
            _dataframe = _dataframe.merge(
                self._world,
                left_on=self._entity_id,
                right_on="gid",
                how="left",
            )
            return self.__check_missing_geometries(
                gpd.GeoDataFrame(_dataframe, geometry="geometry", crs=self._world.crs)
            )

        else:
            raise ValueError("Invalid dataset type. Must be a _PGDataset or _CDataset.")

    def __add_isoab(self, dataframe: pd.DataFrame):
        """
        Enrich DataFrame with ISO country codes and names.

        Merges country identification data (ISO codes and names) from the
        dataset's metadata into the working DataFrame.

        Internal Use:
            Called by __init_mapping_dataframe() during data preparation.

        Args:
            dataframe: DataFrame to enrich with geographic identifiers

        Returns:
            pd.DataFrame: Input DataFrame with added columns:
                - isoab: ISO 3-letter country code
                - country_name: Country name

        Note:
            - Uses dataset's get_isoab() and get_name() methods
            - Merges on time_id and entity_id
            - Left join preserves all input rows
        """
        iso_df = self._dataset.get_isoab().reset_index()
        name_df = self._dataset.get_name(with_id=True).reset_index()

        dataframe = dataframe.merge(
            iso_df[[self._time_id, self._entity_id, "isoab"]],
            on=[self._time_id, self._entity_id],
            how="left",
        )
        dataframe = dataframe.merge(
            name_df[[self._time_id, self._entity_id, "name"]],
            on=[self._time_id, self._entity_id],
            how="left",
        )
        dataframe.rename(columns={"name": "country_name"}, inplace=True)
        return dataframe

    def get_subset_mapping_dataframe(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
    ) -> pd.DataFrame:
        """
        Extract geographically-enabled subset of dataset for visualization.

        Retrieves filtered data and merges with appropriate shapefiles to create
        a GeoDataFrame ready for mapping.

        Args:
            time_ids: Time periods to include. Either:
                - Single integer: 528 (one month)
                - List of integers: [528, 529, 530]
                - None: All time periods
            entity_ids: Entities to include. Either:
                - Single integer: 180 (one country/grid)
                - List of integers: [180, 181, 182]
                - None: All entities

        Returns:
            pd.DataFrame: GeoDataFrame containing:
                - Filtered data rows
                - geometry column with polygons
                - Geographic metadata (ISO codes, names)
                - Original target/feature columns

        Example:
            >>> mapper = MappingModule(dataset)
            >>> # Get data for specific month and countries
            >>> gdf = mapper.get_subset_mapping_dataframe(
            ...     time_ids=528,
            ...     entity_ids=[180, 181, 182]
            ... )
            >>> print(gdf.columns)
            Index(['pred_ged_sb', 'geometry', 'isoab', 'country_name', ...])

        Note:
            - Automatically handles single values or lists
            - Uses dataset's get_subset_dataframe() for filtering
            - Returns GeoDataFrame with valid geometries
        """
        _dataframe = self._dataset.get_subset_dataframe(
            time_ids=time_ids, entity_ids=entity_ids
        )
        _dataframe = self.__init_mapping_dataframe(dataframe=_dataframe)
        return _dataframe

    def _plot_interactive_map(self, mapping_dataframe: gpd.GeoDataFrame, target: str):
        """
        Generate animated Plotly choropleth with temporal controls.

        Creates interactive web-based map with play/pause controls, time slider,
        and hover tooltips showing location details and values.

        Internal Use:
            Called by plot_map() when interactive=True.

        Args:
            mapping_dataframe: GeoDataFrame with data to visualize
            target: Column name to visualize on the map

        Returns:
            plotly.graph_objs._figure.Figure: Interactive Plotly figure with:
                - Animated time slider
                - Play/pause buttons
                - Hover tooltips with metadata
                - Color scale based on 50th-95th quantiles

        Note:
            - Optimizes memory by using float32 and pivot tables
            - Color scale fixed globally across all frames
            - Hover shows location ID, metadata, time, and value
            - Animation duration: 500ms per frame
            - Typical render time: 2-10 seconds for full dataset
        """
        # Create pivot table for efficient data storage
        all_locations = mapping_dataframe[self._location_col].unique()
        all_times = sorted(mapping_dataframe[self._time_id].unique())

        # Create pivot table
        pivot_df = mapping_dataframe.pivot_table(
            index=self._location_col,
            columns=self._time_id,
            values=target,
            aggfunc="first",
        ).reindex(all_locations)

        # Convert to float32 to save memory
        z_data = pivot_df[all_times].astype(np.float32).values

        # Precompute fixed properties for hover data
        fixed_props = mapping_dataframe.drop_duplicates(self._location_col).set_index(
            self._location_col
        )

        exclude_cols = ["geometry", self._time_id, self._entity_id, target]
        hover_columns = [
            col
            for col in self._hover_columns
            if col in fixed_props.columns and col not in exclude_cols
        ]

        # Determine location label based on dataset type
        if isinstance(self._dataset, _PGDataset):
            location_label = "gid"
        elif isinstance(self._dataset, _CDataset):
            location_label = "ADM0_A3"
        else:
            raise ValueError("Invalid dataset type. Must be a _PGDataset or _CDataset.")

        # Prepare base customdata (fixed properties)
        base_customdata = []
        for loc in all_locations:
            row_data = [loc]  # Add location ID as first element
            # Add all hover columns (excluding target)
            for attr in hover_columns:
                if attr in fixed_props.columns:
                    row_data.append(fixed_props.loc[loc, attr])
                else:
                    row_data.append(None)
            row_data.append(all_times[0])  # Add time
            base_customdata.append(row_data)

        # Create hovertemplate
        hover_attrs = "<br>".join(
            [
                f"<b>{attr}</b>: %{{customdata[{i+1}]}}"
                for i, attr in enumerate(hover_columns)
            ]
        )
        hovertemplate = (
            f"<b>{location_label}</b>: %{{customdata[0]}}<br>"
            + hover_attrs
            + f"<br>{self._time_id}: %{{customdata[{len(hover_columns)+1}]}}<br>{target}: %{{z}}<extra></extra>"
        )

        # Calculate global color range
        z_min, z_max = np.nanquantile(z_data, [0.5, 0.95])

        # Create figure with graph objects for better control
        fig = go.Figure(
            data=go.Choropleth(
                geojson=self._base_geojson,
                locations=all_locations,
                z=z_data[:, 0],  # First time step
                featureidkey=self._featureidkey,
                customdata=base_customdata,
                hovertemplate=hovertemplate,
                marker_line_width=0.5,
                coloraxis="coloraxis",
            )
        )

        # Prepare frames with time-specific data
        frames = []
        for i, time in enumerate(all_times[1:], start=1):
            # Prepare customdata for this frame
            frame_customdata = []
            for loc in all_locations:
                row_data = [loc]  # Add location ID as first element
                # Add all hover columns
                for attr in hover_columns:
                    if attr in fixed_props.columns:
                        row_data.append(fixed_props.loc[loc, attr])
                    else:
                        row_data.append(None)
                row_data.append(time)  # Add time
                frame_customdata.append(row_data)

            frame_hover_attrs = "<br>".join(
                [
                    f"<b>{attr}</b>: %{{customdata[{i+1}]}}"
                    for i, attr in enumerate(hover_columns)
                ]
            )
            frame_hovertemplate = (
                f"<b>{location_label}</b>: %{{customdata[0]}}<br>"
                + frame_hover_attrs
                + f"<br>{self._time_id}: %{{customdata[{len(hover_columns)+1}]}}<br>{target}: %{{z}}<extra></extra>"
            )

            frames.append(
                go.Frame(
                    data=[
                        go.Choropleth(
                            z=z_data[:, i],
                            customdata=frame_customdata,
                            hovertemplate=frame_hovertemplate,
                        )
                    ],
                    name=str(time),
                )
            )

        fig.frames = frames

        # Add play button and slider
        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 300},
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 14},
                        "prefix": f"{self._time_id}: ",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [str(time)],
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": str(time),
                            "method": "animate",
                        }
                        for time in all_times
                    ],
                }
            ],
        )

        # Layout adjustments with increased padding
        fig.update_layout(
            height=900,
            autosize=True,
            margin={"r": 20, "t": 60, "l": 20, "b": 60},  # Increased padding
            coloraxis=dict(colorscale="OrRd", cmin=z_min, cmax=z_max),
            annotations=[
                dict(
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                    text="",
                    xref="paper",
                    yref="paper",
                )
            ],
        )

        fig.update_geos(
            fitbounds="locations",
            visible=False,
            showcountries=True,
            countrycolor="rgba(100,100,100,0.3)",
            countrywidth=0.3,
            showlakes=True,
            showocean=False,
            showsubunits=True,
            subunitcolor="rgba(200,200,200,0.2)",
            subunitwidth=0.05,
        )

        # Free memory
        del pivot_df, z_data, fixed_props, base_customdata
        gc.collect()

        return fig

    def _plot_static_map(
        self, mapping_dataframe: gpd.GeoDataFrame, target: str, time_unit: int
    ):
        """
        Generate static matplotlib choropleth for single time period.

        Creates publication-quality static map with customizable styling and
        color scale for a single snapshot in time.

        Internal Use:
            Called by plot_map() when interactive=False.

        Args:
            mapping_dataframe: GeoDataFrame with data to visualize
            target: Column name to visualize on the map
            time_unit: Time period identifier for title

        Returns:
            matplotlib.figure.Figure: Matplotlib figure object with:
                - Choropleth with OrRd color scheme
                - Color scale: 50th-95th quantile
                - Black boundaries (0.3pt width)
                - Horizontal colorbar with label
                - Axis labels (Longitude, Latitude)

        Raises:
            ValueError: If target column not found or contains only null values

        Note:
            - Figure size: 15" × 10"
            - Suitable for publication and reports
            - Color range optimized to highlight variation
            - Edge color: #404040 (dark gray)
            - Alpha: 0.9 for slight transparency
        """
        if target not in mapping_dataframe.columns:
            raise ValueError(f"Target column '{target}' not found")
        if mapping_dataframe[target].isnull().all():
            raise ValueError(f"No valid values for target '{target}'")

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        mapping_dataframe.boundary.plot(ax=ax, linewidth=0.3, color="black")

        plot = mapping_dataframe.plot(
            column=target,
            ax=ax,
            legend=True,
            cmap="OrRd",
            vmin=mapping_dataframe[target].quantile(0.5),
            vmax=mapping_dataframe[target].quantile(0.95),
            linewidth=0.1,
            edgecolor="#404040",
            alpha=0.9,
        )

        plt.title(f"{target} for {self._time_id} {int(time_unit)}", fontsize=15)
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)

        sm = plt.cm.ScalarMappable(
            cmap="OrRd",
            norm=plt.Normalize(
                vmin=self._mapping_dataframe[target].min(),
                vmax=self._mapping_dataframe[target].max(),
            ),
        )
        sm._A = []

        cbar = fig.colorbar(
            sm, ax=ax, orientation="horizontal", fraction=0.036, pad=0.1
        )
        cbar.set_label(f"{target}", fontsize=12)

        return fig

    def plot_map(
        self,
        mapping_dataframe: pd.DataFrame,
        target: str,
        interactive: bool = False,
        as_html: bool = False,
    ):
        """
        Generate choropleth map visualization for specified target variable.

        Creates either interactive (Plotly) or static (Matplotlib) map showing
        geographic distribution of values. Supports temporal animation and
        HTML export.

        Args:
            mapping_dataframe: GeoDataFrame from get_subset_mapping_dataframe()
            target: Variable to visualize. Must be in dataset's targets or features.
            interactive: If True, creates animated Plotly map with controls.
                If False, creates static Matplotlib plot. Default: False
            as_html: If True, returns HTML string instead of figure object.
                Useful for embedding in reports. Default: False

        Returns:
            Union[str, matplotlib.figure.Figure, plotly.graph_objs.Figure]:
                - If as_html=True: HTML string for web embedding
                - If as_html=False and interactive=True: Plotly Figure
                - If as_html=False and interactive=False: Matplotlib Figure

        Raises:
            ValueError: If target not in dataset's targets or features
            ValueError: If static plot requested with multiple time periods

        Example:
            >>> # Interactive map for report
            >>> mapper = MappingModule(dataset)
            >>> gdf = mapper.get_subset_mapping_dataframe(time_ids=[520, 521, 522])
            >>> html = mapper.plot_map(
            ...     gdf,
            ...     target='pred_ged_sb',
            ...     interactive=True,
            ...     as_html=True
            ... )
            >>> with open('map.html', 'w') as f:
            ...     f.write(html)

            >>> # Static map for publication
            >>> gdf_single = mapper.get_subset_mapping_dataframe(time_ids=520)
            >>> fig = mapper.plot_map(gdf_single, 'pred_ged_sb', interactive=False)
            >>> fig.savefig('conflict_map.png', dpi=300)

        Note:
            - Interactive maps require single target across multiple times
            - Static maps require single time period
            - HTML output includes Plotly.js (works offline)
            - Array values automatically extracted if single-element
            - Memory optimized for large datasets (float32, garbage collection)
        """
        target_options = set(self._dataset.targets).union(set(self._dataset.features))
        if target not in target_options:
            raise ValueError(
                f"Target must be a dependent variable or feature. Choose from {target_options}"
            )

        mapping_dataframe[target] = mapping_dataframe[target].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x
        )

        if interactive:
            fig = self._plot_interactive_map(mapping_dataframe, target)
            if as_html:
                html_str = fig.to_html(
                    full_html=True,
                    include_plotlyjs=True,  # Should work offline
                    default_height=900,
                    div_id="map-container",
                )
                # Free memory after generating HTML
                del fig
                gc.collect()
                return html_str
            else:
                return fig
        else:
            time_units = mapping_dataframe[self._time_id].dropna().unique()
            if len(time_units) > 1:
                raise ValueError("Static plots require single time unit")
            fig = self._plot_static_map(mapping_dataframe, target, time_units[0])
            if as_html:
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                plt.close(fig)
                buf.seek(0)
                img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                return f'<img src="data:image/png;base64,{img_str}">'
            else:
                return fig
