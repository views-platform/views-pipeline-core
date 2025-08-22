import pandas as pd
import numpy as np
from ..data.handlers import (
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


class MappingManager:
    _COUNTRY_HOVER_COLS = ['country_name']
    _PRIOGRID_HOVER_COLS = ['gid', 'row', 'col', 'country_name', 'isoab', 'xcoord', 'ycoord']
    def __init__(self, views_dataset: Union[_PGDataset, _CDataset]):
        """
        Initializes the mapping manager with the provided dataset, setting up internal references
        to the dataset's dataframe, entity and time identifiers, and loading the appropriate
        shapefile and location metadata based on the dataset type.
        Args:
            views_dataset (Union[_PGDataset, _CDataset]): The dataset to be managed, either a
                priogrid dataset (_PGDataset) or a country dataset (_CDataset).
        Raises:
            ValueError: If the provided dataset is not an instance of _PGDataset or _CDataset.
        """
        self._dataset = views_dataset
        self._dataframe = self._dataset.dataframe
        self._entity_id = self._dataset._entity_id
        self._time_id = self._dataset._time_id
        
        if isinstance(views_dataset, _PGDataset):
            self._world = self.__get_priogrid_shapefile()
            self._location_col = 'gid'
            self._featureidkey = "properties.gid"
            # Get all available priogrid attributes (excluding geometry)
            self._priogrid_attributes = [col for col in self._world.columns if col != 'geometry']
            self._hover_columns = self._PRIOGRID_HOVER_COLS
        elif isinstance(views_dataset, _CDataset):
            self._world = self.__get_country_shapefile()
            self._location_col = 'ADM0_A3'
            self._featureidkey = "properties.ADM0_A3"
            # Get all available country attributes (excluding geometry)
            self._country_attributes = [col for col in self._world.columns if col != 'geometry']
            self._hover_columns = self._COUNTRY_HOVER_COLS
        else:
            raise ValueError("Invalid dataset type. Must be a _PGDataset or _CDataset.")

        self._mapping_dataframe = None
        self._base_geojson = None
        self._prepare_base_geojson()  # Initialize base GeoJSON

    def _prepare_base_geojson(self):
        """
        Creates a simplified base GeoJSON representation of the world dataset with only essential properties.
        This method transforms the world GeoDataFrame to WGS84 (EPSG:4326), retains only the necessary columns
        (depending on the dataset type), and simplifies the geometries to reduce file size. The resulting
        GeoJSON is stored in the `_base_geojson` attribute. Memory used by intermediate objects is released
        after processing.
        Returns:
            None
        """
        base_gdf = self._world.to_crs(epsg=4326).copy()
        
        # Keep only essential properties to reduce size
        if isinstance(self._dataset, _PGDataset):
            base_gdf = base_gdf[['gid', 'geometry']]
        elif isinstance(self._dataset, _CDataset):
            # For country datasets, keep ADM0_A3 (which matches isoab) and geometry
            base_gdf = base_gdf[['ADM0_A3', 'geometry']]
        else:
            raise ValueError("Invalid dataset type. Must be a _PGDataset or _CDataset.")
            
        # Simplify geometries to reduce file size
        base_gdf['geometry'] = base_gdf.geometry.simplify(
            tolerance=0.01,
            preserve_topology=True
        )
        
        self._base_geojson = base_gdf.__geo_interface__
        
        # Free memory
        del base_gdf
        gc.collect()

    def __get_country_shapefile(self):
        """
        Loads and returns the country shapefile as a GeoDataFrame.

        Returns:
            geopandas.GeoDataFrame: A GeoDataFrame containing the country boundaries
            from the 'ne_110m_admin_0_countries.shp' shapefile.

        Raises:
            FileNotFoundError: If the shapefile does not exist at the specified path.
            OSError: If there is an issue reading the shapefile.
        """
        path = (
            Path(__file__).parent.parent
            / "mapping"
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
        Loads and returns the PrioGrid shapefile as a GeoDataFrame.

        Returns:
            geopandas.GeoDataFrame: A GeoDataFrame containing the PrioGrid cell shapefile data.

        Raises:
            FileNotFoundError: If the shapefile does not exist at the specified path.
            OSError: If there is an issue reading the shapefile.
        """
        path = (
            Path(__file__).parent.parent
            / "mapping"
            / "shapefiles"
            / "priogrid"
            / "priogrid_cell.shp"
        )
        return gpd.read_file(path)

    def __check_missing_geometries(
        self, mapping_dataframe: pd.DataFrame, drop_missing_geometries: bool = True
    ):
        """
        Checks for missing geometries in the provided GeoDataFrame and optionally drops rows with missing geometries.

        Args:
            mapping_dataframe (pd.DataFrame): The DataFrame containing geometry data to check.
            drop_missing_geometries (bool, optional): If True, rows with missing or empty geometries are dropped from the DataFrame. Defaults to True.

        Returns:
            pd.DataFrame: The cleaned DataFrame with missing geometries removed if `drop_missing_geometries` is True; otherwise, returns the original DataFrame.

        Logs:
            - Warns if any missing geometries are found, listing their unique 'isoab' values.
            - Warns about the number of rows dropped and the IDs of missing geometries if any rows are removed.
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
        Initializes and returns a GeoDataFrame for mapping purposes based on the provided DataFrame and the dataset type.

        This method processes the input DataFrame by:
        - Resetting its index and selecting relevant columns (targets, entity ID, and time ID).
        - Casting all numeric columns to float32 for consistency.
        - Depending on the dataset type (`_CDataset` or `_PGDataset`), it adds ISO country codes and merges with a world geometry DataFrame (`self._world`) using appropriate keys.
        - Constructs a GeoDataFrame with the merged data and checks for missing geometries.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing the data to be mapped.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame ready for mapping, with geometries merged and validated.

        Raises:
            KeyError: If required columns for merging are missing in the input DataFrame or `self._world`.
            ValueError: If geometries are missing after merging.
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
        Adds 'isoab' and 'country_name' columns to the given DataFrame by merging with ISO and name datasets.

        Parameters:
            dataframe (pd.DataFrame): The input DataFrame to which 'isoab' and 'country_name' columns will be added.

        Returns:
            pd.DataFrame: The input DataFrame with additional 'isoab' and 'country_name' columns merged in.
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
        Retrieves a subset of the mapping DataFrame based on specified time and entity IDs.

        Parameters:
            time_ids (Optional[Union[int, List[int]]]): An integer or list of integers specifying the time IDs to filter the DataFrame. If None, no filtering is applied on time IDs.
            entity_ids (Optional[Union[int, List[int]]]): An integer or list of integers specifying the entity IDs to filter the DataFrame. If None, no filtering is applied on entity IDs.

        Returns:
            pd.DataFrame: A DataFrame containing the subset of mapping data filtered by the provided time and entity IDs.
        """
        _dataframe = self._dataset.get_subset_dataframe(
            time_ids=time_ids, entity_ids=entity_ids
        )
        _dataframe = self.__init_mapping_dataframe(dataframe=_dataframe)
        return _dataframe

    def _plot_interactive_map(self, mapping_dataframe: gpd.GeoDataFrame, target: str):
        """
        Generates an interactive animated choropleth map using Plotly, visualizing the temporal evolution of a target variable across geographic locations.
        Args:
            mapping_dataframe (gpd.GeoDataFrame): A GeoDataFrame containing geographic features, time identifiers, and the target variable to visualize.
            target (str): The name of the column in `mapping_dataframe` representing the variable to be visualized on the map.
        Returns:
            plotly.graph_objs._figure.Figure: A Plotly Figure object containing the interactive animated choropleth map.
        Details:
            - The map animates over the time dimension, allowing users to explore changes in the target variable across locations.
            - Hover tooltips display location-specific metadata, excluding geometry and index columns.
            - The color scale is globally normalized based on quantiles of the target variable.
            - Includes play/pause controls and a time slider for animation.
            - Memory usage is optimized by converting data to float32 and cleaning up intermediate variables.
        """
        # Create pivot table for efficient data storage
        all_locations = mapping_dataframe[self._location_col].unique()
        all_times = sorted(mapping_dataframe[self._time_id].unique())
        
        # Create pivot table
        pivot_df = mapping_dataframe.pivot_table(
            index=self._location_col,
            columns=self._time_id,
            values=target,
            aggfunc='first'
        ).reindex(all_locations)
        
        # Convert to float32 to save memory
        z_data = pivot_df[all_times].astype(np.float32).values
        
        # Precompute fixed properties for hover data
        fixed_props = mapping_dataframe.drop_duplicates(self._location_col).set_index(self._location_col)
        
        exclude_cols = ['geometry', self._time_id, self._entity_id, target]
        hover_columns = [col for col in self._hover_columns if col in fixed_props.columns and col not in exclude_cols]
        
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
        hover_attrs = "<br>".join([f"<b>{attr}</b>: %{{customdata[{i+1}]}}" for i, attr in enumerate(hover_columns)])
        hovertemplate = f"<b>{location_label}</b>: %{{customdata[0]}}<br>" + hover_attrs + f"<br>{self._time_id}: %{{customdata[{len(hover_columns)+1}]}}<br>{target}: %{{z}}<extra></extra>"
        
        # Calculate global color range
        z_min, z_max = np.nanquantile(z_data, [0.05, 1.00])
        
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
                coloraxis="coloraxis"
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
            
            frame_hover_attrs = "<br>".join([f"<b>{attr}</b>: %{{customdata[{i+1}]}}" for i, attr in enumerate(hover_columns)])
            frame_hovertemplate = f"<b>{location_label}</b>: %{{customdata[0]}}<br>" + frame_hover_attrs + f"<br>{self._time_id}: %{{customdata[{len(hover_columns)+1}]}}<br>{target}: %{{z}}<extra></extra>"
            
            frames.append(go.Frame(
                data=[go.Choropleth(
                    z=z_data[:, i],
                    customdata=frame_customdata,
                    hovertemplate=frame_hovertemplate
                )],
                name=str(time)
            ))
        
        fig.frames = frames
        
        # Add play button and slider
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, 
                                "fromcurrent": True, "transition": {"duration": 300}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, 
                                "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 14},
                    "prefix": f"{self._time_id}: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [{
                    "args": [
                        [str(time)],
                        {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}
                    ],
                    "label": str(time),
                    "method": "animate"
                } for time in all_times]
            }]
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
            ]
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
        Plots a static choropleth map for a specified target column in a GeoDataFrame.

        Parameters
        mapping_dataframe : gpd.GeoDataFrame
            The GeoDataFrame containing the geometries and data to plot.
        target : str
            The name of the column in `mapping_dataframe` to visualize.
        time_unit : int
            The time unit to display in the plot title.

        Raises
        ValueError
            If the target column is not found in the dataframe or contains only null values.

        Returns
        matplotlib.figure.Figure
            The matplotlib Figure object containing the generated map.
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
            vmin=mapping_dataframe[target].quantile(0.05),
            vmax=mapping_dataframe[target].quantile(1.00),
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
        Plots a map visualization of the provided mapping DataFrame for a specified target variable.

        Depending on the parameters, the map can be rendered as either an interactive Plotly map or a static Matplotlib plot.
        The output can also be returned as an HTML string for embedding in web pages.

        Args:
            mapping_dataframe (pd.DataFrame): The DataFrame containing mapping data to plot.
            target (str): The name of the target variable or feature to visualize. Must be present in the dataset's targets or features.
            interactive (bool, optional): If True, generates an interactive Plotly map. If False, generates a static Matplotlib plot. Defaults to False.
            as_html (bool, optional): If True, returns the plot as an HTML string (either Plotly HTML or base64-encoded PNG). If False, returns the figure object. Defaults to False.

        Returns:
            Union[str, matplotlib.figure.Figure, plotly.graph_objs.Figure]:
                - If as_html is True: HTML string representation of the plot.
                - If as_html is False: The figure object (Plotly or Matplotlib) for further manipulation or display.

        Raises:
            ValueError: If the target is not a valid dependent variable or feature.
            ValueError: If a static plot is requested for multiple time units in the data.
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
                    include_plotlyjs="cdn",  # Use CDN for plotly.js
                    default_height=900,
                    div_id="map-container"
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