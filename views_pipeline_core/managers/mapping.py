import pandas as pd
import numpy as np
from views_pipeline_core.data.handlers import PGMDataset, CMDataset, _CDataset
import logging
from typing import Union, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
from io import BytesIO
import base64

# nbformat dependency is required

logger = logging.getLogger(__name__)


class MappingManager:
    def __init__(self, views_dataset: Union[PGMDataset, CMDataset]):
        if views_dataset.sample_size > 1:
            logger.info(
                f"Calculating MAP for dataset. Found sample size of {views_dataset.sample_size}"
            )
            self._dataset = type(views_dataset)(
                views_dataset.calculate_map(features=None)
            )
        else:
            self._dataset = views_dataset
        self._dataframe = self._dataset.dataframe
        self._entity_id = self._dataset._entity_id
        self._time_id = self._dataset._time_id
        if isinstance(views_dataset, PGMDataset):
            from ingester3.extensions import PgAccessor

            self._world = self.__get_priogrid_shapefile()

        elif isinstance(views_dataset, CMDataset):
            from ingester3.extensions import CAccessor

            self._world = self.__get_country_shapefile()
        else:
            raise ValueError("Invalid dataset type. Must be a PGMDataset or CMDataset.")
        # self._mapping_dataframe = self.__init_mapping_dataframe(self._dataframe)
        self._mapping_dataframe = None

    def __get_country_shapefile(self):
        path = (
            Path(__file__).parent.parent
            / "mapping"
            / "shapefiles"
            / "country"
            / "ne_110m_admin_0_countries.shp"
        )
        return gpd.read_file(path)

    def __get_priogrid_shapefile(self):
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
        missing = mapping_dataframe[
            mapping_dataframe.geometry.is_empty | mapping_dataframe.geometry.isna()
        ]
        if not missing.empty:
            logger.warning(f"Missing geometries for: {missing['isoab'].unique()}")
            # Handle missing cases (e.g., filter or impute)
        if drop_missing_geometries:
            initial_count = len(mapping_dataframe)
            # Filter out null/empty geometries
            cleaned_gdf = mapping_dataframe[
                (~mapping_dataframe.geometry.is_empty)
                & (~mapping_dataframe.geometry.isna())
            ].copy()

            # Calculate dropped rows
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
        _dataframe = dataframe.reset_index()[
            self._dataset.targets + [self._entity_id, self._time_id]
        ]

        numeric_cols = _dataframe.select_dtypes(include=np.number).columns
        _dataframe[numeric_cols] = _dataframe[numeric_cols].astype(np.float32)

        if isinstance(self._dataset, CMDataset):
            # Add ISO codes to the dataframe
            _dataframe = self.__add_isoab(dataframe=_dataframe)

            # Merge with shapefile geometries
            _dataframe = _dataframe.merge(
                self._world[["ADM0_A3", "geometry"]],
                left_on="isoab",
                right_on="ADM0_A3",
                how="left",
            )

            # Create GeoDataFrame with ALL columns and explicit geometry
            merged_gdf = gpd.GeoDataFrame(
                _dataframe,
                geometry="geometry",  # Use merged geometry column
                crs=self._world.crs,
            )

            return self.__check_missing_geometries(merged_gdf)

        elif isinstance(self._dataset, PGMDataset):
            # _dataframe = self.__add_cid(dataframe=_dataframe)
            # Merge priogrid geometries
            _dataframe = _dataframe.merge(
                self._world[["gid", "geometry"]],
                left_on=self._entity_id,
                right_on="gid",
                how="left",
            )

            return self.__check_missing_geometries(
                gpd.GeoDataFrame(_dataframe, geometry="geometry", crs=self._world.crs)
            )

    def __add_isoab(self, dataframe: pd.DataFrame):
        # if isinstance(self._dataset, CMDataset):
        #     dataframe.rename(columns={self._entity_id: "c_id"}, inplace=True)
        #     dataframe["country_name"] = dataframe.c.name
        #     dataframe["isoab"] = dataframe.c.isoab
        #     dataframe.rename(columns={"c_id": self._entity_id}, inplace=True)
        # return dataframe
        # Get ISO codes and country names through dataset methods
        iso_df = self._dataset.get_isoab().reset_index()
        name_df = self._dataset.get_name().reset_index()

        # Merge with main dataframe
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

    def __add_cid(self, dataframe: pd.DataFrame):
        # if isinstance(self._dataset, PGMDataset):
        #     dataframe.rename(columns={self._entity_id: "pg_id"}, inplace=True)
        #     dataframe["c_id"] = dataframe.pg.c_id
        #     dataframe["country_name"] = dataframe.pg.name
        #     dataframe.rename(columns={"pg_id": self._entity_id}, inplace=True)
        # return dataframe
        if isinstance(self._dataset, PGMDataset):
            # Get country IDs through dataset method
            cid_df = self._dataset.get_country_id().reset_index()

            # Get country names using CDataset
            country_ids = cid_df["c_id"].unique()
            temp_country_df = pd.DataFrame({"c_id": country_ids})
            temp_cdataset = CMDataset(
                temp_country_df, targets=[]
            )  # Use CMDataset for month alignment

            # Merge names
            name_df = temp_cdataset.get_name().reset_index()
            cid_df = cid_df.merge(name_df[["c_id", "name"]], on="c_id", how="left")

            # Merge with main dataframe
            dataframe = dataframe.merge(
                cid_df, on=[self._time_id, self._entity_id], how="left"
            )
            dataframe.rename(columns={"name": "country_name"}, inplace=True)

        return dataframe

    def get_subset_mapping_dataframe(
        self,
        time_ids: Optional[Union[int, List[int]]] = None,
        entity_ids: Optional[Union[int, List[int]]] = None,
    ) -> pd.DataFrame:
        """
        Get subset dataframe for specified time.

        Parameters:
        time_ids: Single time ID for static maps or list of time IDs for interactive maps
        entity_ids: Single entity ID or list of entity IDs
        """
        _dataframe = self._dataset.get_subset_dataframe(
            time_ids=time_ids, entity_ids=entity_ids
        )
        _dataframe = self.__init_mapping_dataframe(dataframe=_dataframe)
        return _dataframe

    def _plot_interactive_map(self, mapping_dataframe: gpd.GeoDataFrame, target: str):
        if not isinstance(mapping_dataframe, gpd.GeoDataFrame):
            mapping_dataframe = gpd.GeoDataFrame(mapping_dataframe, geometry="geometry")
        # Convert to geographic CRS for Plotly
        mapping_dataframe = mapping_dataframe.to_crs(epsg=4326).copy()

        # Simplify geometries for faster rendering
        mapping_dataframe["geometry"] = mapping_dataframe.geometry.simplify(
            tolerance=0.20,  # Adjust based on data scale (degrees for EPSG:4326)
            preserve_topology=True,
        )
        hover_data = [self._entity_id, self._time_id, target]
        if isinstance(self._dataset, _CDataset):
            hover_data.append("country_name")

        # Create figure with slider
        fig = px.choropleth(
            mapping_dataframe,
            geojson=mapping_dataframe.geometry,
            locations=mapping_dataframe.index,
            color=target,
            animation_frame=self._time_id,
            projection="natural earth",
            hover_data=hover_data,
            color_continuous_scale="OrRd",
            range_color=(
                # mapping_dataframe[target].min(),
                # mapping_dataframe[target].max(),
                mapping_dataframe[target].quantile(0.05),
                mapping_dataframe[target].quantile(0.95),
            ),
            labels={self._time_id: "Time Period", target: target},
        )

        # Adjust layout for larger size
        fig.update_layout(
            height=900,  # Increased from default 450
            # width=1200,  # Increased from default 700
            autosize=True,
            margin={"r": 0, "t": 40, "l": 0, "b": 40},
            sliders=[
                {
                    "currentvalue": {
                        "prefix": f"{self._time_id}: ",
                        "font": {"size": 14},
                    },
                    "pad": {"t": 50, "b": 20},  # Added bottom padding
                    "len": 0.9,  # Make slider wider
                }
            ],
        )
        # fig.update_layout(
        #     height=1000,  # Increased height
        #     margin={"r": 0, "t": 40, "l": 0, "b": 140},  # More bottom space
        #     sliders=[
        #         {
        #             "currentvalue": {
        #                 "prefix": f"{self._time_id}: ",
        #                 "font": {"size": 14},
        #                 "xanchor": "right",
        #                 "offset": 20,
        #             },
        #             "pad": {"t": 50, "b": 100},  # Increased bottom padding
        #             "len": 0.95,
        #             "x": 0.05,  # Left-align slider
        #         }
        #     ],
        # )
        fig.update_layout(
            annotations=[
                dict(
                    x=0.5,
                    y=-0.15,  # Position below main plot
                    showarrow=False,
                    text="",
                    xref="paper",
                    yref="paper",
                )
            ]
        )

        fig.update_traces(
            marker_line_width=0.5, marker_opacity=0.9, selector=dict(type="choropleth")
        )

        # Improve map rendering
        fig.update_geos(
            fitbounds="locations",
            visible=False,
            showcountries=True,
            countrycolor="rgba(100,100,100,0.3)",  # Lighter gray
            countrywidth=0.3,  # Thinner borders
            # Add grid line styling
            showlakes=True,
            showocean=False,
            showsubunits=True,
            subunitcolor="rgba(200,200,200,0.2)",
            subunitwidth=0.05,
        )

        # Add latitude/longitude grid styling
        # fig.update_layout(
        #     geo=dict(
        #         lonaxis=dict(
        #             showgrid=True, gridcolor="rgba(200,200,200,0.3)", gridwidth=0.2
        #         ),
        #         lataxis=dict(
        #             showgrid=True, gridcolor="rgba(200,200,200,0.3)", gridwidth=0.2
        #         ),
        #     )
        # )

        return fig

    def _plot_static_map(
        self, mapping_dataframe: gpd.GeoDataFrame, target: str, time_unit: int
    ):
        # Validation checks
        if target not in mapping_dataframe.columns:
            raise ValueError(f"Target column '{target}' not found in mapping dataframe")

        if mapping_dataframe[target].isnull().all():
            raise ValueError(f"No valid values found for target '{target}'")

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Plot boundaries
        mapping_dataframe.boundary.plot(ax=ax, linewidth=0.3, color="black")

        # Plot data values
        plot = mapping_dataframe.plot(
            column=target,
            ax=ax,
            legend=True,
            # legend_kwds={
            #     "label": f"",
            #     "orientation": "horizontal",
            #     "pad": 0.01,
            #     "aspect": 40,
            # },
            cmap="OrRd",
            vmin=mapping_dataframe[target].quantile(0.05),
            vmax=mapping_dataframe[target].quantile(0.95),
            linewidth=0.1,
            edgecolor="#404040",
            alpha=0.9,
        )

        # Add metadata
        plt.title(
            f"{target} for {self._time_id} {int(time_unit)}",
            fontsize=15,
        )
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
        Plots a map based on the provided mapping dataframe and target variable.

        Parameters:
        -----------
        mapping_dataframe : pd.DataFrame
            The dataframe containing the mapping data.
        target : str
            The target variable to plot. Must be a dependent variable or feature in the dataset.
        interactive : bool, optional
            If True, creates an interactive plot. Default is False.
        as_html : bool, optional
            If True, returns the plot as an HTML string. Default is False.

        Returns:
        --------
        fig or str
            The plot figure object or an HTML string if `as_html` is True.

        Raises:
        -------
        ValueError
            If the target is not a dependent variable or feature in the dataset.
            If static plots are requested with multiple time units.
        """
        target_options = set(self._dataset.targets).union(set(self._dataset.features))
        if target not in target_options:
            raise ValueError(
                f"Target must be a dependent variable or feature in the dataset. Choose from {target_options}"
            )

        mapping_dataframe[target] = mapping_dataframe[target].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x
        )

        if interactive:
            fig = self._plot_interactive_map(mapping_dataframe, target)
            if as_html:
                return fig.to_html(full_html=True)
            else:
                return fig
        else:
            time_units = mapping_dataframe[self._time_id].dropna().unique()
            if len(time_units) > 1:
                raise ValueError("Static plots require single time unit")
            fig = self._plot_static_map(mapping_dataframe, target, time_units[0])
            if as_html:
                # Convert figure to HTML image
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                plt.close(fig)
                buf.seek(0)
                img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                html_img = f'<img src="data:image/png;base64,{img_str}">'
                return html_img
            else:
                return fig
