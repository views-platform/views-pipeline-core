import pandas as pd
import numpy as np
from views_pipeline_core.data.handlers import PGMDataset, CMDataset
import logging
from typing import Union, Optional, List
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
import plotly.express as px

# nbformat dependency is required

logger = logging.getLogger(__name__)


class MappingManager:
    def __init__(self, views_dataset: Union[PGMDataset, CMDataset]):
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
        self._mapping_dataframe = self.__init_mapping_dataframe(self._dataframe)
        # self._mapping_dataframe = self._dataframe.reset_index()

        self._isoab_cache_dataframe = None

    def __get_country_shapefile(self):
        # path = Path(__file__).parent.parent / "mapping" / "shapefiles" / "country" / "ne_110m_admin_0_countries.shp"
        path = "/Users/dylanpinheiro/Desktop/views-platform/views-pipeline-core/views_pipeline_core/mapping/shapefiles/country/ne_110m_admin_0_countries.shp"
        return gpd.read_file(path)

    def __get_priogrid_shapefile(self):
        path = "/Users/dylanpinheiro/Desktop/views-platform/experiments/shapefiles/priogrid_cellshp/priogrid_cell.shp"
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
        _dataframe = dataframe.reset_index()

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
        if isinstance(self._dataset, CMDataset):
            dataframe.rename(columns={self._entity_id: "c_id"}, inplace=True)
            dataframe["country_name"] = dataframe.c.name
            dataframe["isoab"] = dataframe.c.isoab
            dataframe.rename(columns={"c_id": self._entity_id}, inplace=True)
        return dataframe

    def __add_cid(self, dataframe: pd.DataFrame):
        if isinstance(self._dataset, PGMDataset):
            dataframe.rename(columns={self._entity_id: "pg_id"}, inplace=True)
            dataframe["c_id"] = dataframe.pg.c_id
            dataframe["country_name"] = dataframe.pg.name
            dataframe.rename(columns={"pg_id": self._entity_id}, inplace=True)
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

        # Create figure with slider
        fig = px.choropleth(
            mapping_dataframe,
            geojson=mapping_dataframe.geometry,
            locations=mapping_dataframe.index,
            color=target,
            animation_frame=self._time_id,
            projection="natural earth",
            hover_data=[self._entity_id, self._time_id, target],
            color_continuous_scale="OrRd",
            range_color=(
                mapping_dataframe[target].min(),
                mapping_dataframe[target].max(),
            ),
            labels={self._time_id: "Time Period", target: target},
        )

        # Adjust layout for larger size
        fig.update_layout(
            height=800,  # Increased from default 450
            width=1200,  # Increased from default 700
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

        # Improve map rendering
        fig.update_geos(
            fitbounds="locations",
            visible=False,
            showcountries=True,
            countrycolor="rgba(100,100,100,0.3)",  # Lighter gray
            countrywidth=0.3,  # Thinner borders
            # Add grid line styling
            showlakes=False,
            showocean=False,
            showsubunits=True,
            subunitcolor="rgba(200,200,200,0.2)",
            subunitwidth=0.2,
        )

        # Add latitude/longitude grid styling
        fig.update_layout(
            geo=dict(
                lonaxis=dict(
                    showgrid=True, gridcolor="rgba(200,200,200,0.3)", gridwidth=0.2
                ),
                lataxis=dict(
                    showgrid=True, gridcolor="rgba(200,200,200,0.3)", gridwidth=0.2
                ),
            )
        )

        return fig

    def _plot_static_map(
        self, mapping_dataframe: gpd.GeoDataFrame, target: str, time_unit: int
    ):
        # mapping_dataframe = mapping_dataframe.set_geometry("geometry")
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        mapping_dataframe.boundary.plot(ax=ax, linewidth=0.3, color="black")
        # ax.set_axis_off()

        # Plot the data
        mapping_dataframe.plot(
            column=f"{target}",
            ax=ax,
            legend=False,
            legend_kwds={"label": f"{target}", "orientation": "horizontal"},
            cmap="OrRd",
            vmin=self._mapping_dataframe[target].min(),
            vmax=self._mapping_dataframe[target].max(),
            linewidth=0.1,
            color="#404040",  # Darker gray instead of black
            alpha=0.5,  # Add transparency
        )

        # Add title and labels
        plt.title(f"{target} for {self._time_id} {int(time_unit)}", fontsize=15)
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)

        # Add a color bar
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

        # plt.show()
        return plt

    def plot_map(
        self, mapping_dataframe: pd.DataFrame, target: str, interactive: bool = False
    ):
        target_options = set(self._dataset.dep_vars).union(
            set(self._dataset.indep_vars)
        )
        if target not in target_options:
            raise ValueError(
                f"Target must be a dependent variable or feature in the dataset. Choose from {target_options}"
            )

        mapping_dataframe[target] = mapping_dataframe[target].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x
        )

        if interactive:
            return self._plot_interactive_map(mapping_dataframe, target)
        else:
            time_units = mapping_dataframe[self._time_id].dropna().unique()
            if len(time_units) > 1:
                raise ValueError("Static plots require single time unit")
            return self._plot_static_map(mapping_dataframe, target, time_units[0])