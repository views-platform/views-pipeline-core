import pandas as pd
import numpy as np
from ..data.handlers import (
    PGMDataset,
    CMDataset,
    _CDataset,
    _PGDataset,
)
import logging
from typing import Union, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


class MappingManager:
    def __init__(self, views_dataset: Union[PGMDataset, CMDataset]):
        self._dataset = views_dataset
        self._dataframe = self._dataset.dataframe
        self._entity_id = self._dataset._entity_id
        self._time_id = self._dataset._time_id
        if isinstance(views_dataset, _PGDataset):
            self._world = self.__get_priogrid_shapefile()
        elif isinstance(views_dataset, _CDataset):
            self._world = self.__get_country_shapefile()
        else:
            raise ValueError("Invalid dataset type. Must be a PGMDataset or CMDataset.")
        self._mapping_dataframe = None
        self._prepare_base_geojson()  # Initialize base GeoJSON

    def _prepare_base_geojson(self):
        """Create simplified base GeoJSON and set location parameters"""
        base_gdf = self._world.to_crs(epsg=4326).copy()
        base_gdf['geometry'] = base_gdf.geometry.simplify(
            tolerance=0.20,
            preserve_topology=True
        )
        
        if isinstance(self._dataset, _PGDataset):
            self._location_col = 'gid'
            self._featureidkey = "properties.gid"
        else:  # Country dataset
            self._location_col = 'isoab'
            self._featureidkey = "properties.ADM0_A3"

        self._base_geojson = base_gdf.__geo_interface__

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
        _dataframe = dataframe.reset_index()[
            self._dataset.targets + [self._entity_id, self._time_id]
        ]

        numeric_cols = _dataframe.select_dtypes(include=np.number).columns
        _dataframe[numeric_cols] = _dataframe[numeric_cols].astype(np.float32)

        if isinstance(self._dataset, _CDataset):
            _dataframe = self.__add_isoab(dataframe=_dataframe)
            _dataframe = _dataframe.merge(
                self._world[["ADM0_A3", "geometry"]],
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
        iso_df = self._dataset.get_isoab().reset_index()
        name_df = self._dataset.get_name().reset_index()

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
        _dataframe = self._dataset.get_subset_dataframe(
            time_ids=time_ids, entity_ids=entity_ids
        )
        _dataframe = self.__init_mapping_dataframe(dataframe=_dataframe)
        return _dataframe

    def _plot_interactive_map(self, mapping_dataframe: gpd.GeoDataFrame, target: str):
        # Convert to regular DataFrame without geometries
        plot_df = pd.DataFrame(mapping_dataframe.drop(columns='geometry', errors='ignore'))
        
        # Prepare hover data
        hover_data = [self._entity_id, self._time_id, target]
        if isinstance(self._dataset, _CDataset):
            hover_data.append("country_name")

        # Create optimized figure using base GeoJSON
        fig = px.choropleth(
            plot_df,
            geojson=self._base_geojson,
            locations=self._location_col,
            featureidkey=self._featureidkey,
            color=target,
            animation_frame=self._time_id,
            projection="natural earth",
            hover_data=hover_data,
            color_continuous_scale="OrRd",
            range_color=(
                plot_df[target].quantile(0.05),
                plot_df[target].quantile(0.95),
            ),
            labels={self._time_id: "Time Period", target: target},
        )

        # Layout adjustments
        fig.update_layout(
            height=900,
            autosize=True,
            margin={"r": 0, "t": 40, "l": 0, "b": 40},
            sliders=[
                {
                    "currentvalue": {
                        "prefix": f"{self._time_id}: ",
                        "font": {"size": 14},
                    },
                    "pad": {"t": 50, "b": 20},
                    "len": 0.9,
                }
            ],
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

        fig.update_traces(
            marker_line_width=0.5,
            marker_opacity=0.9,
            selector=dict(type="choropleth")
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

        return fig

    def _plot_static_map(
        self, mapping_dataframe: gpd.GeoDataFrame, target: str, time_unit: int
    ):
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
                return fig.to_html(
                    full_html=False,
                    include_plotlyjs="cdn",  # Use CDN for plotly.js
                    default_height=900
                )
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