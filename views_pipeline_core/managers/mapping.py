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

logger = logging.getLogger(__name__)


class MappingManager:
    def __init__(self, views_dataset: Union[PGMDataset, CMDataset]):
        self._dataset = views_dataset
        self._dataframe = views_dataset.dataframe
        if isinstance(views_dataset, PGMDataset):
            from ingester3.extensions import PgAccessor
            self._world = self.__get_priogrid_shapefile()
            self.target = "pgm"
        elif isinstance(views_dataset, CMDataset):
            from ingester3.extensions import CAccessor
            self._world = self.__get_country_shapefile()
            self.target = "cm"
        else:
            raise ValueError("Invalid dataset type. Must be a PGMDataset or CMDataset.")
        self._mapping_dataframe = self.__init_mapping_dataframe(self._dataframe)

        self._isoab_cache_dataframe = None

    def __get_country_shapefile(self):
        path = Path(__file__).parent.parent / "mapping" / "shapefiles" / "country" / "ne_110m_admin_0_countries.shp"
        return gpd.read_file(path)
    
    def __get_priogrid_shapefile(self):
        pass

    def __init_mapping_dataframe(self, dataframe: pd.DataFrame):
        _dataframe = dataframe.copy().reset_index()
        if isinstance(self._dataset, CMDataset):
            _dataframe = self.__add_isoab(dataframe=_dataframe)
            _dataframe = _dataframe.merge(
                self._world, left_on="isoab", right_on="ADM0_A3", how="left"
            )
        return _dataframe

    def __add_isoab(self, dataframe: pd.DataFrame):
        if isinstance(self._dataset, CMDataset):
            dataframe.rename(columns={"country_id": "c_id"}, inplace=True)
            dataframe["isoab"] = dataframe.c.isoab
            dataframe.rename(columns={"c_id": "country_id"}, inplace=True)
        return dataframe

    def get_subset_mapping_dataframe(self,
        time_ids: int,
        entity_ids: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
        """
        Retrieves a subset of the mapping dataframe based on the provided time and entity IDs.

        Args:
            time_ids (int): The time identifier(s) to filter the dataframe.
            entity_ids (Optional[Union[int, List[int]]], optional): The entity identifier(s) [priogrid_id or country_id] to filter the dataframe. 
            Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the subset of the mapping data.
        """
        _dataframe = self._dataset.get_subset_dataframe(
            time_ids=time_ids, entity_ids=entity_ids
        )
        _dataframe = self.__init_mapping_dataframe(dataframe=_dataframe)
        return _dataframe

    def plot_map(self, mapping_dataframe: pd.DataFrame, target: str):
        """
        Plots a map based on the provided mapping dataframe and target variable.

        Parameters:
        -----------
        mapping_dataframe : pd.DataFrame
            DataFrame containing the mapping data, including geometries and the target variable.
        target : str
            The target variable to be plotted. Must be a dependent variable or feature in the dataset.

        Raises:
        -------
        ValueError
            If the target is not a dependent variable or feature in the dataset.
            If more than one month is found in the mapping_dataframe.

        Notes:
        ------
        This function is designed to work with CMDataset instances and assumes that the mapping_dataframe
        contains a 'geometry' column for plotting.

        The function will plot the boundaries and the target variable on a map, with a color bar indicating
        the range of the target variable values.
        """
        target_options = set(self._dataset.dep_vars).union(
            set(self._dataset.indep_vars)
        )
        if target not in target_options:
            raise ValueError(
                f"Target must be a dependent variable or feature in the dataset. Choose from {target_options}"
            )

        month = mapping_dataframe["month_id"].dropna().unique()
        if len(month) > 1:
            raise ValueError(
                f"Only one month can be plotted at a time. Found months {month} in mapping_dataframe"
            )
        month = month[0]

        mapping_dataframe[target] = mapping_dataframe[target].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x
        )

        if isinstance(self._dataset, CMDataset):
            mapping_dataframe = mapping_dataframe.set_geometry("geometry")
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
            )

            # Add title and labels
            plt.title(f"{target} for month {int(month)}", fontsize=15)
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

            plt.show()