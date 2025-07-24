import os
from dotenv import load_dotenv
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from views_pipeline_core.configs import drift_detection
from views_pipeline_core.files.utils import create_data_fetch_log_file
from views_pipeline_core.data.utils import ensure_float64
from views_pipeline_core.files.utils import read_dataframe, save_dataframe
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.managers.model import ModelPathManager
from ingester3.ViewsMonth import ViewsMonth

import views_transformation_library as vtl
import views_transformation_library.views_2 as views2
import views_transformation_library.splag4d as splag4d
import views_transformation_library.missing as missing
from viewser import Queryset, Column
import traceback
import views_transformation_library.splag_country as splag_country 
import views_transformation_library.spatial_tree as spatial_tree
import views_transformation_library.spacetime_distance as spacetime_distance

logger = logging.getLogger(__name__)


transformation_mapping={'ops.ln': views2.ln,
                       'missing.fill': missing.fill,
                       'bool.gte': views2.greater_or_equal,
                       'temporal.time_since': views2.time_since,
                       'temporal.decay': views2.decay,
                       'missing.replace_na': missing.replace_na,
                       'spatial.countrylag': splag_country.get_splag_country, 
                       'temporal.tlag': views2.tlag,
                       'spatial.lag': splag4d.get_splag4d,
                       'spatial.treelag':spatial_tree.get_tree_lag,
                       'spatial.sptime_dist': spacetime_distance.get_spacetime_distances,
                       'temporal.moving_sum':views2.moving_sum,
                       'temporal.moving_average':views2.moving_sum,}

TRANSFORMATIONS_EXPECTING_DF = {
    "spatial.lag"
}



class UpdateViewser:



    def __init__(self, queryset, viewser_df, data_path: str | Path, months_to_update: List[int]):
        self.queryset = queryset
        self.viewser_df = viewser_df
        self.data_path = Path(data_path)
        self.months_to_update = list(months_to_update)  

        (self.base_variables,
         self.var_names,
         self.transformation_list) = self._extract_from_queryset()

        self.df_external = self._load_update_df()

        self.result: pd.DataFrame | None = None  # filled by .run()


    def run(self) -> pd.DataFrame:
        """
        Execute every step and return the final dataframe. Safe to call twice:
        we memoise the result after first run.
        """
        if self.result is not None: 
            logger.debug('Use saved dataframe')       # already done
            return self.result

        # 1) Adapt update df to queryset and month_ids to update
        df_update = self.preprocess_update_df()


        # 2) Update df from viewser
        #df = self.queryset.publish().fetch()
        self.viewser_df.update(df_update)

        logger.info('fetched and updated from viewser')

        # 3) Apply transformations
        df_final = self._apply_all_transformations(df_old = self.viewser_df)
        logger.info('all transformations done')

        cols_to_drop = df_final.columns[df_final.columns.str.startswith('raw')]
        df_final = df_final.drop(columns=cols_to_drop)  

        # 4)return
        return df_final


    # 1. -------------  PARSE THE QUERYSET  -------------------------------- #
    def _extract_from_queryset(
        self
    ) -> Tuple[List[str], List[str], List[List[Dict[str, Any]]]]:
        """
        Extract core information from the queryset that are needed later on for the 
        transformations.

        Outputs: Base Variable, Variable Name and Transformations applied to the Base Variable.
        """
        ops = self.queryset.model_dump()['operations']

        base_variables: list[str] = []
        var_names: list[str] = []
        transformation_list: list[list[dict[str, Any]]] = []

        for cand in ops:
            transformations: list[dict[str, Any]] = []

            for step in cand:
                match (step['namespace'], step['name']):
                    # record variable renames
                    case ('trf', 'util.rename'):
                        var_names.append(step['arguments'][0])

                    # record other trf-namespace transformations
                    case ('trf', other) if other != 'util.base':
                        transformations.append({
                            'name': step['name'],
                            'arguments': step['arguments'],
                        })

                    # record "base variables"
                    case ('base', _):
                        base_variables.append(step['name'])

            transformations.reverse()
            transformation_list.append(transformations)

        return base_variables, var_names, transformation_list

    # 2. -------------  LOAD THE EXTERNAL FILE  ---------------------------- #
    def _load_update_df(self) -> pd.DataFrame:
        """
        Loads the update dataframe containg Acled and GED data
        """
        if self.data_path.suffix == ".csv":
            return pd.read_csv(self.data_path)
        elif self.data_path.suffix in {".parquet", ".pq"}:
            return pd.read_parquet(self.data_path)
        else:
            msg = f"Unsupported file type: {self.data_path.suffix!r}"
            raise ValueError(msg)

    # 3. ------------  PREPROCESS THE UPDATE DF  ---------- #
    def preprocess_update_df(self, *, overwrite_external: bool = False) -> pd.DataFrame:
        """
        Adapts the update dataframe for the queryset and only keeps the columns and time stamps necessary

        Parameters
        ----------
        overwrite_external : bool, default False
            If True, replace self.df_external with the processed result.
        Returns
        -------
        pd.DataFrame
            The processed dataframe.
        """
       
        df_new = self.df_external

  
        last_parts = [
            s.rsplit(".", 1)[1] if "." in s else s
            for s in self.base_variables
        ]
        overlap = set(last_parts).intersection(df_new.columns)

        combined_subset = df_new[list(overlap)]

        # ------------------------------------- #
        # 2. keep only the requested months
        #    (assumes month_id is the index; adapt otherwise)
        # ------------------------------------- #
        df_new = combined_subset.loc[self.months_to_update]

        # ------------------------------------- #
        # 3. build the rename map (raw_* only)
        # ------------------------------------- #
        matching: dict[str, str] = {}
        for last, vname in zip(last_parts, self.var_names):
            if vname.startswith("raw_"):
                matching[last] = vname
            # else: transformed -- ignore for renaming

        self.last_parts = last_parts
        self.matching   = matching

        df_new = df_new.rename(columns=matching)

        # ------------------------------------- #
        # 4. optionally persist inside the object
        # ------------------------------------- #
        if overwrite_external:
            self.df_external = df_new

        return df_new

    # 4. ------------  APPLY THE TRANSFORMATIONS  ------------------------- #
    def _apply_all_transformations(self, df_old: pd.DataFrame) -> pd.DataFrame:
        """
        Re-build every *transformed* GED / ACLED variable described in the queryset in the right order.
        Operates **in-place** on `df_old` and returns it for convenience.
        """
        ix = pd.IndexSlice

        # Detect the group level (e.g., pg_id, country_id)
        group_level = next((lvl for lvl in df_old.index.names if lvl != 'month_id'), None)
        if not group_level:
            raise ValueError("Could not determine group level from MultiIndex")

        for idx, (var_name, transformations) in enumerate(zip(self.var_names, self.transformation_list)):
            # Skip non-ged/acled variables
            if not any(prefix in var_name for prefix in ("ged", "acled")):
                logger.debug(f"No Acled or GED variable: {var_name}")
                continue

            # Skip raw variables
            if var_name.startswith("raw_"):
                logger.debug(f"Raw Variable: {var_name}")
                continue

            # Skip if no transformations to apply
            if not transformations:
                logger.debug(f"No transformations: {var_name}")
                continue

            # Correctly fetch base variable
            base_var_key = self.last_parts[idx]
            base_var = self.matching.get(base_var_key)

            if not base_var:
                logger.warning(f"⚠️ Could not find base_var for {var_name} (from key '{base_var_key}')")
                continue
            if base_var not in df_old.columns:
                logger.warning(f"⚠️ base_var '{base_var}' not in df_old.columns for {var_name}")
                continue

            current_series = df_old[base_var]

            for transformation in transformations:
                name = transformation["name"]
                #args = list(map(int, transformation.get("arguments", [])))
                args = transformation.get("arguments", [])
                transform_func = transformation_mapping.get(name)

                if not transform_func:
                    raise ValueError(f"Unknown transformation: {name}")

                logger.info(f"Applying transformation {name} with args {args} to {var_name}")

                # Special case: spatial.countrylag
                if name == "spatial.countrylag":
                    logger.debug(f"Special transformation: {name}")
                    ffilled_col = current_series.groupby(level=group_level).ffill()
                    df_old.loc[ix[self.months_to_update, :], var_name] = ffilled_col.loc[ix[self.months_to_update, :]]
                    continue

                # Determine input shape: Series vs DataFrame
                if name in TRANSFORMATIONS_EXPECTING_DF:
                    input_data = current_series.to_frame()
                else:
                    input_data = current_series
                

                # Apply transformation
                try:
                    current_series = transform_func(input_data, *args) if args else transform_func(input_data)
                except Exception as e:
                    raise RuntimeError(f"Error applying {name} to {var_name}: {e}")

                # Optional: ensure index matches to prevent NaNs
                if not current_series.index.equals(df_old.index):
                    logger.warning(f"[WARNING] Index mismatch after {name} → reindexing")
                    current_series = current_series.reindex(df_old.index)

            # Final assignment to df
            df_old[var_name] = current_series

        return df_old


class ViewsDataLoader:
    """
    A class to handle data loading, fetching, and processing for different partitions.

    This class provides methods to fetch data from viewser, validate data partitions,
    create or load volumes, and handle drift detection configurations.
    """

    def __init__(self, model_path: ModelPathManager, partition_dict: Dict = None, steps: int = 36, **kwargs):
        """
        Initializes the DataLoaders class with a ModelPathManager object and optional keyword arguments.

        Args:
            model_path (ModelPathManager): An instance of the ModelPathManager class.
            **kwargs: Additional keyword arguments to set instance attributes.

        Attributes:
            partition (str, optional): The partition type. Defaults to None.
            partition_dict (dict, optional): The dictionary containing partition information. Defaults to None.
            drift_config_dict (dict, optional): The dictionary containing drift detection configuration. Defaults to None.
            override_month (str, optional): The override month. Defaults to None.
            month_first (str, optional): The first month in the range. Defaults to None.
            month_last (str, optional): The last month in the range. Defaults to None.
            steps (int, optional): The step size for the forecasting partition. Defaults to 36.
        """
        self._model_path = model_path
        self._model_name = model_path.model_name
        # if self._model_path.target == "model":
        self._path_raw = model_path.data_raw
        self._path_processed = model_path.data_processed
        self.partition = None
        self.partition_dict = partition_dict
        self.drift_config_dict = None
        self.override_month = None
        self.month_first, self.month_last = None, None
        self.steps = steps

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _get_partition_dict(self, steps) -> Dict:
        """
        Returns the partitioner dictionary for the given partition.

        Args:
            steps (int, optional): The step size for the forecasting partition. Defaults to 36.

        Returns:
            dict: A dictionary containing the train and predict ranges for the specified partition.

        Raises:
            ValueError: If the partition attribute is not one of "calibration", "validation", or "forecast".

        Notes:
            - For the "calibration" partition, the train range is (121, 396) and the test range is (397, 444).
            - For the "validation" partition, the train range is (121, 444) and the test range is (445, 492).
            - For the "forecasting" partition, the train range starts at 121 and ends at the current month minus 2.
              The predict range starts at the current month minus 1 and extends by the step size.
        """
        logger.warning("Did not use config_partitions.py, using default partition dictionary instead...")
        match self.partition:
            case "calibration":
                return {
                    "train": (121, 396),
                    "test": (397, 444),
                    }  # calib_partitioner_dict - (01/01/1990 - 12/31/2012) : (01/01/2013 - 31/12/2015)
            case "validation":
                return {
                    "train": (121, 444), 
                    "test": (445, 492)
                    }
            case "forecasting":
                month_last = (
                    ViewsMonth.now().id - 2
                )  # minus 2 because the current month is not yet available. Verified but can be tested by changing this and running the check_data notebook.
                return {
                    "train": (121, month_last),
                    "test": (month_last + 1, month_last + 1 + steps),
                }  
            case _:
                raise ValueError(
                    'partition should be either "calibration", "validation" or "forecasting"'
                )
        pass

    def _fetch_data_from_viewser(self, self_test: bool) -> tuple[pd.DataFrame, list]:
        """
        Fetches and prepares the initial DataFrame from viewser.

        Args:
            self_test (bool): Flag indicating whether to perform self-testing.
            target (str): The target variable.

        Returns:
            pd.DataFrame: The prepared DataFrame with initial processing done.
            list: List of alerts generated during data fetching.

        Raises:
            RuntimeError: If the queryset for the model is not found.
        """
        logger.info(
            f"Beginning file download through viewser with month range {self.month_first},{self.month_last}"
        )

        queryset_base = self._model_path.get_queryset()  # just used here..
        dotenv_path = self._model_path.find_project_root()/ 'ensembles' / '.env'
        logger.debug(f"Path to dotenv file: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)

        months_to_update = PipelineConfig().months_to_update
        logger.debug(f"months to update: {months_to_update}")
        loa_qs = queryset_base.model_dump()['loa']
        print(loa_qs)
        logger.debug(f"Level of Analysis: {loa_qs}")

        if loa_qs == 'priogrid_month':
            update_path = os.getenv("pgm_path")
        elif loa_qs =='country_month':
            update_path = os.getenv("cm_path")
        else:
            logger.warning("Unknown loa, no update path")

        print("Update path", update_path)

        if queryset_base is None:
            raise RuntimeError(
                f"Could not find queryset for {self._model_name}"
            )
        else:
            logger.info(f"Found queryset for {self._model_name}")

        try:
            df, alerts = queryset_base.publish().fetch_with_drift_detection(
                start_date=self.month_first,
                end_date=self.month_last - 1,
                drift_config_dict=self.drift_config_dict,
                self_test=self_test,
            )

            logger.info('NSPECTING DF FROM VIEWSER')
            for ialert, alert in enumerate(
            str(alerts).strip("[").strip("]").split("Input")):
                if "offender" in alert:
                    logger.warning(
                        {f"{self._model_path.model_name} data alert {ialert}": str(alert)}
                    )
            logger.info("Updating the VIEWSER DF")
            logger.debug(f"NaNs found in dataframe from viewser: {df.isna().sum()}")

            builder = UpdateViewser(queryset_base, viewser_df=df, data_path=update_path, months_to_update=months_to_update)
            df = builder.run()
            logger.info("VIEWSER UPDATE DONE")
            logger.debug(f"NaNs in df after transformations: {df.isna().sum()}")
            df = ensure_float64(df)
            return df, alerts
        
        except KeyError as e:
            logger.error(f"\033[91mError fetching data from viewser: {e}. Trying to fetch without drift detection.\033[0m", exc_info=True)
            df = queryset_base.publish().fetch(
                start_date=self.month_first,
                end_date=self.month_last - 1,
            )
            logger.info("Updating the VIEWSER DF after failed drift detection")
            builder = UpdateViewser(queryset_base, viewser_df=df, data_path = update_path, months_to_update=months_to_update)
            df = builder.run()
            logger.info("VIEWSER UPDATE DONE")
            logger.debug(f"NaNs in df after transformations: {df.isna().sum()}")
            df = ensure_float64(df)
            return df, None
        
        except Exception as e:
            logger.error(f"Error fetching data from viewser: {e}", exc_info=True)
            logger.error(traceback.format_exc())
            raise RuntimeError(
                f"Error fetching data from viewser: {e.with_traceback()}"
            )


    def _get_month_range(self) -> tuple[int, int]:
        """
        Determines the month range based on the partition type.

        Returns:
            tuple: The start and end month IDs for the partition.

        Raises:
            ValueError: If partition is not 'calibration', 'validation', or 'forecasting'.
        """
        month_first = self.partition_dict["train"][0]

        if self.partition == "forecasting":
            month_last = self.partition_dict["train"][1] + 1
        elif self.partition in ["calibration", "validation"]:
            month_last = self.partition_dict["test"][1] + 1
        else:
            raise ValueError(
                'partition should be either "calibration", "validation" or "forecasting"'
            )
        if self.partition == "forecasting" and self.override_month is not None:
            month_last = self.override_month
            logger.warning(
                f"Overriding end month in forecasting partition to {month_last} ***\n"
            )

        return month_first, month_last

    def _validate_df_partition(
        self, df: pd.DataFrame
    ) -> bool:
        """
        Checks to see if the min and max months in the input dataframe are the same as the min
        month in the train and max month in the predict portions (or min and max months in the train portion for
        the forecasting partition).

        Args:
            df (pd.DataFrame): The dataframe to be checked.
            partition_dict (Dict): The partition dictionary.
            override_month (int, optional): If user has overridden the end month of the forecasting partition, this value
                                            is substituted for the last month in the forecasting train portion.

        Returns:
            bool: True if the dataframe is valid for the partition, False otherwise.
        """
        if "month_id" in df.columns:
            df_time_units = df["month_id"].values
        else:
            df_time_units = df.index.get_level_values("month_id").values
        # partitioner_dict = get_partitioner_dict(partition)
        if self.partition in ["calibration", "validation"]:
            first_month = self.partition_dict["train"][0]
            last_month = self.partition_dict["test"][1]
        else:
            first_month = self.partition_dict["train"][0]
            last_month = self.partition_dict["train"][1]
            if self.override_month is not None:
                last_month = self.override_month - 1
        if [np.min(df_time_units), np.max(df_time_units)] != [first_month, last_month]:
            return False
        else:
            return True

    @staticmethod
    def filter_dataframe_by_month_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the DataFrame to include only the specified month range.

        Args:
            df (pd.DataFrame): The input DataFrame to be filtered.
            month_first (int): The first month ID to include.
            month_last (int): The last month ID to include.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        month_range = np.arange(self.month_first, self.month_last)
        return df[df["month_id"].isin(month_range)].copy()

    def get_data(
        self,
        self_test: bool,
        partition: str,
        use_saved: bool,
        validate: bool = True,
        override_month: int = None,
    ) -> tuple[pd.DataFrame, list]:
        """
        Fetches or loads a DataFrame for a given partition from viewser.

        This function handles the retrieval or loading of raw data for the specified partition.

        The default behaviour is to fetch fresh data via viewser. This can be overridden by setting the
        used_saved flag to True, in which case saved data is returned, if it can be found.

        Args:
            self_test (bool): Flag indicating whether to perform self-testing.
            partition (str): The partition type.
            use_saved (bool, optional): Flag indicating whether to use saved data if available.
            validate (bool, optional): Flag indicating whether to validate the fetched data. Defaults to True.
            override_month (int, optional): If provided, overrides the end month for the forecasting partition.

        Returns:
            pd.DataFrame: The DataFrame fetched or loaded from viewser, with minimum preprocessing applied.
            list: List of alerts generated during data fetching.

        Raises:
            RuntimeError: If the saved data file is not found or if the data is incompatible with the partition.
        """
        self.partition = partition #if self.partition is None else self.partition
        self.partition_dict = self._get_partition_dict(steps=self.steps) if self.partition_dict is None else self.partition_dict.get(partition, None)
        self.drift_config_dict = drift_detection.drift_detection_partition_dict[
            partition
        ] if self.drift_config_dict is None else self.drift_config_dict
        self.override_month = override_month if self.override_month is None else override_month
        if self.month_first is None or self.month_last is None:
            self.month_first, self.month_last = self._get_month_range()

        path_viewser_df = Path(
            os.path.join(str(self._path_raw), f"{self.partition}_viewser_df{PipelineConfig.dataframe_format}")
        )  
        alerts = None

        if use_saved:
            if path_viewser_df.exists():
                try:
                    df = read_dataframe(path_viewser_df)
                    logger.info(f"Reading saved data from {path_viewser_df}")
                except Exception as e:
                    raise RuntimeError(
                        f"Use of saved data was specified but getting {path_viewser_df} failed with: {e}"
                    )
            else:
                logger.info(f"Saved data not found at {path_viewser_df}, fetching from viewser...")
                df, alerts = self._fetch_data_from_viewser(self_test)
                data_fetch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                create_data_fetch_log_file(
                    self._path_raw, self.partition, self._model_name, data_fetch_timestamp
                )
                logger.info(f"Saving data to {path_viewser_df}")
                save_dataframe(df, path_viewser_df)
        else:
            logger.info(f"Fetching data from viewser...")
            df, alerts = self._fetch_data_from_viewser(self_test) 
            data_fetch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            create_data_fetch_log_file(
                self._path_raw, self.partition, self._model_name, data_fetch_timestamp
            )
            logger.info(f"Saving data to {path_viewser_df}")
            save_dataframe(df, path_viewser_df)
            
        if validate:
            if self._validate_df_partition(df=df):
                return df, alerts
            else:
                raise RuntimeError(
                    f"file {path_viewser_df.name} incompatible with partition {self.partition}"
                )
        logger.debug(f"DataFrame shape: {df.shape if df is not None else 'None'}")
        for ialert, alert in enumerate(
            str(alerts).strip("[").strip("]").split("Input")
        ):
            if "offender" in alert:
                logger.warning({f"{partition} data alert {ialert}": str(alert)})

        # df = df.reset_index()
        # if "priogrid_gid" in df.columns():
        #     df = df.rename(columns={"priogrid_gid": "priogrid_id"})
        #     df = df.set_index(["month_id", "priogrid_id"])

        return df, alerts
    