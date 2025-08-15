import os
from dotenv import load_dotenv
from typing import Dict, Tuple, List, Any
import ast
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from views_pipeline_core.configs import drift_detection
from views_pipeline_core.files.utils import create_data_fetch_log_file
from views_pipeline_core.data.utils import ensure_float64
from views_pipeline_core.files.utils import read_dataframe, save_dataframe
from views_pipeline_core.cli.utils import parse_args
import argparse
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.managers.model import ModelPathManager
from ingester3.ViewsMonth import ViewsMonth

# import views_transformation_library as vtl
import views_transformation_library.views_2 as views2
import views_transformation_library.splag4d as splag4d
import views_transformation_library.missing as missing
from viewser import Queryset
import traceback
import views_transformation_library.splag_country as splag_country
import views_transformation_library.spatial_tree as spatial_tree
import views_transformation_library.spacetime_distance as spacetime_distance

logger = logging.getLogger(__name__)


transformation_mapping = {
    "ops.ln": views2.ln,
    "missing.fill": missing.fill,
    "bool.gte": views2.greater_or_equal,
    "temporal.time_since": views2.time_since,
    "temporal.decay": views2.decay,
    "missing.replace_na": missing.replace_na,
    "spatial.countrylag": splag_country.get_splag_country,
    "temporal.tlag": views2.tlag,
    "spatial.lag": splag4d.get_splag4d,
    "spatial.treelag": spatial_tree.get_tree_lag,
    "spatial.sptime_dist": spacetime_distance.get_spacetime_distances,
    "temporal.moving_sum": views2.moving_sum,
    "temporal.moving_average": views2.moving_sum,
}

# The TRANSFORMATIONS_EXPECTING_DF set lists transformation names that require a DataFrame as input,
# rather than a Series. This is important for handling transformations that operate on multiple columns
# or require access to the full DataFrame structure. When applying these transformations, the code
# ensures that the input is converted to a DataFrame before calling the transformation function.
TRANSFORMATIONS_EXPECTING_DF = {"spatial.lag", "spatial.sptime_dist"}


class UpdateViewser:
    """
    A class to update VIEWSER dataframes with new GED and ACLED values.

    This class parses a queryset to extract base variable names, transformation sequences, and output variable names.
    It preprocesses an external update dataframe to align columns and rows with the viewser dataframe, updates raw values,
    and applies all required transformations as specified in the queryset.

    The workflow:
    1. Extracts variable and transformation metadata from the queryset.
    2. Loads and preprocesses the external update dataframe for the specified months.
    3. Updates the raw columns in the viewser dataframe with new values.
    4. Applies all transformations to produce the final variables.

    This ensures the viewser dataframe is up-to-date and consistent with the latest external data and transformation logic.
    """

    def __init__(
        self,
        queryset: Queryset,
        viewser_df: pd.DataFrame,
        data_path: str | Path,
        months_to_update: List[int],
    ):
        """
        Initializes the UpdateViewser class with a queryset, a dataframe from viewser (with missed updated),
        a path to the dataframe with updates from GED & Acled and a list of months to update.

        Args:
            queryset: The queryset for the respective model.
            viewser_df: a dataframe from VIEWSER that does not contain the latest updates for GED & Acled data.
            data_path: the path to the update dataframe with updates for GED and Acled data.
            months_to_update: A list of months

        This class initializes by extracting the following information from the queryset:
        - Base_Variables: A list of column names based on the 'from_column' e.g.: 'country_month.ged_sb_best_sum_nokgi'.
          Columns in the update df have the last part of base_variables as their column names e.g. 'ged_sb_best_sum_nokgi'.
        - Var_Names: A list of column names after transformations have been applied.
          Columns in the VIEWSER df have var_names as their column names e.g. 'raw_ged_sb'.
        - Transformation_list: A list of lists of transformations applied to Base_Variables to produce
        """

        self.queryset = queryset
        self.viewser_df = viewser_df
        self.data_path = Path(data_path)
        self.months_to_update = list(months_to_update)

        (self.base_variables, self.var_names, self.transformation_list) = (
            self._extract_from_queryset()
        )

        if not any(var.startswith("raw_") for var in self.var_names):
            raise ValueError(
                "Queryset does not contain any variable staring with raw_. "
                "At least one raw_ variable is required to update the viewser df."
            )

        # self.df_external = self._load_update_df()
        self.df_external = read_dataframe(self.data_path)

        max_month_id_viewser = self.viewser_df.index.get_level_values("month_id").max()
        max_month_id_external = self.df_external.index.get_level_values(
            "month_id"
        ).max()
        logger.info(f"Max month_id: viewser_df={max_month_id_viewser}")
        logger.info(f"Max month_id: update_df={max_month_id_external}")

        if max_month_id_viewser > max_month_id_external:
            raise ValueError(
                f"Max month_id mismatch: viewser_df={max_month_id_viewser}, "
                f"update dataframe={max_month_id_external}, "
                f"Make sure to get the latest update dataframe! "
            )

        self.result: pd.DataFrame | None = None  # filled by .run()

    def run(self) -> pd.DataFrame:
        """
        Function executes all frunctions in the right order to update the dataframe from VIEWSER. Safe to call twice:
        we memorise the result after first run.
        """
        if self.result is not None:
            logger.debug("Use saved dataframe")  # already done
            return self.result

        # 1) Adapt update df to queryset and month_ids to update
        df_update = self._preprocess_update_df()

        # 2) Update df from viewser
        # df = self.queryset.publish().fetch()
        self.viewser_df.update(df_update)

        logger.info("Fetched and updated from viewser")

        # 3) Apply transformations
        df_final = self._apply_all_transformations(df_old=self.viewser_df)
        logger.info("All transformations done")

        cols_to_drop = df_final.columns[df_final.columns.str.startswith("raw")]
        df_final = df_final.drop(columns=cols_to_drop)

        # 4)return
        return df_final

    # 1. -------------  PARSE THE QUERYSET  -------------------------------- #
    def _extract_from_queryset(
        self,
    ) -> Tuple[List[str], List[str], List[List[Dict[str, Any]]]]:
        """
        Extracts core information from the queryset that are needed later on for the
        transformations. For every line in the queryset it records the 'from_column' as
        Base Variable, 'Column' as Variable Name and all Transformations applied to the Base
        Variable.

        Outputs: Base Variable, Variable Name and Transformations applied to the Base Variable.
        """
        ops = self.queryset.model_dump()["operations"]

        base_variables: list[str] = []
        var_names: list[str] = []
        transformation_list: list[list[dict[str, Any]]] = []

        for cand in ops:
            transformations: list[dict[str, Any]] = []

            for step in cand:
                match (step["namespace"], step["name"]):
                    # record variable renames
                    case ("trf", "util.rename"):
                        var_names.append(step["arguments"][0])

                    # record other trf-namespace transformations
                    case ("trf", other) if other != "util.base":
                        transformations.append(
                            {
                                "name": step["name"],
                                "arguments": step["arguments"],
                            }
                        )

                    # record "base variables"
                    case ("base", _):
                        base_variables.append(step["name"])

            transformations.reverse()
            transformation_list.append(transformations)

        return base_variables, var_names, transformation_list
    
    # 2. ------------  PREPROCESS THE UPDATE DF  ---------- #
    def _preprocess_update_df(
        self, *, overwrite_external: bool = False
    ) -> pd.DataFrame:
        """
        Processes the external update dataframe (`self.df_external`) to align it with the variables and months required for updating the viewser dataframe.

        Steps:
        1. Extracts the base variable names from the queryset, removing any LOA prefixes (e.g., 'country_month.ged_sb_best_sum_nokgi' → 'ged_sb_best_sum_nokgi').
        2. Identifies which base variables have corresponding updates in the external dataframe (overlap).
        3. Filters the external dataframe to retain only the overlapping columns.
        4. Selects only the rows for the specified `months_to_update`.
        5. Builds a mapping from base variable names to their corresponding raw variable names in the viewser dataframe (e.g., 'ged_sb_best_sum_nokgi' → 'raw_ged_sb').
        6. Renames the columns in the update dataframe using this mapping so they match the viewser dataframe.
        7. Optionally replaces `self.df_external` with the processed dataframe if `overwrite_external` is True.

        Parameters
        ----------
        overwrite_external : bool, default False
            If True, replaces self.df_external with the processed result.

        Returns
        -------
        pd.DataFrame
            The processed and aligned update dataframe, ready to be used for updating the viewser dataframe.
        """

        df_new = self.df_external

        # 1. For each string in self.base_variables (which are typically fully-qualified variable names like 'country_month.ged_sb_best_sum_nokgi'),
        #    it splits the string at the last period ('.') and takes the part after the period. If there is no period, it uses the whole string.
        #    This produces a list of "base" variable names (e.g., 'ged_sb_best_sum_nokgi') that match the column names in the external update dataframe.
        #
        # 2. It then computes the intersection between these extracted base variable names and the columns present in df_new (the external update dataframe).
        #    This ensures that only variables present in both the queryset and the update dataframe are considered for further processing.
        #
        # 3. Finally, it creates a new dataframe (combined_subset) containing only the columns from df_new that are present in the overlap set.
        #    This filters the external dataframe down to just the relevant columns that can be used for updating the viewser dataframe.
        # This is dangerous!
        last_parts = [
            s.rsplit(".", 1)[1] if "." in s else s for s in self.base_variables
        ]
        overlap = set(last_parts).intersection(df_new.columns)
        if not overlap:
            raise ValueError(
                "No overlapping columns found between base variables and update dataframe. "
                "Check if the update dataframe contains the expected columns."
            )  # D: Check if the update dataframe contains the expected columns.

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
        self.matching = matching

        df_new = df_new.rename(columns=matching)

        # ------------------------------------- #
        # 4. optionally persist inside the object
        # ------------------------------------- #
        if overwrite_external:
            self.df_external = df_new

        return df_new

    def _smart_cast(self, arg):
        """
        Safely converts a string representation of a Python literal to its corresponding Python object.

        Attempts to evaluate the input `arg` using `ast.literal_eval`, which can parse valid Python literals
        such as numbers, lists, dictionaries, booleans, etc. If the evaluation is successful, returns the
        resulting Python object. If the input is not a valid literal or is already a non-string type,
        returns the original argument unchanged.

        Args:
            arg: The input to be converted, typically a string representation of a Python literal.

        Returns:
            The evaluated Python object if conversion is successful; otherwise, the original input.
        """
        try:
            return ast.literal_eval(arg)
        except Exception:
            return arg

    # 3. ------------  APPLY THE TRANSFORMATIONS  ------------------------- #
    def _apply_all_transformations(self, df_old: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all required transformations to GED/ACLED variables as described in the queryset.

        For each variable:
        - Skips non-GED/ACLED and raw variables.
        - Applies the sequence of transformations in the correct order.
        - Handles special cases (e.g., spatial.countrylag).
        - Ensures index alignment after transformation.
        - Assigns the final transformed series to the output DataFrame.

        Operates in-place on `df_old` and returns it.
        """
        ix = pd.IndexSlice

        # Detect the group level (e.g., pg_id, country_id)
        group_level = next(
            (lvl for lvl in df_old.index.names if lvl != "month_id"), None
        )
        if not group_level:
            raise ValueError("Could not determine group level from MultiIndex")

        for idx, (var_name, transformations) in enumerate(
            zip(self.var_names, self.transformation_list)
        ):
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
                logger.warning(
                    f"⚠️ Could not find base_var for {var_name} (from key '{base_var_key}')"
                )
                continue
            if base_var not in df_old.columns:
                logger.warning(
                    f"⚠️ base_var '{base_var}' not in df_old.columns for {var_name}"
                )
                continue

            current_series = df_old[base_var]

            for transformation in transformations:
                name = transformation["name"]

                # args = list(map(int, transformation.get("arguments", [])))
                # args = [smart_cast(arg) for arg in transformation.get("arguments", [])]
                # args = transformation.get("arguments", [])
                args = [
                    self._smart_cast(arg) for arg in transformation.get("arguments", [])
                ]
                transform_func = transformation_mapping.get(name)

                if not transform_func:
                    raise ValueError(f"Unknown transformation: {name}")

                logger.info(
                    f"Applying transformation {name} with args {args} to {var_name}"
                )

                # Special case: spatial.countrylag
                if name == "spatial.countrylag":
                    logger.debug(f"Special transformation: {name}")
                    ffilled_col = current_series.groupby(level=group_level).ffill()
                    df_old.loc[ix[self.months_to_update, :], var_name] = (
                        ffilled_col.loc[ix[self.months_to_update, :]]
                    )
                    continue

                # Determine input shape: Series vs DataFrame
                if name in TRANSFORMATIONS_EXPECTING_DF:
                    input_data = current_series.to_frame()
                else:
                    input_data = current_series

                # Apply transformation
                try:
                    current_series = (
                        transform_func(input_data, *args)
                        if args
                        else transform_func(input_data)
                    )
                except Exception as e:
                    raise RuntimeError(f"Error applying {name} to {var_name}: {e}")

                # Optional: ensure index matches to prevent NaNs
                if not current_series.index.equals(df_old.index):
                    logger.warning(
                        f"[WARNING] Index mismatch after {name} → reindexing"
                    )
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

    def __init__(
        self,
        model_path: ModelPathManager,
        partition_dict: Dict = None,
        steps: int = 36,
        **kwargs,
    ):
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
        logger.warning(
            "Did not use config_partitions.py, using default partition dictionary instead..."
        )
        match self.partition:
            case "calibration":
                return {
                    "train": (121, 396),
                    "test": (397, 444),
                }  # calib_partitioner_dict - (01/01/1990 - 12/31/2012) : (01/01/2013 - 31/12/2015)
            case "validation":
                return {"train": (121, 444), "test": (445, 492)}
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

    def _get_viewser_update_config(self, queryset_base: Queryset) -> tuple[int, str]:
        """
        Retrieves the update configuration for the viewser dataset based on the provided queryset.

        This method:
        - Locates and loads the `.env` file from the project root.
        - Extracts the list of months to update from the environment variable.
        - Determines the update file path according to the queryset's level of analysis (LOA).

        Args:
            queryset_base (Queryset): The queryset object, expected to have a `model_dump()` method with an 'loa' key.

        Returns:
            tuple[list[int], str | None]: A tuple containing the months to update (as a list of ints)
                          and the update file path (str or None if LOA is unknown).

        Raises:
            FileNotFoundError: If the `.env` file is not found.
            RuntimeError: If the `.env` file cannot be loaded.
        """
        dotenv_path = self._model_path.find_project_root() / ".env"
        logger.debug(f"Path to dotenv file: {dotenv_path}")

        if not dotenv_path.exists():
            raise FileNotFoundError(f"Required .env file not found: {dotenv_path}")

        if not load_dotenv(dotenv_path=dotenv_path):
            raise RuntimeError(
                f".env file found but could not be loaded: {dotenv_path}"
            )

        # months_to_update = PipelineConfig().months_to_update #read from .env
        months_to_update_str = os.getenv("month_to_update")
        if not months_to_update_str or months_to_update_str is "":
            raise ValueError("Could not find months to update in the .env file. Add the line: month_to_update=[123, 124, 125]")

        months_to_update = ast.literal_eval(months_to_update_str)
        logger.debug(f"Months to update: {months_to_update}")

        loa_qs = queryset_base.model_dump()["loa"]
        logger.debug(f"Level of Analysis: {loa_qs}")

        if loa_qs == "priogrid_month":
            update_path = os.getenv("pgm_path")
        elif loa_qs == "country_month":
            update_path = os.getenv("cm_path")
        else:
            logger.warning("Unknown LOA; no update path set")
            update_path = None

        logger.debug(f"Update path: {update_path}")
        return months_to_update, update_path

    def _overwrite_viewser(
        self, df: pd.DataFrame, queryset_base: Queryset, args: argparse.Namespace
    ) -> pd.DataFrame:
        """
        Updates the provided DataFrame using the Viewser update process if specified in the arguments.

        If `args.update_viewser` is True, this method retrieves the update configuration,
        initializes an UpdateViewser instance, and updates the DataFrame accordingly.
        Logs the update process and the number of NaN values after transformation.
        If updating is not requested, logs that the DataFrame has not been updated.

        Args:
            df (pd.DataFrame): The DataFrame to potentially update.
            queryset_base (Any): The base queryset used for configuration and updating.
            args (Namespace): Arguments containing the `update_viewser` flag.

        Returns:
            pd.DataFrame: The (possibly updated) DataFrame.
        """
        if args.update_viewser:
            logger.info(
                "Overwriting Viewser dataframe with new values from GED and ACLED"
            )
            months_to_update, update_path = self._get_viewser_update_config(
                queryset_base
            )
            builder = UpdateViewser(
                queryset_base,
                viewser_df=df,
                data_path=update_path,
                months_to_update=months_to_update,
            )
            df = builder.run()
            logger.info("Viewser dataframe updated")
            logger.debug(f"NaNs in df after transformations: {df.isna().sum()}")
        else:
            logger.info("Viewser dataframe will not be overwritten")
        return df

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

        if queryset_base is None:
            raise RuntimeError(f"Could not find queryset for {self._model_name}")
        else:
            logger.info(f"Found queryset for {self._model_name}")

        args = parse_args()
        # dotenv_path = self._model_path.find_project_root()/ 'ensembles' / '.env'
        df, alerts = None, None

        try:
            df, alerts = queryset_base.publish().fetch_with_drift_detection(
                start_date=self.month_first,
                end_date=self.month_last - 1,
                drift_config_dict=self.drift_config_dict,
                self_test=self_test,
            )

            for ialert, alert in enumerate(
                str(alerts).strip("[").strip("]").split("Input")
            ):
                if "offender" in alert:
                    logger.warning(
                        {
                            f"{self._model_path.model_name} data alert {ialert}": str(
                                alert
                            )
                        }
                    )
            # df = self._overwrite_viewser(df, queryset_base, args)
            # df = ensure_float64(df)
        except KeyError as e:
            logger.error(
                f"\033[91mError fetching data from viewser: {e}. Trying to fetch without drift detection.\033[0m",
                exc_info=True,
            )
            df = queryset_base.publish().fetch(
                start_date=self.month_first,
                end_date=self.month_last - 1,
            )
        except Exception as e:
            logger.error(f"Error fetching data from viewser: {e}", exc_info=True)
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Error fetching data from viewser: {e}") from e
        
        df = self._overwrite_viewser(df, queryset_base, args)
        df = ensure_float64(df)
        return df, alerts

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

    def _validate_df_partition(self, df: pd.DataFrame) -> bool:
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
        self.partition = partition  # if self.partition is None else self.partition
        self.partition_dict = (
            self._get_partition_dict(steps=self.steps)
            if self.partition_dict is None
            else self.partition_dict.get(partition, None)
        )
        self.drift_config_dict = (
            drift_detection.drift_detection_partition_dict[partition]
            if self.drift_config_dict is None
            else self.drift_config_dict
        )
        self.override_month = (
            override_month if self.override_month is None else override_month
        )
        if self.month_first is None or self.month_last is None:
            self.month_first, self.month_last = self._get_month_range()

        path_viewser_df = Path(
            os.path.join(
                str(self._path_raw),
                f"{self.partition}_viewser_df{PipelineConfig.dataframe_format}",
            )
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
                logger.info(
                    f"Saved data not found at {path_viewser_df}, fetching from viewser..."
                )
                df, alerts = self._fetch_data_from_viewser(self_test)
                data_fetch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                create_data_fetch_log_file(
                    self._path_raw,
                    self.partition,
                    self._model_name,
                    data_fetch_timestamp,
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
