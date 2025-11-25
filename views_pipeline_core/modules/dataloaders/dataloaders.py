import os
from typing import Dict
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

# import views_transformation_library as vtl
import views_transformation_library.views_2 as views2
# import views_transformation_library.splag4d as splag4d
import views_transformation_library.missing as missing
from viewser import Queryset
import traceback
# import views_transformation_library.splag_country as splag_country
# import views_transformation_library.spatial_tree as spatial_tree
# import views_transformation_library.spacetime_distance as spacetime_distance
from dotenv import load_dotenv
from typing import Dict, Tuple, List, Any
import ast
import argparse
from views_pipeline_core.cli.utils import parse_args

logger = logging.getLogger(__name__)


# Ingester dependent imports. Breaks tests on github because no certs
def _get_splag_country(*args, **kwargs):
    import views_transformation_library.splag_country as splag_country
    return splag_country.get_splag_country(*args, **kwargs)

def _get_splag4d(*args, **kwargs):
    import views_transformation_library.splag4d as splag4d
    return splag4d.get_splag4d(*args, **kwargs)

def _get_spatial_tree(*args, **kwargs):
    import views_transformation_library.spatial_tree as spatial_tree
    return spatial_tree.get_tree_lag(*args, **kwargs)

def _get_spacetime_distance(*args, **kwargs):
    import views_transformation_library.spacetime_distance as spacetime_distance
    return spacetime_distance.get_spacetime_distances(*args, **kwargs)

transformation_mapping = {
    "ops.ln": views2.ln,
    "missing.fill": missing.fill,
    "bool.gte": views2.greater_or_equal,
    "temporal.time_since": views2.time_since,
    "temporal.decay": views2.decay,
    "missing.replace_na": missing.replace_na,
    "spatial.countrylag": _get_splag_country,
    "temporal.tlag": views2.tlag,
    "spatial.lag": _get_splag4d,
    "spatial.treelag": _get_spatial_tree,
    "spatial.sptime_dist": _get_spacetime_distance,
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
    Update VIEWSER dataframes with latest GED and ACLED data.

    Applies queryset transformations to update existing VIEWSER data with
    new values from external sources. Handles raw variable updates and
    recomputes all downstream transformations to maintain consistency.

    The workflow:
    1. Parses queryset to extract base variables, transformations, and output names
    2. Loads and preprocesses external update data
    3. Updates raw columns in VIEWSER dataframe
    4. Reapplies all transformations in correct sequence
    5. Returns updated dataframe ready for model consumption

    Supports:
    - Temporal transformations (lags, moving averages, decay)
    - Spatial transformations (country lags, grid lags, spatial trees)
    - Missing value handling and imputation
    - Mathematical operations (log transforms, boolean operations)

    Attributes:
        queryset (Queryset): Model queryset defining transformations
        viewser_df (pd.DataFrame): VIEWSER data to update
        data_path (Path): Path to external update data
        months_to_update (List[int]): Month IDs to update
        base_variables (List[str]): Raw input variable names
        var_names (List[str]): Final output variable names
        transformation_list (List[List[Dict]]): Transformation sequences
        df_external (pd.DataFrame): External update data
        result (Optional[pd.DataFrame]): Cached update result

    Example:
        >>> from viewser import Queryset
        >>> queryset = Queryset.from_file('config_queryset.py')
        >>> viewser_df = pd.read_parquet('viewser_data.parquet')
        >>> updater = UpdateViewser(
        ...     queryset=queryset,
        ...     viewser_df=viewser_df,
        ...     data_path='updates/ged_acled_latest.parquet',
        ...     months_to_update=[528, 529, 530]
        ... )
        >>> updated_df = updater.run()
        >>> print(f"Updated {len(updated_df)} rows")

    Note:
        - Requires at least one raw_ variable in queryset
        - External data must cover specified months_to_update
        - Updates applied in-place to viewser_df
        - Safe to call run() multiple times (result cached)
    """


    def __init__(
        self,
        queryset: Queryset,
        viewser_df: pd.DataFrame,
        data_path: str | Path,
        months_to_update: List[int],
    ):
        """
        Initialize UpdateViewser with queryset, data, and update configuration.

        Sets up update infrastructure by parsing queryset, loading external data,
        and validating temporal alignment between VIEWSER and update data.

        Args:
            queryset: Model queryset defining variables and transformations.
                Must contain at least one variable starting with 'raw_'
            viewser_df: VIEWSER DataFrame to update. Should have MultiIndex
                with 'month_id' and entity ID (country_id or priogrid_id)
            data_path: Path to external update file (parquet format).
                Must contain columns matching queryset base variables
            months_to_update: Month IDs to update (e.g., [528, 529, 530]).
                Must be present in both viewser_df and external data

        Raises:
            ValueError: If queryset doesn't contain any raw_ variables
            ValueError: If max month_id in viewser_df exceeds external data
                (indicates outdated update file)
            FileNotFoundError: If data_path doesn't exist

        Example:
            >>> queryset = Queryset.from_file('configs/config_queryset.py')
            >>> viewser_df = pd.read_parquet('data/viewser.parquet')
            >>> updater = UpdateViewser(
            ...     queryset=queryset,
            ...     viewser_df=viewser_df,
            ...     data_path='data/ged_acled_updates.parquet',
            ...     months_to_update=[528, 529]
            ... )
            INFO: Max month_id: viewser_df=527
            INFO: Max month_id: update_df=529

        Note:
            - External data should be newer than VIEWSER data
            - Result is None until run() is called
            - Parses queryset immediately to validate structure
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
        Execute complete update workflow to refresh VIEWSER data.

        Applies external updates to raw variables and recomputes all
        downstream transformations. Safe to call multiple times as
        result is cached after first execution.

        Execution Flow:
            1. Check if already run (return cached result)
            2. Preprocess external data to match queryset structure
            3. Update raw variables in VIEWSER dataframe
            4. Reapply all queryset transformations in sequence
            5. Drop temporary raw_ columns
            6. Cache and return updated dataframe

        Returns:
            Updated VIEWSER DataFrame with:
                - Raw variables updated for specified months
                - All transformations recomputed
                - Original structure preserved
                - Raw columns removed (only transformed remain)

        Example:
            >>> updater = UpdateViewser(queryset, viewser_df, data_path, [528, 529])
            >>> # First call executes update
            >>> df1 = updater.run()
            INFO: Fetched and updated from viewser
            INFO: All transformations done
            >>> # Second call returns cached result
            >>> df2 = updater.run()
            DEBUG: Use saved dataframe
            >>> assert df1 is df2  # Same object

        Performance:
            - First run: Depends on data size and transformation count
                Typical: 10-60 seconds for full dataset
            - Subsequent runs: <1ms (cached result)

        Note:
            - Updates applied in-place to self.viewser_df
            - Result cached in self.result
            - Raw columns dropped from final output
            - Transformations applied in queryset order
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
        Parse queryset to extract variables and transformations.

        Analyzes queryset operations to build three parallel lists that
        define the complete transformation pipeline for each variable.

        Internal Use:
            Called by __init__() to parse queryset structure.

        Returns:
            Tuple of three lists (same length):
                - base_variables: Source column names from 'base' namespace
                    Example: ['country_month.ged_sb_best_sum_nokgi']
                - var_names: Output column names after rename
                    Example: ['raw_ged_sb', 'ln_ged_sb_tlag_1']
                - transformation_list: List of transformation sequences
                    Example: [[{'name': 'ops.ln', 'arguments': []}]]

        Parsing Rules:
            - 'base' operations → base_variables
            - 'trf.util.rename' → var_names
            - Other 'trf' operations → transformation_list
            - Operations processed in reverse queryset order

        Example:
            >>> base_vars, names, transforms = self._extract_from_queryset()
            >>> print(base_vars[0])
            'country_month.ged_sb_best_sum_nokgi'
            >>> print(names[0])
            'raw_ged_sb'
            >>> print(transforms[0])
            [{'name': 'ops.ln', 'arguments': []}]

        Note:
            - Each queryset line produces one entry in each list
            - Transformations stored in application order
            - 'util.base' operations skipped (metadata only)
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
        Prepare external update data to match VIEWSER structure.

        Filters external data to relevant columns and months, then renames
        columns to match VIEWSER's raw_ variable naming convention.

        Internal Use:
            Called by run() to preprocess external updates before merging.

        Args:
            overwrite_external: If True, replaces self.df_external with result.
                Use with caution - mainly for testing. Default: False

        Returns:
            Preprocessed DataFrame with:
                - Only overlapping columns from base_variables
                - Only rows for months_to_update
                - Columns renamed to match raw_ variable names
                - Same index structure as viewser_df

        Processing Steps:
            1. Extract base names from fully-qualified variables
                'country_month.ged_sb' → 'ged_sb'
            2. Find overlap between base names and external columns
            3. Filter to overlapping columns only
            4. Filter to specified months_to_update
            5. Build mapping: base_name → raw_variable_name
            6. Rename columns using mapping
            7. Optionally overwrite self.df_external

        Example:
            >>> df_update = self._preprocess_update_df()
            >>> print(df_update.columns)
            Index(['raw_ged_sb', 'raw_ged_os', 'raw_acled_count'])
            >>> print(df_update.index.names)
            ['month_id', 'country_id']

        Raises:
            ValueError: If no overlapping columns found between
                queryset variables and external data

        Note:
            - Only processes raw_ variables (transformations computed later)
            - Preserves MultiIndex structure from external data
            - Column overlap determined by suffix matching
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
        Safely convert string arguments to Python literals.

        Attempts to parse string representations of Python objects
        (numbers, lists, dicts, etc.) into actual Python types.

        Internal Use:
            Called during transformation argument processing.

        Args:
            arg: Input to convert, typically transformation argument.
                Can be any type; strings attempted for conversion.

        Returns:
            Evaluated Python object if conversion successful,
            otherwise original input unchanged.

        Example:
            >>> self._smart_cast("123")
            123
            >>> self._smart_cast("[1, 2, 3]")
            [1, 2, 3]
            >>> self._smart_cast("{'key': 'value'}")
            {'key': 'value'}
            >>> self._smart_cast("not_a_literal")
            'not_a_literal'

        Note:
            - Uses ast.literal_eval for safe evaluation
            - No arbitrary code execution (safe)
            - Returns original on conversion failure
        """
        try:
            return ast.literal_eval(arg)
        except Exception:
            return arg

    # 3. ------------  APPLY THE TRANSFORMATIONS  ------------------------- #
    def _apply_all_transformations(self, df_old: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all queryset transformations to updated data.

        Recomputes all derived variables by applying transformation sequences
        to updated raw variables. Handles special cases like spatial lags and
        ensures index alignment throughout.

        Internal Use:
            Called by run() after raw variable updates.

        Args:
            df_old: VIEWSER DataFrame with updated raw values.
                Must have MultiIndex (month_id, entity_id)

        Returns:
            DataFrame with all transformations applied.
            Contains both raw and transformed variables.

        Transformation Handling:
            - Skips non-GED/ACLED variables (untouched)
            - Skips raw_ variables (already updated)
            - Applies transformations in queryset order
            - Special handling for spatial.countrylag (forward fill)
            - Reindexes after each transformation for alignment

        Example:
            >>> df_updated = self._apply_all_transformations(viewser_df)
            INFO: Applying transformation ops.ln to ln_ged_sb
            INFO: Applying transformation temporal.tlag to ln_ged_sb_tlag_1
            >>> print(df_updated.columns)
            Index(['raw_ged_sb', 'ln_ged_sb', 'ln_ged_sb_tlag_1'])

        Raises:
            RuntimeError: If transformation fails to apply
            ValueError: If unknown transformation name encountered

        Note:
            - Operates in-place on df_old
            - Uses transformation_mapping for function lookup
            - Handles both Series and DataFrame inputs per transformation
            - Index alignment crucial for spatial transformations
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
    Handle data loading, fetching, and preprocessing for VIEWS forecasting models.

    Manages complete data pipeline from VIEWSER fetch to model-ready DataFrames.
    Supports partition-based splitting (calibration/validation/forecasting),
    drift detection, optional VIEWSER updates, and automatic validation.

    Key Features:
        - Fetches data from VIEWSER with queryset filters
        - Partitions data by time for train/test splits
        - Validates temporal alignment and completeness
        - Applies drift detection for production runs
        - Updates VIEWSER data with latest GED/ACLED
        - Caches fetched data for reuse

    Partition Types:
        - calibration: Training period for model development
            Train: 1990-2012, Test: 2013-2015
        - validation: Holdout period for final evaluation
            Train: 1990-2015, Test: 2016-2018
        - forecasting: Production mode with live data
            Train: 1990-present, Test: future months

    Attributes:
        _model_path (ModelPathManager): Path manager for data directories
        _model_name (str): Model name for logging
        _path_raw (Path): Raw data directory
        _path_processed (Path): Processed data directory
        partition (Optional[str]): Current partition type
        partition_dict (Optional[Dict]): Partition time ranges
        drift_config_dict (Optional[Dict]): Drift detection config
        override_month (Optional[int]): Override end month
        month_first (Optional[int]): Start month ID
        month_last (Optional[int]): End month ID
        steps (int): Forecast horizon in months

    Example:
        >>> from views_pipeline_core.managers import ModelPathManager
        >>> model_path = ModelPathManager("purple_alien")
        >>> loader = ViewsDataLoader(
        ...     model_path=model_path,
        ...     steps=36
        ... )
        >>> # Fetch calibration data
        >>> df, alerts = loader.get_data(
        ...     self_test=False,
        ...     partition='calibration',
        ...     use_saved=False
        ... )
        INFO: Fetching data from viewser...
        INFO: Data validation complete.
        >>> print(df.shape)
        (180000, 45)

    Note:
        - Queryset must be defined in model configs
        - Raw data cached in data/raw/
        - Drift detection only on forecasting runs
        - VIEWSER updates require .env configuration
    """

    def __init__(self, model_path: ModelPathManager, partition_dict: Dict = None, steps: int = 36, **kwargs):
        """
        Initialize ViewsDataLoader with model paths and configuration.

        Sets up data loading infrastructure including paths, partition settings,
        and optional configurations from kwargs.

        Args:
            model_path: ModelPathManager instance for the model.
                Must have valid data_raw and data_processed directories
            partition_dict: Custom partition configuration.
                If None, uses default partitions from _get_partition_dict().
                Format: {'train': (start, end), 'test': (start, end)}
            steps: Forecast horizon in months. Default: 36
                Used for forecasting partition end date calculation
            **kwargs: Additional configuration options:
                - partition (str): Set initial partition
                - drift_config_dict (Dict): Custom drift detection config
                - override_month (int): Override forecasting end month
                - month_first (int): Override start month
                - month_last (int): Override end month

        Example:
            >>> model_path = ModelPathManager("purple_alien")
            >>> # Basic initialization
            >>> loader = ViewsDataLoader(model_path, steps=36)
            >>>
            >>> # With custom partition
            >>> custom_part = {
            ...     'train': (121, 400),
            ...     'test': (401, 450)
            ... }
            >>> loader = ViewsDataLoader(
            ...     model_path,
            ...     partition_dict={'calibration': custom_part},
            ...     steps=48
            ... )

        Note:
            - Partition dict can be provided later via get_data()
            - Steps determines forecasting test range
            - Override options mainly for debugging/testing
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
        Generate default partition dictionary for data splitting.

        Creates standard time ranges for calibration, validation, and
        forecasting partitions. Uses fixed historical ranges for
        calibration/validation and dynamic ranges for forecasting.

        Internal Use:
            Called by get_data() when partition_dict not provided.

        Args:
            steps: Forecast horizon in months for forecasting partition.
                Determines how far into future the test range extends.

        Returns:
            Dictionary with train/test ranges for requested partition:
                {
                    'train': (start_month_id, end_month_id),
                    'test': (start_month_id, end_month_id)
                }

        Partition Definitions:
            calibration:
                Train: 121-396 (1990-01 to 2012-12)
                Test: 397-444 (2013-01 to 2015-12)

            validation:
                Train: 121-444 (1990-01 to 2015-12)
                Test: 445-492 (2016-01 to 2018-12)

            forecasting:
                Train: 121 to (current_month - 2)
                Test: (current_month - 1) to (current_month + steps)

        Example:
            >>> # Current month is 530 (2024-02)
            >>> part_dict = loader._get_partition_dict(steps=36)
            >>> print(part_dict)
            {
                'train': (121, 528),
                'test': (529, 565)
            }

        Raises:
            ValueError: If partition not in ('calibration', 'validation', 'forecasting')

        Note:
            - Month IDs are months since 1980-01
            - Forecasting uses ViewsMonth.now() for current time
            - Warns about using default instead of config file
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
                    ViewsMonth.now().id - 1
                )  # minus 1 because the current month is not yet available. Verified but can be tested by changing this and running the check_data notebook.
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
        Extract VIEWSER update configuration from environment.

        Loads .env file and retrieves months to update and update file path
        based on queryset's level of analysis (LOA).

        Internal Use:
            Called by _overwrite_viewser() to get update parameters.

        Args:
            queryset_base: Queryset with LOA specification.
                LOA must be 'priogrid_month' or 'country_month'

        Returns:
            Tuple of (months_to_update, update_file_path):
                - months_to_update: List of month IDs to update (e.g., [528, 529])
                - update_file_path: Path to update data file or None if LOA unknown

        Environment Variables Required:
            - month_to_update: List of month IDs as string (e.g., "[528, 529, 530]")
            - pgm_path: Path to priogrid update file (if LOA is priogrid_month)
            - cm_path: Path to country update file (if LOA is country_month)

        Example:
            >>> # .env file contains:
            >>> # month_to_update=[528, 529, 530]
            >>> # pgm_path=/data/updates/pgm_latest.parquet
            >>> months, path = loader._get_viewser_update_config(queryset)
            >>> print(months)
            [528, 529, 530]
            >>> print(path)
            '/data/updates/pgm_latest.parquet'

        Raises:
            FileNotFoundError: If .env file not found in project root
            RuntimeError: If .env file cannot be loaded
            ValueError: If month_to_update not found or invalid in .env

        Note:
            - Searches for .env in project root (using find_project_root)
            - Uses ast.literal_eval for safe parsing of month list
            - Returns None for update_path if LOA is unknown
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
        if not months_to_update_str or months_to_update_str == "":
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
        Update VIEWSER DataFrame with latest GED and ACLED values.

        Applies external updates to raw variables and recomputes all
        transformations if update_viewser flag is set in arguments.

        Internal Use:
            Called by _fetch_data_from_viewser() after initial data fetch.

        Args:
            df: VIEWSER DataFrame to potentially update.
                Must have MultiIndex (month_id, entity_id)
            queryset_base: Model queryset defining transformations.
                Used to determine which variables to update
            args: Command line arguments with update_viewser flag.
                If False, returns df unchanged

        Returns:
            Updated DataFrame with:
                - Raw variables updated for specified months
                - All transformations recomputed
                - NaN values handled according to queryset
                Or original df if args.update_viewser=False

        Example:
            >>> args = parse_args()  # update_viewser=True
            >>> df_updated = loader._overwrite_viewser(df, queryset, args)
            INFO: Overwriting Viewser dataframe with new values...
            INFO: Viewser dataframe updated
            DEBUG: NaNs in df after transformations: 0
            >>> print(df_updated.equals(df))
            False  # df was updated

        Note:
            - Requires months_to_update and update path in .env
            - Logs NaN count after transformations for debugging
            - Updates applied in-place to df
            - Original df returned if updates disabled
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
        Fetch data from VIEWSER with queryset filters and drift detection.

        Downloads or loads data using model's queryset, applies transformations,
        optionally performs drift detection, and updates with latest GED/ACLED.

        Internal Use:
            Core data fetching method called by get_data().

        Args:
            self_test: Whether to perform drift detection self-testing.
                If True, runs drift checks against historical data

        Returns:
            Tuple of (dataframe, alerts):
                - dataframe: Fetched and processed DataFrame
                - alerts: List of drift detection alerts (if any)

        Pipeline Steps:
            1. Load queryset from model configs
            2. Fetch data via queryset.publish().fetch_with_drift_detection()
            3. Log any drift detection alerts
            4. On KeyError: Retry without drift detection
            5. Apply VIEWSER updates if enabled
            6. Convert to float64 for numerical stability

        Example:
            >>> df, alerts = loader._fetch_data_from_viewser(self_test=False)
            INFO: Beginning file download through viewser...
            INFO: Found queryset for purple_alien
            >>> print(f"Fetched {len(df)} rows")
            Fetched 180000 rows
            >>> if alerts:
            ...     print(f"Drift alerts: {len(alerts)}")

        Raises:
            RuntimeError: If queryset not found or fetch fails
            Exception: If data fetching fails (logged and re-raised)

        Note:
            - Uses month_first, month_last from instance
            - Drift detection config from self.drift_config_dict
            - Updates applied based on args.update_viewser flag
            - Alerts logged as warnings if drift detected
        """
        logger.info(
            f"Beginning file download through viewser with month range {self.month_first},{self.month_last}"
        )

        queryset_base = self._model_path.get_queryset()  # just used here..

        if queryset_base is None:
            raise RuntimeError(f"Could not find queryset for {self._model_name}")
        else:
            logger.info(f"Found queryset for {self._model_name}")

        # args = parse_args()
        df, alerts = None, None

        try:
            df, alerts = queryset_base.publish().fetch_with_drift_detection(
                start_date=self.month_first,
                end_date=self.month_last,
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
                end_date=self.month_last,
            )


        except Exception as e:
            logger.error(f"Error fetching data from viewser: {e}", exc_info=True)
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Error fetching data from viewser: {e}") from e
        
        # df = self._overwrite_viewser(df, queryset_base, args)
        df = ensure_float64(df)
        return df, alerts

    def _get_month_range(self) -> tuple[int, int]:
        """
        Determine month range based on partition type.

        Calculates start and end month IDs from partition configuration,
        with optional override for forecasting runs.

        Internal Use:
            Called by get_data() to set month_first and month_last.

        Returns:
            Tuple of (month_first, month_last):
                - month_first: Start month ID from partition train range
                - month_last: End month ID based on partition type:
                    - calibration/validation: End of test range
                    - forecasting: End of train range (or override)

        Example:
            >>> loader.partition = 'calibration'
            >>> loader.partition_dict = {
            ...     'train': (121, 396),
            ...     'test': (397, 444)
            ... }
            >>> first, last = loader._get_month_range()
            >>> print(first, last)
            121 444

            >>> # Forecasting with override
            >>> loader.partition = 'forecasting'
            >>> loader.override_month = 530
            >>> first, last = loader._get_month_range()
            WARNING: Overriding end month in forecasting partition to 530
            >>> print(first, last)
            121 530

        Raises:
            ValueError: If partition not in ('calibration', 'validation', 'forecasting')

        Note:
            - Forecasting range includes only training data (test is future)
            - Override only applies to forecasting partition
            - Logs warning when override is used
        """
        month_first = self.partition_dict["train"][0]

        if self.partition == "forecasting":
            month_last = self.partition_dict["train"][1]
        elif self.partition in ["calibration", "validation"]:
            month_last = self.partition_dict["test"][1]
        else:
            raise ValueError(
                'partition should be either "calibration", "validation" or "forecasting"'
            )
        if self.partition == "forecasting" and self.override_month is not None:
            month_last = self.override_month
            logger.warning(
                f"Overriding end month in forecasting partition to {month_last}\n"
            )

        return month_first, month_last

    def _validate_df_partition(
        self, df: pd.DataFrame
    ) -> bool:
        """
        Validate DataFrame temporal alignment with partition.

        Checks that DataFrame's month range exactly matches the expected
        range from partition configuration, ensuring data completeness.

        Internal Use:
            Called by get_data() when validate=True.

        Args:
            df: DataFrame to validate.
                Must have 'month_id' in index or columns

        Returns:
            True if month range matches partition, False otherwise

        Validation Logic:
            For calibration/validation:
                - first_expected = partition['train'][0]
                - last_expected = partition['test'][1]

            For forecasting:
                - first_expected = partition['train'][0]
                - last_expected = partition['train'][1] or override_month

        Example:
            >>> loader.partition = 'calibration'
            >>> loader.partition_dict = {
            ...     'train': (121, 396),
            ...     'test': (397, 444)
            ... }
            >>> # Valid DataFrame
            >>> is_valid = loader._validate_df_partition(df)
            >>> print(is_valid)
            True
            >>>
            >>> # Invalid DataFrame (wrong range)
            >>> is_valid = loader._validate_df_partition(df_wrong)
            ERROR: Dataframe time units do not match partition time units...
            >>> print(is_valid)
            False

        Note:
            - Checks min and max month_id in DataFrame
            - Logs detailed error if validation fails
            - Override_month respected for forecasting
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
                last_month = self.override_month
        if [np.min(df_time_units), np.max(df_time_units)] != [first_month, last_month]:
            logger.error(f"Dataframe time units do not match partition time units. Got {np.min(df_time_units)}, {np.max(df_time_units)} but expected {first_month}, {last_month}.")
            return False
        else:
            return True

    # @staticmethod
    # def filter_dataframe_by_month_range(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Filters the DataFrame to include only the specified month range.

    #     Args:
    #         df (pd.DataFrame): The input DataFrame to be filtered.
    #         month_first (int): The first month ID to include.
    #         month_last (int): The last month ID to include.

    #     Returns:
    #         pd.DataFrame: The filtered DataFrame.
    #     """
    #     month_range = np.arange(self.month_first, self.month_last)
    #     return df[df["month_id"].isin(month_range)].copy()

    def get_data(
        self,
        self_test: bool,
        partition: str,
        use_saved: bool,
        validate: bool = True,
        override_month: int = None,
    ) -> tuple[pd.DataFrame, list]:
        """
        Fetch or load model data for specified partition.

        Main data loading interface. Handles complete workflow from VIEWSER
        fetch to validated, partition-aligned DataFrame ready for modeling.

        Args:
            self_test: Whether to run drift detection self-tests.
                Recommended False for normal use, True for validation
            partition: Data partition type:
                - 'calibration': Development data (1990-2015)
                - 'validation': Holdout data (2016-2018)
                - 'forecasting': Production data (1990-present)
            use_saved: Whether to use cached data if available.
                True: Load from disk if exists, fetch if missing
                False: Always fetch fresh data from VIEWSER
            validate: Whether to validate temporal alignment.
                Recommended True to catch data issues. Default: True
            override_month: Override end month for forecasting.
                Mainly for debugging/testing. Default: None

        Returns:
            Tuple of (dataframe, alerts):
                - dataframe: Model-ready DataFrame with:
                    - MultiIndex (month_id, entity_id)
                    - Feature columns from queryset
                    - Target columns from queryset
                    - Validated time range
                - alerts: List of drift detection alerts (empty if none)

        Data Flow:
            If use_saved=True and file exists:
                1. Load cached data from data/raw/
                2. Validate partition alignment
                3. Return cached data

            If use_saved=False or file missing:
                1. Fetch from VIEWSER via queryset
                2. Apply drift detection
                3. Update with latest GED/ACLED
                4. Save to data/raw/
                5. Create fetch log
                6. Validate partition alignment
                7. Return fresh data

        Example:
            >>> loader = ViewsDataLoader(model_path, steps=36)
            >>> # Fetch fresh calibration data
            >>> df, alerts = loader.get_data(
            ...     self_test=False,
            ...     partition='calibration',
            ...     use_saved=False,
            ...     validate=True
            ... )
            INFO: Fetching data from viewser...
            INFO: Saving data to data/raw/calibration_viewser_df.parquet
            >>> print(df.shape)
            (180000, 45)
            >>>
            >>> # Use cached data
            >>> df_cached, _ = loader.get_data(
            ...     self_test=False,
            ...     partition='calibration',
            ...     use_saved=True
            ... )
            INFO: Reading saved data from data/raw/calibration_viewser_df.parquet

        Raises:
            RuntimeError: If use_saved=True but file loading fails
            RuntimeError: If fetched data incompatible with partition
            ValueError: If partition type is invalid

        File Naming:
            Cached files: {partition}_viewser_df{extension}
            Examples:
            - calibration_viewser_df.parquet
            - validation_viewser_df.parquet
            - forecasting_viewser_df.parquet

        Note:
            - Always validates unless validate=False
            - Creates data fetch log for provenance
            - Drift config from drift_detection module
            - Alerts logged even if no drift detected
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
    