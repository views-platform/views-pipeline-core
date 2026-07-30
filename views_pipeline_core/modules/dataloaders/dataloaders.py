# LEGACY DataFrame tier — pandas by design; retires with roadmap G5–G7 (#313/#307). See C-226.
import os
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from views_pipeline_core.configs import drift_detection
from views_pipeline_core.files.utils import create_data_fetch_log_file
from views_pipeline_core.data.utils import ensure_float64
from views_pipeline_core.files.utils import read_dataframe, save_dataframe
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.data.constants import (
    CACHE_FILENAME_TEMPLATE,
    PARTITION_TEST,
    PARTITION_TRAIN,
)
from views_pipeline_core.data.model_path import ModelPathManager
from views_pipeline_core.modules.validation.core_data_sniffer import CoreDataSniffer
from views_pipeline_core.modules.dataloaders.fetch_context import (
    FetchContext,
    resolve_default_partition_dict,
    resolve_month_range,
)
from views_pipeline_core.modules.dataloaders.frame_cache import frame_cache_path
from views_pipeline_core.modules.dataloaders.feature_frame_path import (
    fetch_feature_frame,
)
from views_pipeline_core.modules.dataloaders.datafactory_contract import (
    DATAFACTORY_REQUIRED_KEYS,
    import_datafactory_contract,
    require_descriptor_keys,
)

import views_transformation_library.views_2 as views2
import views_transformation_library.missing as missing
from viewser import Queryset
import traceback
from dotenv import load_dotenv
import ast
import argparse

logger = logging.getLogger(__name__)

_PRIOGRID_NCOL = 720
_DATAFACTORY_REQUIRED_KEYS = DATAFACTORY_REQUIRED_KEYS  # canonical home: datafactory_contract.py
_SYNTHETIC_REQUIRED_KEYS = {"pattern", "level", "features"}
# A FLAT (already-resolved) partition dict is recognized by these keys (C-210).
_PARTITION_KEYS = {PARTITION_TRAIN, PARTITION_TEST}


def detect_data_source(queryset: Any, model_name: str) -> str:
    """Determine the data source type from a queryset/descriptor value (pure).

    The single-read contract (#289): callers read get_queryset() ONCE, detect
    the source from that snapshot, and pass the same snapshot onward — source
    detection and the fetch can never describe different querysets.

    Returns:
        'viewser', 'datafactory', or 'synthetic'.

    Raises:
        RuntimeError: If the queryset is None.
        TypeError: If the type/source is not recognized.
    """
    if queryset is None:
        raise RuntimeError(f"Could not find queryset for {model_name}")

    if isinstance(queryset, dict):
        source = queryset.get("source")
        if source == "views-datafactory":
            return "datafactory"
        if source == "synthetic":
            return "synthetic"
        raise TypeError(
            f"Dict queryset for {model_name} has unrecognized "
            f"source='{source}'. Expected 'views-datafactory' or 'synthetic'."
        )

    if hasattr(queryset, "publish"):
        return "viewser"

    raise TypeError(
        f"Unrecognized queryset type for {model_name}: "
        f"{type(queryset).__name__}. Expected viewser Queryset "
        f"(with .publish() method) or datafactory dict descriptor "
        f"(with 'source': 'views-datafactory')."
    )

#: LOA → ``load_dataset(output_format=...)``. The values are datafactory's consumer
#: contract vocabulary (their ADR-050 ``OutputFormat``): validated at fetch time via
#: ``is_valid_output_format`` (datafactory is importable there by construction) and in
#: CI against the vendored contract fixture (tests/fixtures/feature_frame_contract/,
#: tests/test_modules/test_datafactory_contract_conformance.py). Extend only with
#: strings the upstream vocabulary defines (e.g. ``feature_frame`` for #161).
_LOA_TO_OUTPUT_FORMAT = {
    "priogrid_month": "dataframe",
    "country_month": "country_month",
}

#: Grid-entity index-name consolidation (views-frames ADR-015). viewser and old
#: on-disk caches still carry the legacy ``priogrid_gid`` (datafactory retired it in
#: their #316; synthetic emits ``priogrid_id`` natively); ``_normalize_grid_index`` is
#: the single seam that rewrites it to the canonical ``priogrid_id`` so the on-disk
#: cache and every downstream consumer see one name. Remove this seam (and its call
#: sites in ``_fetch_data`` / ``get_data``) once viewser emits ``priogrid_id`` and the
#: legacy caches have aged out (#259).
_LEGACY_GRID_ID = "priogrid_gid"
_CANONICAL_GRID_ID = "priogrid_id"


def _normalize_grid_index(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Rename a legacy ``priogrid_gid`` index level to the canonical ``priogrid_id``.

    No-op when the level is absent or already canonical, or when ``df`` is not a MultiIndex
    frame. Renames the index only (no data copy); callers own the frame they pass in.
    """
    if df is None or not isinstance(df.index, pd.MultiIndex):
        return df
    names = list(df.index.names)
    if _LEGACY_GRID_ID in names and _CANONICAL_GRID_ID in names:
        # A frame carrying BOTH names is malformed; renaming would create two levels
        # named priogrid_id (duplicate). Fail loud rather than silently corrupt.
        raise ValueError(
            f"DataFrame index carries both '{_LEGACY_GRID_ID}' and '{_CANONICAL_GRID_ID}' "
            f"levels: {names}. Cannot normalize an ambiguous grid index."
        )
    if _LEGACY_GRID_ID in names:
        df.index = df.index.set_names(
            [_CANONICAL_GRID_ID if n == _LEGACY_GRID_ID else n for n in names]
        )
    return df

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
        self._cached_data_path = None
        # Set by the FeatureFrame path when it materializes its directory cache
        # (#287/#289); mirrors _cached_data_path for the pandas parquet cache.
        self._cached_frame_path = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def cached_data_path(self) -> Optional[Path]:
        return self._cached_data_path

    @property
    def cached_frame_path(self) -> Optional[Path]:
        """Path of the FeatureFrame directory cache (None until the frame path runs).

        Engines must consume this attribute rather than rebuilding the cache
        name (C-59 lesson: five hardcoded cache-name sites across three repos).
        """
        return self._cached_frame_path

    def _get_partition_dict(self, steps) -> Dict:
        """Thin backward-compat wrapper over fetch_context.resolve_default_partition_dict.

        Reads self.partition (callers/tests set it first). get_data() itself no
        longer calls this — it resolves via _resolve_fetch_context (#286). The
        partition definitions and the config_partitions.py warning live on the
        pure resolver; see fetch_context.py for the authoritative docs.
        """
        return resolve_default_partition_dict(self.partition, steps)

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

        # months_to_update = PipelineConfig.months_to_update #read from .env
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

    def _detect_data_source(self) -> str:
        """Inspect get_queryset() return to determine the data source type.

        Delegates to the pure detect_data_source() (single-read contract, #289).
        """
        return detect_data_source(self._model_path.get_queryset(), self._model_name)

    def _fetch_data_from_datafactory(
        self, self_test: bool, descriptor: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, None]:
        """Fetch data from views-datafactory using a dict descriptor.

        Counterpart to _fetch_data_from_viewser(). Lazy-imports datafactory_query,
        renames columns to VIEWSER conventions, derives row/col for priogrid models,
        fills NaN, and casts to float64. Does NOT support drift detection (C-52).

        Args:
            self_test: Whether drift detection self-testing was requested.
                Logged as a warning since datafactory has no drift detection.
            descriptor: Pre-fetched dict descriptor. If None, calls get_queryset().

        Returns:
            Tuple of (dataframe, None). Alerts are always None.

        Raises:
            RuntimeError: If descriptor is invalid, the resolved output_format is not
                in the datafactory consumer contract, or load_dataset() fails.
            ImportError: If datafactory_query is not installed or predates the
                ADR-050 contract exports (views-datafactory >= 1.8.0).
        """
        if descriptor is None:
            descriptor = self._model_path.get_queryset()

        if descriptor is None or not isinstance(descriptor, dict):
            raise RuntimeError(
                f"Expected dict descriptor for datafactory model {self._model_name}, "
                f"got {type(descriptor).__name__}"
            )

        require_descriptor_keys(descriptor, self._model_name)

        loa = descriptor["loa"]
        if loa not in _LOA_TO_OUTPUT_FORMAT:
            raise RuntimeError(
                f"Unsupported loa '{loa}' in datafactory descriptor for "
                f"{self._model_name}. Supported: {list(_LOA_TO_OUTPUT_FORMAT)}"
            )
        output_format = _LOA_TO_OUTPUT_FORMAT[loa]

        logger.info(
            f"Beginning data fetch from views-datafactory for {self._model_name} "
            f"(zarr_url={descriptor.get('zarr_url', '?')}, "
            f"region={descriptor.get('region', '?')}, "
            f"loa={loa}, output_format={output_format}, "
            f"months={self.month_first}-{self.month_last})"
        )

        contract = import_datafactory_contract(self._model_name)
        load_dataset = contract.load_dataset

        if not contract.is_valid_output_format(output_format):
            raise RuntimeError(
                f"output_format '{output_format}' (from loa='{loa}') is not in the "
                f"datafactory consumer contract (CONTRACT_VERSION={contract.CONTRACT_VERSION}). "
                f"_LOA_TO_OUTPUT_FORMAT is out of step with datafactory's OutputFormat "
                f"vocabulary — reconcile against the vendored contract fixture "
                f"(tests/fixtures/feature_frame_contract/contract.json)."
            )

        try:
            df = load_dataset(
                region=descriptor["region"],
                start=self.month_first,
                end=self.month_last,
                features=list(descriptor["features"].keys()),
                output_format=output_format,
                data_dir=descriptor["zarr_url"],
            )
        except Exception as e:
            logger.error(
                f"Error fetching data from datafactory: {e}", exc_info=True
            )
            raise RuntimeError(
                f"Error fetching data from datafactory for {self._model_name}: {e}"
            ) from e

        feature_rename = descriptor.get("features", {})
        if feature_rename:
            df = df.rename(columns=feature_rename)

        # Resolve the grid level by alias, not a hardcoded literal, so this still derives
        # row/col when datafactory emits the canonical priogrid_id (this runs before the
        # _fetch_data normalization seam). See _LEGACY_GRID_ID / _CANONICAL_GRID_ID.
        _grid_level = next(
            (n for n in df.index.names if n in (_LEGACY_GRID_ID, _CANONICAL_GRID_ID)), None
        )
        if loa == "priogrid_month" and _grid_level is not None:
            pgids = df.index.get_level_values(_grid_level)
            if "row" not in df.columns:
                df["row"] = ((pgids - 1) // _PRIOGRID_NCOL + 1).astype(float)
            if "col" not in df.columns:
                df["col"] = ((pgids - 1) % _PRIOGRID_NCOL + 1).astype(float)

        df = df.fillna(0.0)
        df = df.sort_index()
        df = ensure_float64(df)

        if self_test:
            logger.warning(
                f"Drift detection self-test requested for {self._model_name} "
                f"but is not available for views-datafactory sources. "
                f"Returning alerts=None. See risk register C-52."
            )

        logger.info(
            f"Datafactory fetch complete for {self._model_name}: "
            f"{len(df)} rows, {len(df.columns)} columns"
        )

        return df, None

    def _fetch_data_from_synthetic(
        self, self_test: bool, descriptor: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, None]:
        """Generate synthetic data from a descriptor dict.

        Args:
            self_test: Whether drift detection self-testing was requested.
                Logged as a warning since synthetic data has no drift detection.
            descriptor: Pre-fetched dict descriptor. If None, calls get_queryset().

        Returns:
            Tuple of (dataframe, None). Alerts are always None.

        Raises:
            ValueError: If descriptor is invalid.
        """
        from views_pipeline_core.modules.dataloaders.synthetic import (
            generate_synthetic_data,
            SYNTHETIC_REQUIRED_KEYS,
        )

        if descriptor is None:
            descriptor = self._model_path.get_queryset()

        if descriptor is None or not isinstance(descriptor, dict):
            raise RuntimeError(
                f"Expected dict descriptor for synthetic model {self._model_name}, "
                f"got {type(descriptor).__name__}"
            )

        missing = SYNTHETIC_REQUIRED_KEYS - descriptor.keys()
        if missing:
            raise ValueError(
                f"Synthetic descriptor for {self._model_name} is missing "
                f"required keys: {sorted(missing)}"
            )

        if self_test:
            logger.warning(
                f"Drift detection self_test requested for {self._model_name} "
                f"but is not available for synthetic data sources. "
                f"Returning alerts=None."
            )

        df = generate_synthetic_data(
            descriptor, self.month_first, self.month_last
        )

        logger.info(
            f"Synthetic fetch complete for {self._model_name}: "
            f"{len(df)} rows, {len(df.columns)} columns"
        )

        return df, None

    def _fetch_data(self, self_test: bool, source: str) -> tuple[pd.DataFrame, list | None]:
        """Dispatch to the correct fetch strategy based on detected source.

        Args:
            self_test: Whether to perform drift detection self-testing.
            source: Data source identifier ('viewser', 'datafactory', or 'synthetic')
                as returned by _detect_data_source().

        Returns:
            Tuple of (dataframe, alerts_or_None).

        Raises:
            ValueError: If source is not recognized.
        """
        if source == "viewser":
            df, alerts = self._fetch_data_from_viewser(self_test)
        elif source == "datafactory":
            df, alerts = self._fetch_data_from_datafactory(self_test)
        elif source == "synthetic":
            df, alerts = self._fetch_data_from_synthetic(self_test)
        else:
            raise ValueError(
                f"Unknown data source '{source}' for model {self._model_name}. "
                f"Expected 'viewser', 'datafactory', or 'synthetic'."
            )
        # Single normalization seam: every RAW source funnels through here, so the cache
        # written by get_data() and all downstream consumers carry the canonical
        # priogrid_id (views-frames ADR-015). See _normalize_grid_index.
        return _normalize_grid_index(df), alerts

    def _get_month_range(self) -> tuple[int, int]:
        """Thin backward-compat wrapper over fetch_context.resolve_month_range.

        Reads self.partition/self.partition_dict/self.override_month (callers/
        tests set them first). get_data() itself no longer calls this — it
        resolves via _resolve_fetch_context (#286); see fetch_context.py for the
        authoritative range semantics.
        """
        return resolve_month_range(
            self.partition, self.partition_dict, self.override_month
        )

    def _select_partition_dict(self, partition: str) -> Optional[Dict]:
        """Select this partition's bounds from the loader's stored dict.

        Handles both shapes the attribute legitimately holds: the NESTED form
        callers provide at construction ({"calibration": {train, test}, ...})
        and the FLAT form a previous get_data/get_feature_frame call stored
        back ({train, test} — the legacy attribute contract). Re-running the
        same partition reuses the flat dict; asking for a DIFFERENT partition
        against a flat dict is ambiguous and fails loud (this crash used to
        surface as a bare TypeError deep in the month-range rule — C-210).
        """
        stored = self.partition_dict
        if stored is None:
            return resolve_default_partition_dict(partition, self.steps)
        if partition in stored:
            return stored[partition]
        if _PARTITION_KEYS <= set(stored):
            if self.partition == partition:
                return stored  # same partition re-run: already resolved
            raise ValueError(
                f"ViewsDataLoader holds flat train/test bounds "
                f"(for partition '{self.partition}') but '{partition}' was "
                f"requested. Provide a per-partition partition_dict "
                f"({{'{partition}': {{'train': ..., 'test': ...}}}}) or use a "
                f"fresh loader when switching partitions."
            )
        return stored.get(partition, None)  # legacy fallthrough (unknown shape)

    def _resolve_fetch_context(
        self, partition: str, override_month: Optional[int]
    ) -> FetchContext:
        """Resolve everything a fetch needs, without mutating loader state.

        Pure with respect to the loader: reads current attributes as fallbacks
        (preserving get_data's legacy resolution rules) but writes nothing.
        get_data() assigns the result to the legacy attributes; the FeatureFrame
        path (#289) consumes the returned value directly.
        """
        partition_dict = self._select_partition_dict(partition)
        drift_config_dict = (
            drift_detection.drift_detection_partition_dict[partition]
            if self.drift_config_dict is None
            else self.drift_config_dict
        )
        if self.month_first is None or self.month_last is None:
            month_first, month_last = resolve_month_range(
                partition, partition_dict, override_month
            )
            # Operational warning lives here at the fetch layer (#288): the
            # resolver itself is pure and silent (sniffers call it too).
            if partition == "forecasting" and override_month is not None:
                logger.warning(
                    f"Overriding end month in forecasting partition to {month_last}\n"
                )
        else:
            month_first, month_last = self.month_first, self.month_last

        # Single-read contract (#289): one get_queryset() snapshot feeds both
        # source detection and (via ctx.queryset) any downstream fetch.
        queryset = self._model_path.get_queryset()
        source = detect_data_source(queryset, self._model_name)
        df_cache_filename = CACHE_FILENAME_TEMPLATE.format(
            partition=partition, source=source, ext=PipelineConfig.dataframe_format,
        )
        return FetchContext(
            partition=partition,
            partition_dict=partition_dict,
            drift_config_dict=drift_config_dict,
            override_month=override_month,
            month_first=month_first,
            month_last=month_last,
            source=source,
            df_cache_filename=df_cache_filename,
            queryset=queryset,
        )

    def _apply_context(self, ctx: FetchContext) -> None:
        """Assign the resolved context to the legacy loader attributes (#286).

        The single write-back both entry points share — extend HERE if the
        legacy attribute contract ever grows.
        """
        self.partition = ctx.partition
        self.partition_dict = ctx.partition_dict
        self.drift_config_dict = ctx.drift_config_dict
        self.override_month = ctx.override_month
        self.month_first, self.month_last = ctx.month_first, ctx.month_last


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
        level: Optional[str] = None,
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
        # Resolution is value-based (#286); _apply_context keeps the legacy
        # attribute contract for every existing reader of loader state.
        ctx = self._resolve_fetch_context(partition, override_month)
        self._apply_context(ctx)

        source = ctx.source
        path_cached_df = self._path_raw / ctx.df_cache_filename
        self._cached_data_path = path_cached_df
        alerts = None

        if use_saved:
            if path_cached_df.exists():
                try:
                    df = read_dataframe(path_cached_df)
                    # Upgrade legacy caches written before the consolidation: an on-disk
                    # priogrid_gid cache is normalized to priogrid_id on read (ADR-015).
                    df = _normalize_grid_index(df)
                    logger.info(f"Reading saved data from {path_cached_df}")
                except Exception as e:
                    raise RuntimeError(
                        f"Use of saved data was specified but getting {path_cached_df} failed with: {e}"
                    )
            else:
                logger.info(f"Saved data not found at {path_cached_df}, fetching from {source}...")
                df, alerts = self._fetch_data(self_test, source)
                data_fetch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                create_data_fetch_log_file(
                    self._path_raw, self.partition, self._model_name, data_fetch_timestamp
                )
                logger.info(f"Saving data to {path_cached_df}")
                save_dataframe(df, path_cached_df)
        else:
            logger.info(f"Fetching data from {source}...")
            df, alerts = self._fetch_data(self_test, source)
            data_fetch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            create_data_fetch_log_file(
                self._path_raw, self.partition, self._model_name, data_fetch_timestamp
            )
            logger.info(f"Saving data to {path_cached_df}")
            save_dataframe(df, path_cached_df)
            
        if validate:
            CoreDataSniffer(
                partition_dict=self.partition_dict,
                partition=self.partition,
                level=level,
                override_month=self.override_month,
            ).sniff_loaded_data(df)
            return df, alerts
        logger.debug(f"DataFrame shape: {df.shape if df is not None else 'None'}")
        for ialert, alert in enumerate(
            str(alerts).strip("[").strip("]").split("Input")
        ):
            if "offender" in alert:
                logger.warning({f"{partition} data alert {ialert}": str(alert)})

        return df, alerts

    def get_feature_frame(
        self,
        partition: str,
        use_saved: bool,
        level: str,
        validate: bool = True,
        override_month: Optional[int] = None,
    ):
        """Fetch or load a validated ``views_frames.FeatureFrame`` — no pandas.

        The FeatureFrame counterpart of :meth:`get_data` (#289, epic #285):
        datafactory-only, cache-first on ``use_saved`` (directory cache via
        ``frame_cache``, exposed as :attr:`cached_frame_path`), frame-native
        audit (``CoreFrameSniffer``) on every delivery, bare frame returned
        (no ``(frame, alerts)`` tuple — alerts are a viewser concept, always
        ``None`` for datafactory, C-52). ``level`` is required — no permissive
        mode. Deliberate differences from the pandas path are documented in
        ``feature_frame_path``'s module docstring (no row/col derivation, no
        silent NaN fill, no drift detection).

        End-state note (epic #285 condition 5): this method is the successor
        of the pandas input path, not a sibling — ``get_data`` deprecates at
        Epic C close-out (#267).
        """
        ctx = self._resolve_fetch_context(partition, override_month)
        self._apply_context(ctx)

        # Artifact handle set BEFORE the fetch (mirrors get_data's
        # _cached_data_path contract): on failure, callers still see where the
        # cache lives/would live.
        cache_dir = frame_cache_path(self._path_raw, ctx.partition, ctx.source)
        self._cached_frame_path = cache_dir

        def _write_fetch_log() -> None:
            # Invoked by fetch_feature_frame exactly when a fresh fetch
            # succeeded — provenance decision and fetch decision are one event.
            data_fetch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            create_data_fetch_log_file(
                self._path_raw, ctx.partition, self._model_name, data_fetch_timestamp
            )

        return fetch_feature_frame(
            ctx,
            ctx.queryset,
            level,
            cache_dir,
            use_saved=use_saved,
            validate=validate,
            model_name=self._model_name,
            on_fetch=_write_fetch_log,
        )
