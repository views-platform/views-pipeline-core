# LEGACY DataFrame tier — pandas by design; retires with roadmap G5-G7 (#313/#307). See C-226.
"""`UpdateViewser` — replay GED/ACLED updates through a queryset's transformation chain.

Split out of `dataloaders.py` by #431 (register C-164). That file held this class and
`ViewsDataLoader` — two classes sharing no state, no base class and no reason to change
together — plus this module's spatial-lag shims. At 1,746 lines it had become what the
register called a dumping ground: epic #410 touched it four times in one week, and every
change to the cache path scrolled past 560 lines of transformation-replay logic.

The move is a pure relocation. `UpdateViewser`, the four `views_transformation_library`
shims, `transformation_mapping` and `TRANSFORMATIONS_EXPECTING_DF` came across unchanged —
all four were used only by this class. `ViewsDataLoader` constructs an `UpdateViewser`, so
`dataloaders` imports this module and not the reverse.

`views_pipeline_core.modules.dataloaders.UpdateViewser` is unchanged as an import path: the
package re-exports lazily, and #431 repointed the mapping rather than the name.
"""
import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from viewser import Queryset

import views_transformation_library.missing as missing
import views_transformation_library.views_2 as views2

from views_pipeline_core.files.utils import read_dataframe

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


