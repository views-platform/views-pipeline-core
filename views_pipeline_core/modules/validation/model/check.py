from typing import Union
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def validate_prediction_dataframe(dataframe: pd.DataFrame, target: Union[list, str]) -> None:
    """
    Validate prediction dataframe structure and required components.

    Checks that the prediction DataFrame contains required target columns,
    proper index structure (priogrid_id/country_id + month_id), and valid
    data for either PGM or CM models.

    Args:
        dataframe: Prediction DataFrame to validate. Must contain:
            - Index: (priogrid_id|country_id, month_id) for MultiIndex
            - Columns: (country_id, month_id, pred_*) for regular index
            - Prediction columns matching target names
        target: Target variable name(s). Either:
            - Single string: 'ged_sb'
            - List of strings: ['ged_sb', 'ged_ns']
            Prediction columns must be named 'pred_{target}'

    Raises:
        ValueError: If validation fails:
            - DataFrame is empty
            - Invalid target type (not str or list)
            - Missing prediction columns (pred_{target})
            - Missing month_id in index or columns
            - Unrecognized model structure (not PGM or CM)

    Example:
        >>> # Valid PGM prediction dataframe
        >>> df = pd.DataFrame({
        ...     'pred_ged_sb': [0.1, 0.2, 0.3]
        ... }, index=pd.MultiIndex.from_tuples([
        ...     (100, 480), (100, 481), (101, 480)
        ... ], names=['priogrid_gid', 'month_id']))
        >>> validate_prediction_dataframe(df, 'ged_sb')
        ✓ PASS    | Dataframe validation complete

        >>> # Valid CM prediction dataframe
        >>> df = pd.DataFrame({
        ...     'country_id': [1, 1, 2],
        ...     'month_id': [480, 481, 480],
        ...     'pred_ged_sb': [0.1, 0.2, 0.3]
        ... })
        >>> validate_prediction_dataframe(df, 'ged_sb')
        ✓ PASS    | Dataframe validation complete

    Note:
        - Supports both PGM (priogrid) and CM (country-month) models
        - For MultiIndex: checks index names for model type
        - For regular index: checks column names for model type
        - Prints colored validation status to console
    """

    # Table formatting helpers
    def print_status(message: str, passed: bool) -> None:
        color = "92" if passed else "91"
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\033[{color}m{status:<8} | {message}\033[0m\n")

    # Print table header
    # print("\n\033[1mVALIDATION REPORT\033[0m")
    # print("\033[94mStatus   | Check\033[0m")
    # print("---------|----------------------------------------")

    # Base validation
    if dataframe.empty:
        # print_status("DataFrame contains data", False)
        raise ValueError("Prediction DataFrame is empty")
    # print_status("DataFrame contains data", True)

    # target validation
    # target = self.config["targets"]
    if not isinstance(target, (str, list)):
        # print_status("Valid target type", False)
        raise ValueError(f"Invalid target type: {type(target)}")
    # print_status("Valid target type format", True)

    required_columns = {
        f"pred_{dv}" for dv in ([target] if isinstance(target, str) else target)
    }
    missing = [col for col in required_columns if col not in dataframe.columns]

    if missing:
        # print_status("Required prediction columns present", False)
        raise ValueError(
            f"Missing columns: {missing}. Found: {list(dataframe.columns)}"
        )
    # print_status("All required prediction columns present", True)

    # Structural validation
    model_config = {
        "pgm": {"indices": ["priogrid_id", "priogrid_gid"], "columns": []},
        "cm": {"indices": ["country_id"], "columns": ["country_id", "month_id"]},
    }
    found_model = None
    index_names = (
        dataframe.index.names if isinstance(dataframe.index, pd.MultiIndex) else []
    )

    if isinstance(dataframe.index, pd.MultiIndex):
        for model, config in model_config.items():
            if any(idx in config["indices"] for idx in index_names):
                found_model = model
                if "month_id" not in index_names:
                    # print_status(f"{model.upper()} month_id index present", False)
                    raise ValueError(
                        f"Missing month_id in index for {model.upper()}"
                    )
                # print_status(f"{model.upper()} index structure valid", True)
                break
    else:
        for model, config in model_config.items():
            if any(col in dataframe.columns for col in config["columns"]):
                found_model = model
                if "month_id" not in dataframe.columns:
                    # print_status(f"{model.upper()} month_id column present", False)
                    raise ValueError(f"Missing month_id column for {model.upper()}")
                # print_status(f"{model.upper()} column structure valid", True)
                break

    if not found_model:
        # print_status("Data structure recognized", False)
        raise ValueError(
            f"Unrecognized structure. Index: {index_names}, Columns: {list(dataframe.columns)}"
        )
    print_status("Dataframe validation complete", True)

    # print("--------------------------------------------------\n")


def validate_config(config):
    """
    Validate model configuration and normalize target format.

    Checks deployment status and ensures targets are in list format.
    Modifies config in-place to normalize target field.

    Args:
        config: Model configuration dictionary with keys:
            - 'deployment_status' (str): Model status
                'production' | 'deprecated' | 'shadow'
            - 'regression_targets' (list): Continuous target variable(s)
            - 'classification_targets' (list): Categorical target variable(s)
            - 'regression_metrics' (list): Metrics for regression
            - 'classification_metrics' (list): Metrics for classification
            - 'targets' (str | list): Legacy target field
            - 'metrics' (list): Legacy metrics field
            - 'name' (str): Model name (for error messages)

    Raises:
        ValueError: If validation fails:
            - deployment_status is 'deprecated'
            - targets are missing or invalid type
            - mixing legacy and explicit keys

    Note:
        - Modifies config dictionary in-place
        - Handles legacy 'targets' and 'metrics' with warnings
        - Normalizes all target/metric fields to lists
    """
    model_name = config.get("name", "Unknown Model")

    # Check if deployment status is deprecated. If so, raise an error.
    if config.get("deployment_status") == "deprecated":
        logger.error(
            f"Model {model_name} has been deprecated. Please use a different model."
        )
        raise ValueError(f"Model {model_name} is deprecated and cannot be used.")

    # Define the sets of keys
    legacy_keys = {"targets", "metrics"}
    new_keys = {
        "regression_targets", 
        "regression_metrics", 
        "classification_targets", 
        "classification_metrics"
    }
    all_valid_keys = legacy_keys.union(new_keys)

    # 1. TYPO / UNRECOGNIZED KEY DETECTION
    # Look for keys that look like they should be targets or metrics but aren't
    unrecognized_target_keys = [
        k for k in config.keys() 
        if ("target" in k.lower() or "metric" in k.lower()) 
        and k not in all_valid_keys
    ]
    if unrecognized_target_keys:
        warning_msg = (
            f"\n\033[91m" + "!" * 80 + "\n"
            f"# {{:^76}} #\n".format("UNRECOGNIZED CONFIGURATION KEYS") +
            "# {:<76} #\n".format("") +
            "# {:<76} #\n".format(f"  The following keys were found in model '{model_name}' but are NOT part") +
            "# {:<76} #\n".format("  of the standard pipeline contract:") +
            "# {:<76} #\n".format(f"  -> {unrecognized_target_keys}") +
            "# {:<76} #\n".format("") +
            "# {:<76} #\n".format("  Valid keys for targets and metrics are:") +
            "# {:<76} #\n".format("  - regression_targets / regression_metrics") +
            "# {:<76} #\n".format("  - classification_targets / classification_metrics") +
            "# {:<76} #\n".format("  - targets / metrics (LEGACY)") +
            "!" * 80 + "\033[0m\n"
        )
        print(warning_msg)

    # 2. TYPE VALIDATION
    for key in all_valid_keys:
        if key in config:
            val = config[key]
            if val is not None and not isinstance(val, (str, list)):
                type_name = "Target" if "target" in key else "Metric"
                error_msg = f"{type_name} defined in '{key}' must be a string or a list of strings. Got {type(val)}."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Special case for legacy parity: 'targets' must not be None
            if key == "targets" and val is None:
                logger.error("Legacy 'targets' key must not be None.")
                raise ValueError("Target must be a string or a list of strings.")

    # Find which keys are present in the config
    present_legacy = legacy_keys.intersection(config.keys())
    present_new = new_keys.intersection(config.keys())

    # 3. STRICT MUTUAL EXCLUSIVITY RULE
    if present_legacy and present_new:
        error_msg = (
            f"Configuration Conflict in '{model_name}': You are mixing legacy keys {list(present_legacy)} "
            f"with new explicit keys {list(present_new)}. This is forbidden.\n"
            "Please migrate entirely to explicit 'regression_*' and 'classification_*' keys."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 4. LEGACY MAPPING (Only if no new keys are present)
    if present_legacy:
        mapping_msg = (
            "\n\033[93m" + "#" * 80 + "\n"
            "# {:^76} #\n".format("LEGACY CONFIGURATION DETECTED") +
            "# {:<76} #\n".format("") +
            "# {:<76} #\n".format(f"  Model: {model_name}") +
            "# {:<76} #\n".format("  The 'targets' and 'metrics' keys are now DEPRECATED.") +
            "# {:<76} #\n".format("  Assumptions being made for backward compatibility:") +
            "# {:<76} #\n".format("  - Your 'targets' are treated as 'regression_targets'.") +
            "# {:<76} #\n".format("  - Your 'metrics' are treated as 'regression_metrics'.") +
            "# {:<76} #\n".format("  - Note: CLASSIFICATION requires using explicit keys.") +
            "# {:<76} #\n".format("") +
            "# {:<76} #\n".format("  Please update your config_meta.py to use the new convention.") +
            "#" * 80 + "\033[0m\n"
        )
        print(mapping_msg)

        if "targets" in config:
            config["regression_targets"] = config["targets"]
        if "metrics" in config:
            config["regression_metrics"] = config["metrics"]

    # 5. NORMALIZATION: Convert everything to lists
    for key in new_keys:
        val = config.get(key, [])
        if isinstance(val, str):
            config[key] = [val]
        elif val is None:
            config[key] = []
        else:
            config[key] = val

    # 6. SYNC BACK: Maintain unified legacy keys for backward compatibility
    all_targets = []
    for t in config.get("regression_targets", []) + config.get("classification_targets", []):
        if t not in all_targets:
            all_targets.append(t)
    
    if not all_targets:
        error_msg = (
            f"Configuration Error in '{model_name}': No targets specified.\n"
            "You must provide at least one target using one of the following conventions:\n"
            "  1. [NEW] regression_targets: list of variables\n"
            "  2. [NEW] classification_targets: list of variables\n"
            "  3. [LEGACY] targets: list of variables (treated as regression)\n"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    config["targets"] = all_targets

    all_metrics = []
    for m in config.get("regression_metrics", []) + config.get("classification_metrics", []):
        if m not in all_metrics:
            all_metrics.append(m)
    config["metrics"] = all_metrics

    