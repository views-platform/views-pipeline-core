from typing import Union
import logging
from pathlib import Path
import pandas as pd
from views_pipeline_core.files.utils import read_log_file

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

    Note:
        - Modifies config dictionary in-place
        - Handles legacy 'targets' and 'metrics' with warnings
        - Normalizes all target/metric fields to lists
    """
    # Check if deployment status is deprecated. If so, raise an error.
    if config.get("deployment_status") == "deprecated":
        logger.error(
            f"Model {config.get('name')} has been deprecated. Please use a different model."
        )
        raise ValueError("Model is deprecated and cannot be used.")

    # Legacy detection and warning
    has_legacy_targets = "targets" in config
    has_legacy_metrics = "metrics" in config

    if has_legacy_targets or has_legacy_metrics:
        warning_msg = (
            "\n\033[93m" + "#" * 80 + "\n"
            "# {:^76} #\n".format("LEGACY CONFIGURATION DETECTED") +
            "# {:<76} #\n".format("") +
            "# {:<76} #\n".format("  The 'targets' and 'metrics' keys are now DEPRECATED.") +
            "# {:<76} #\n".format("  - Your 'targets' are being treated as 'regression_targets'.") +
            "# {:<76} #\n".format("  - Your 'metrics' are being treated as 'regression_metrics'.") +
            "# {:<76} #\n".format("  - CLASSIFICATION will NO LONGER WORK via these legacy keys.") +
            "# {:<76} #\n".format("") +
            "# {:<76} #\n".format("  Please update your config_meta.py to use explicit keys.") +
            "#" * 80 + "\033[0m\n"
        )
        print(warning_msg)

        if has_legacy_targets and "regression_targets" not in config:
            config["regression_targets"] = config["targets"]
        if has_legacy_metrics and "regression_metrics" not in config:
            config["regression_metrics"] = config["metrics"]

    # Normalize all target/metric keys to lists
    target_keys = ["regression_targets", "classification_targets"]
    metric_keys = ["regression_metrics", "classification_metrics"]

    for key in target_keys + metric_keys:
        val = config.get(key, [])
        if isinstance(val, str):
            config[key] = [val]
        elif isinstance(val, list):
            config[key] = val
        else:
            if "target" in key:
                logger.error("Target must be a string or a list of strings.")
                raise ValueError("Target must be a string or a list of strings.")
            else:
                logger.error("Metric must be a string or a list of strings.")
                raise ValueError("Metric must be a string or a list of strings.")

    # Combined targets for backward compatibility
    # We maintain config['targets'] as the union of all targets, preserving order
    all_targets = []
    for t in config.get("regression_targets", []) + config.get("classification_targets", []):
        if t not in all_targets:
            all_targets.append(t)
    config["targets"] = all_targets

    