from typing import Union
import logging
from pathlib import Path
import pandas as pd
from views_pipeline_core.files.utils import read_log_file

logger = logging.getLogger(__name__)


def validate_prediction_dataframe(dataframe: pd.DataFrame, target: Union[list, str]) -> None:
    """Validate prediction dataframe structure and required components."""

    # Table formatting helpers
    def print_status(message: str, passed: bool) -> None:
        color = "92" if passed else "91"
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\033[{color}m{status:<8} | {message}\033[0m")

    # Print table header
    print("\n\033[1mVALIDATION REPORT\033[0m")
    print("\033[94mStatus   | Check\033[0m")
    print("---------|----------------------------------------")

    # Base validation
    if dataframe.empty:
        print_status("DataFrame contains data", False)
        raise ValueError("Prediction DataFrame is empty")
    print_status("DataFrame contains data", True)

    # target validation
    # target = self.config["targets"]
    if not isinstance(target, (str, list)):
        print_status("Valid target type", False)
        raise ValueError(f"Invalid target type: {type(target)}")
    print_status("Valid target type format", True)

    required_columns = {
        f"pred_{dv}" for dv in ([target] if isinstance(target, str) else target)
    }
    missing = [col for col in required_columns if col not in dataframe.columns]

    if missing:
        print_status("Required prediction columns present", False)
        raise ValueError(
            f"Missing columns: {missing}. Found: {list(dataframe.columns)}"
        )
    print_status("All required prediction columns present", True)

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
                    print_status(f"{model.upper()} month_id index present", False)
                    raise ValueError(
                        f"Missing month_id in index for {model.upper()}"
                    )
                print_status(f"{model.upper()} index structure valid", True)
                break
    else:
        for model, config in model_config.items():
            if any(col in dataframe.columns for col in config["columns"]):
                found_model = model
                if "month_id" not in dataframe.columns:
                    print_status(f"{model.upper()} month_id column present", False)
                    raise ValueError(f"Missing month_id column for {model.upper()}")
                print_status(f"{model.upper()} column structure valid", True)
                break

    if not found_model:
        print_status("Data structure recognized", False)
        raise ValueError(
            f"Unrecognized structure. Index: {index_names}, Columns: {list(dataframe.columns)}"
        )

    print("--------------------------------------------------\n")


def validate_config(config):
    # Check if deployment status is deprecated. If so, raise an error.
    if config["deployment_status"] == "deprecated":
        logger.error(
            f"Model {config['name']} has been deprecated. Please use a different model."
        )
        raise ValueError("Model is deprecated and cannot be used.")

    # Check if target is a list. If not, convert it to a list. Otherwise raise an error.
    if isinstance(config["targets"], str):
        config["targets"] = [config["targets"]]
    if not isinstance(config["targets"], list):
        logger.error("Target must be a string or a list of strings.")
        raise ValueError("Target must be a string or a list of strings.")