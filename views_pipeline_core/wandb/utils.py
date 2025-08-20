from typing import Union
from statistics import mean
import re
from dataclasses import asdict
import wandb
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def add_wandb_metrics():
    """
    Defines the WandB metrics for step-wise, month-wise, and time-series-wise evaluation.

    This function sets up the metrics for logging step-wise evaluation metrics in WandB.

    Usage:
        This function should be called at the start of a WandB run to configure
        how metrics are tracked over time steps.

    Example:
        >>> wandb.init(project="example_project")
        >>> add_wandb_metrics()
        >>> wandb.log({"step-wise/mean_squared_error": 0.02, "step-wise/step": 1})

    Notes:
        - The step metric "step-wise/step" will be used to log metrics for each time step.
        - Any metric prefixed with "step-wise/" will follow the "step-wise/step" step metric.

    See Also:
        - `wandb.define_metric`: WandB API for defining metrics and their step relationships

    """
    wandb.define_metric("step-wise/step")
    wandb.define_metric("step-wise/*", step_metric="step-wise/step")

    wandb.define_metric("month-wise/month")
    wandb.define_metric("month-wise/*", step_metric="month-wise/month")

    wandb.define_metric("time-series-wise/time-series")
    wandb.define_metric(
        "time-series-wise/*", step_metric="time-series-wise/time-series"
    )


def generate_wandb_step_wise_log_dict(
    log_dict: dict, 
    dict_of_eval_dicts: dict, 
    step: str,
    conflict_type: str
) -> dict:
    """
    Generate a WandB log dictionary for step-wise evaluation metrics.

    Args:
        log_dict (dict): The log dictionary to be updated with new metrics.
        dict_of_eval_dicts (dict): A dictionary of evaluation metrics,
            where the keys are steps and values are `EvaluationMetrics` instances.
        step (str): The specific time step (month forecasted) for which metrics are logged (e.g., 'step01').
        conflict_type (str): The type of conflict for which the evaluation metrics are logged.

    Returns:
        dict: The updated log dictionary with the evaluation metrics for the specified feature and step.
    """
    for key, value in asdict(dict_of_eval_dicts[step]).items():
        if value is not None:
            log_dict[f"step-wise/{key}-{conflict_type}"] = value

    return log_dict


def generate_wandb_month_wise_log_dict(
    log_dict: dict, 
    dict_of_eval_dicts: dict, 
    month: str,
    conflict_type: str
) -> dict:
    """
    Generate a WandB log dictionary for month-wise evaluation metrics.

    Args:
        log_dict (dict): The log dictionary to be updated with new metrics.
        dict_of_eval_dicts (dict): A dictionary of evaluation metrics,
            where the keys are months and values are `EvaluationMetrics` instances.
        month (str): The specific month for which metrics are logged (e.g., 'month501').
        conflict_type (str): The type of conflict for which the evaluation metrics are logged.

    Returns:
        dict: The updated log dictionary with the evaluation metrics for the specified feature and month.
    """
    for key, value in asdict(dict_of_eval_dicts[month]).items():
        if value is not None:
            log_dict[f"month-wise/{key}-{conflict_type}"] = value

    return log_dict


def generate_wandb_time_series_wise_log_dict(
    log_dict: dict, 
    dict_of_eval_dicts: dict, 
    time_series: str,
    conflict_type: str
) -> dict:
    """
    Generate a WandB log dictionary for time-series-wise evaluation metrics.

    Args:
        log_dict (dict): The log dictionary to be updated with new metrics.
        dict_of_eval_dicts (dict): A dictionary of evaluation metrics,
            where the keys are time series and values are `EvaluationMetrics` instances.
        time_series (str): The specific time series for which metrics are logged (e.g., 'ts01').
        conflict_type (str): The type of conflict for which the evaluation metrics are logged.

    Returns:
        dict: The updated log dictionary with the evaluation metrics for the specified feature and time series
    """
    for key, value in asdict(dict_of_eval_dicts[time_series]).items():
        if value is not None:
            log_dict[f"time-series-wise/{key}-{conflict_type}"] = value

    return log_dict


def calculate_mean_evaluation_metrics(evaluation_dict: dict) -> dict:
    """
    Calculate the mean evaluation metrics for a dictionary of evaluation metrics.

    Args:
        evaluation_dict (dict): A dictionary of evaluation metrics,
            where the keys are time steps, months, or time series, and values are `EvaluationMetrics` instances.

    Returns:
        dict: A dictionary of mean evaluation metrics for the input dictionary.
    """
    mean_dict = {}
    first_item = next(iter(evaluation_dict.values()))
    metric_names = vars(first_item).keys()

    # Compute the mean for each metric, skipping metrics with None values
    for key in metric_names:
        valid_values = [
            value
            for value in (vars(item).get(key) for item in evaluation_dict.values())
            if value is not None
        ]
        if valid_values:
            mean_dict[key] = mean(valid_values)

    return mean_dict


def log_wandb_log_dict(
    step_wise_evaluation: dict,
    time_series_wise_evaluation: dict,
    month_wise_evaluation: dict,
    conflict_type: str,
) -> None:
    """
    This function logs evaluation metrics to WandB for step-wise, month-wise, and time-series-wise evaluation.

    Args:
        step_wise_evaluation (dict): A dictionary of evaluation metrics for each time step.
        time_series_wise_evaluation (dict): A dictionary of evaluation metrics for each time series.
        month_wise_evaluation (dict): A dictionary of evaluation metrics for each month.
        conflict_type (str): The type of conflict for which the evaluation metrics are logged.

    Returns:
        None
    """
    for step in step_wise_evaluation.keys():
        s = int(re.search(r"\d+", step).group())
        log_dict = {}
        log_dict[f"step-wise/step"] = s
        step_wise_log_dict = generate_wandb_step_wise_log_dict(
            log_dict, step_wise_evaluation, step, conflict_type
        )
        wandb.log(step_wise_log_dict)

    for month in month_wise_evaluation.keys():
        m = int(re.search(r"\d+", month).group())
        log_dict = {}
        log_dict[f"month-wise/month"] = m
        month_wise_log_dict = generate_wandb_month_wise_log_dict(
            log_dict, month_wise_evaluation, month, conflict_type
        )
        wandb.log(month_wise_log_dict)

    for time_series in time_series_wise_evaluation.keys():
        ts = int(re.search(r"\d+", time_series).group())
        log_dict = {}
        log_dict[f"time-series-wise/time-series"] = ts
        ts_wise_log_dict = generate_wandb_time_series_wise_log_dict(
            log_dict, time_series_wise_evaluation, time_series, conflict_type
        )
        wandb.log(ts_wise_log_dict)

    # Calculate and log the mean evaluation metrics
    mean_step_wise = calculate_mean_evaluation_metrics(step_wise_evaluation)
    mean_month_wise = calculate_mean_evaluation_metrics(month_wise_evaluation)
    mean_time_series_wise = calculate_mean_evaluation_metrics(
        time_series_wise_evaluation
    )

    for key, value in mean_step_wise.items():
        wandb.log({f"step_wise_{key.lower()}_mean_{conflict_type}": value})

    for key, value in mean_month_wise.items():
        wandb.log({f"month_wise_{key.lower()}_mean_{conflict_type}": value})

    for key, value in mean_time_series_wise.items():
        wandb.log({f"time_series_wise_{key.lower()}_mean_{conflict_type}": value})
        

def wandb_alert(
    title: str,
    text: str = "",
    level: wandb.AlertLevel = wandb.AlertLevel.INFO,
    wandb_notifications: bool = True,
    models_path: Union[Path, str] = None
) -> None:
    """
    Sends an alert to Weights and Biases (WandB) if WandB notifications are enabled and a WandB run is active.

    Args:
        title (str): The title of the alert.
        text (str, optional): The text content of the alert. Defaults to an empty string.
        level (wandb.AlertLevel, optional): The level of the alert. Defaults to wandb.AlertLevel.INFO.

    Returns:
        None

    Raises:
        wandb.errors.CommError: If there is a communication error while sending the alert.
        wandb.errors.UsageError: If there is a usage error while sending the alert.
        Exception: If there is an unexpected error while sending the alert.
    """
    if wandb_notifications and wandb.run:
        try:
            # Replace the user's home directory with '[USER_HOME]' in the alert text
            text = str(text).replace(str(models_path), "[REDACTED]")
            wandb.alert(
                title=title,
                text=text,
                level=level,
            )
        except wandb.errors.CommError as e:
            logger.error(f"Communication error sending WandB alert: {e}")
        except wandb.errors.UsageError as e:
            logger.error(f"Usage error sending WandB alert: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending WandB alert: {e}")

def timestamp_to_date(timestamp):
    from datetime import datetime
    return datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')

def format_evaluation_dict(evaluation_dict):
    """
    Formats an evaluation dictionary by processing its keys and values according to specific rules.

    - Removes leading underscores from keys.
    - Skips the "timestamp" key.
    - Converts "runtime" values (in seconds) to a human-readable string format (e.g., "1h 2m 3s").
    - Skips values that are instances of `wandb.old.summary.SummarySubDict`.
    - Converts string values that represent digits to floats.
    - Preserves integer and float values as-is.
    - Sorts the resulting dictionary by key.

    Args:
        evaluation_dict (dict): The input dictionary containing evaluation metrics.

    Returns:
        dict: A formatted and sorted dictionary with processed keys and values.
    """
    formatted_dict = {}
    for key, value in evaluation_dict.items():
        orig_key = key
        if key.startswith("_"):
            key = key[1:]

        if key == "timestamp":
            # try:
            #     formatted_dict[key] = timestamp_to_date(float(value))
            # except (ValueError, TypeError):
            #     formatted_dict[key] = value
            continue
        elif key == "runtime":
            # convert seconds to hours, minutes, and seconds
            if isinstance(value, (int, float)):
                hours, remainder = divmod(int(value), 3600)
                formatted_dict[key] = f"{hours}h {remainder // 60}m {remainder % 60}s"
            else:
                formatted_dict[key] = value
        elif isinstance(value, wandb.old.summary.SummarySubDict):
            continue
        elif isinstance(value, (int, float)):
            formatted_dict[key] = value
        elif isinstance(value, str) and value.isdigit():
            formatted_dict[key] = float(value)
        else:
            formatted_dict[key] = value

    formatted_dict = dict(sorted(formatted_dict.items(), key=lambda item: item[0]))
    return formatted_dict

def format_metadata_dict(metadata_dict):
    """
    Formats a metadata dictionary by processing its keys and values.

    - Removes leading underscores from keys.
    - Converts string values that represent digits to integers.
    - Keeps integer and float values as-is.
    - Leaves other types of values unchanged.
    - Returns a new dictionary with keys sorted alphabetically.

    Args:
        metadata_dict (dict): The input dictionary containing metadata.

    Returns:
        dict: A formatted and sorted dictionary with processed keys and values.
    """
    formatted_dict = {}
    for key, value in metadata_dict.items():
        # if key == "steps" and isinstance(value, (list, tuple)):
        #     value = len(value)

        if key.startswith("_"):
            # remove the underscore prefix
            key = key[1:]
        if isinstance(value, (int, float)):
            formatted_dict[key] = value
        elif isinstance(value, str) and value.isdigit():
            formatted_dict[key] = int(value)
        else:
            formatted_dict[key] = value
    formatted_dict = dict(sorted(formatted_dict.items(), key=lambda item: item[0]))
    return formatted_dict

def get_latest_run(entity: str, model_name: str, run_type: str) -> Optional['wandb.apis.public.runs.Run']:
    """
    Retrieves the latest WandB run from the current session.

    Returns:
        Optional[wandb.Run]: The latest run object if available, otherwise None.
    """
    from wandb import Api
    api = Api()
    wandb_runs = sorted(
        api.runs(f"{entity}/{model_name}_{run_type}", include_sweeps=False),
        key=lambda run: run.created_at,
        reverse=True,
    )
    # Pick the latest successfully finished run
    latest_run = next(
        run
        for run in wandb_runs
        if run.state == "finished" and len(dict(run.summary)) > 1
    )
    return latest_run if len(dict(latest_run.summary)) > 1 else None