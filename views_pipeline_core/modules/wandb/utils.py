import math
from statistics import mean
import re
from dataclasses import asdict
import wandb
import logging

logger = logging.getLogger(__name__)


def _safe_wandb_log(data: dict) -> None:
    """Log to WandB with error suppression. Never crash the pipeline for a logging failure."""
    try:
        wandb.log(data)
    except Exception as e:
        logger.error(f"Failed to log to WandB: {e}")


def generate_wandb_step_wise_log_dict(
    log_dict: dict, 
    dict_of_eval_dicts: dict, 
    step: str,
    target_identifier: str
) -> dict:
    """
    Generate a WandB log dictionary for step-wise evaluation metrics.

    Args:
        log_dict (dict): The log dictionary to be updated with new metrics.
        dict_of_eval_dicts (dict): A dictionary of evaluation metrics,
            where the keys are steps and values are `EvaluationMetrics` instances or plain dicts.
        step (str): The specific time step (month forecasted) for which metrics are logged (e.g., 'step01').
        target_identifier (str): The target identifier for which the evaluation metrics are logged.

    Returns:
        dict: The updated log dictionary with the evaluation metrics for the specified feature and step.
    """
    entry = dict_of_eval_dicts[step]
    items = entry.items() if isinstance(entry, dict) else asdict(entry).items()
    for key, value in items:
        if value is not None:
            log_dict[f"step-wise/{target_identifier}/{key}"] = value

    return log_dict


def generate_wandb_month_wise_log_dict(
    log_dict: dict, 
    dict_of_eval_dicts: dict, 
    month: str,
    target_identifier: str
) -> dict:
    """
    Generate a WandB log dictionary for month-wise evaluation metrics.

    Args:
        log_dict (dict): The log dictionary to be updated with new metrics.
        dict_of_eval_dicts (dict): A dictionary of evaluation metrics,
            where the keys are months and values are `EvaluationMetrics` instances or plain dicts.
        month (str): The specific month for which metrics are logged (e.g., 'month501').
        target_identifier (str): The target identifier for which the evaluation metrics are logged.

    Returns:
        dict: The updated log dictionary with the evaluation metrics for the specified feature and month.
    """
    entry = dict_of_eval_dicts[month]
    items = entry.items() if isinstance(entry, dict) else asdict(entry).items()
    for key, value in items:
        if value is not None:
            log_dict[f"month-wise/{target_identifier}/{key}"] = value

    return log_dict


def generate_wandb_time_series_wise_log_dict(
    log_dict: dict, 
    dict_of_eval_dicts: dict, 
    time_series: str,
    target_identifier: str
) -> dict:
    """
    Generate a WandB log dictionary for time-series-wise evaluation metrics.

    Args:
        log_dict (dict): The log dictionary to be updated with new metrics.
        dict_of_eval_dicts (dict): A dictionary of evaluation metrics,
            where the keys are time series and values are `EvaluationMetrics` instances or plain dicts.
        time_series (str): The specific time series for which metrics are logged (e.g., 'ts01').
        target_identifier (str): The target identifier for which the evaluation metrics are logged.

    Returns:
        dict: The updated log dictionary with the evaluation metrics for the specified feature and time series
    """
    entry = dict_of_eval_dicts[time_series]
    items = entry.items() if isinstance(entry, dict) else asdict(entry).items()
    for key, value in items:
        if value is not None:
            log_dict[f"time-series-wise/{target_identifier}/{key}"] = value

    return log_dict


def _is_nan(value) -> bool:
    """True for float nan (and numpy's); False for anything that is not a real number."""
    try:
        return math.isnan(value)
    except TypeError:
        return False


def calculate_mean_evaluation_metrics(evaluation_dict: dict) -> dict:
    """
    Calculate the mean evaluation metrics for a dictionary of evaluation metrics.

    A group's value may be ``None`` (metric not computed) or ``nan`` — views-evaluation's
    documented sentinel for a metric that is undefined on that group's data: Pearson on a
    constant series, MCR with nothing observed, and since 1.1.0 AP on a group with no
    positive truth (their ADR-015 R1/R2/R9). Both are left out of the mean, as
    ``numpy.nanmean`` would leave them out and as views-evaluation's own ``mean`` row does;
    a plain ``mean`` would let one empty month turn ``AP_mean`` into ``nan`` on the
    dashboard. ``inf`` (MCR with predicted conflict where none occurred) is a number and
    is averaged as one. A metric with no finite-or-inf value in any group is omitted.

    Args:
        evaluation_dict (dict): A dictionary of evaluation metrics,
            where the keys are time steps, months, or time series, and values are `EvaluationMetrics` instances or plain dicts.

    Returns:
        dict: A dictionary of mean evaluation metrics for the input dictionary.
    """
    if not evaluation_dict:
        return {}
    mean_dict = {}
    # Collect the union of all metric keys across all items so that metrics
    # present in later items but absent from the first are not silently dropped.
    metric_names: set = set()
    for item in evaluation_dict.values():
        metric_names.update(item.keys() if isinstance(item, dict) else vars(item).keys())

    # Compute the mean for each metric, skipping None and nan (see the docstring)
    for key in metric_names:
        valid_values = [
            value
            for value in (
                (item.get(key) if isinstance(item, dict) else vars(item).get(key))
                for item in evaluation_dict.values()
            )
            if value is not None and not _is_nan(value)
        ]
        if valid_values:
            mean_dict[key] = mean(valid_values)

    return mean_dict


def log_wandb_log_dict(
    step_wise_evaluation: dict,
    time_series_wise_evaluation: dict,
    month_wise_evaluation: dict,
    target_identifier: str,
) -> None:
    """
    This function logs evaluation metrics to WandB for step-wise, month-wise, and time-series-wise evaluation.

    Args:
        step_wise_evaluation (dict): A dictionary of evaluation metrics for each time step.
        time_series_wise_evaluation (dict): A dictionary of evaluation metrics for each time series.
        month_wise_evaluation (dict): A dictionary of evaluation metrics for each month.
        target_identifier (str): The target identifier for which the evaluation metrics are logged.

    Returns:
        None
    """
    for step in step_wise_evaluation.keys():
        s = int(re.search(r"\d+", step).group())
        log_dict = {}
        log_dict["step-wise/step"] = s
        step_wise_log_dict = generate_wandb_step_wise_log_dict(
            log_dict, step_wise_evaluation, step, target_identifier
        )
        _safe_wandb_log(step_wise_log_dict)

    for month in month_wise_evaluation.keys():
        m = int(re.search(r"\d+", month).group())
        log_dict = {}
        log_dict["month-wise/month"] = m
        month_wise_log_dict = generate_wandb_month_wise_log_dict(
            log_dict, month_wise_evaluation, month, target_identifier
        )
        _safe_wandb_log(month_wise_log_dict)

    for time_series in time_series_wise_evaluation.keys():
        ts = int(re.search(r"\d+", time_series).group())
        log_dict = {}
        log_dict["time-series-wise/time-series"] = ts
        ts_wise_log_dict = generate_wandb_time_series_wise_log_dict(
            log_dict, time_series_wise_evaluation, time_series, target_identifier
        )
        _safe_wandb_log(ts_wise_log_dict)

    # Calculate and log the mean evaluation metrics
    mean_step_wise = calculate_mean_evaluation_metrics(step_wise_evaluation)
    mean_month_wise = calculate_mean_evaluation_metrics(month_wise_evaluation)
    mean_time_series_wise = calculate_mean_evaluation_metrics(
        time_series_wise_evaluation
    )

    for key, value in mean_step_wise.items():
        _safe_wandb_log({f"step-wise/{target_identifier}/{key}_mean": value})

    for key, value in mean_month_wise.items():
        _safe_wandb_log({f"month-wise/{target_identifier}/{key}_mean": value})

    for key, value in mean_time_series_wise.items():
        _safe_wandb_log({f"time-series-wise/{target_identifier}/{key}_mean": value})
