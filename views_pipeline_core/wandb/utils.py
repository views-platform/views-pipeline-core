from typing import Dict
from statistics import mean
import re
from dataclasses import asdict
import wandb
from views_evaluation.evaluation.metrics import EvaluationMetrics


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
    dict_of_eval_dicts: Dict[str, EvaluationMetrics], 
    step: str
) -> dict:
    """
    Generate a WandB log dictionary for step-wise evaluation metrics.

    Args:
        log_dict (dict): The log dictionary to be updated with new metrics.
        dict_of_eval_dicts (Dict[str, EvaluationMetrics]): A dictionary of evaluation metrics,
            where the keys are steps and values are `EvaluationMetrics` instances.
        step (str): The specific time step (month forecasted) for which metrics are logged (e.g., 'step01').

    Returns:
        dict: The updated log dictionary with the evaluation metrics for the specified feature and step.
    """
    for key, value in asdict(dict_of_eval_dicts[step]).items():
        if value is not None:
            log_dict[f"step-wise/{key}"] = value

    return log_dict


def generate_wandb_month_wise_log_dict(
    log_dict: dict, 
    dict_of_eval_dicts: Dict[str, EvaluationMetrics], 
    month: str
) -> dict:
    """
    Generate a WandB log dictionary for month-wise evaluation metrics.

    Args:
        log_dict (dict): The log dictionary to be updated with new metrics.
        dict_of_eval_dicts (Dict[str, EvaluationMetrics]): A dictionary of evaluation metrics,
            where the keys are months and values are `EvaluationMetrics` instances.
        month (str): The specific month for which metrics are logged (e.g., 'month501').

    Returns:
        dict: The updated log dictionary with the evaluation metrics for the specified feature and month.
    """
    for key, value in asdict(dict_of_eval_dicts[month]).items():
        if value is not None:
            log_dict[f"month-wise/{key}"] = value

    return log_dict


def generate_wandb_time_series_wise_log_dict(
    log_dict: dict, 
    dict_of_eval_dicts: Dict[str, EvaluationMetrics], 
    time_series: str
) -> dict:
    """
    Generate a WandB log dictionary for time-series-wise evaluation metrics.

    Args:
        log_dict (dict): The log dictionary to be updated with new metrics.
        dict_of_eval_dicts (Dict[str, EvaluationMetrics]): A dictionary of evaluation metrics,
            where the keys are time series and values are `EvaluationMetrics` instances.
        time_series (str): The specific time series for which metrics are logged (e.g., 'ts01').

    Returns:
        dict: The updated log dictionary with the evaluation metrics for the specified feature and time series
    """
    for key, value in asdict(dict_of_eval_dicts[time_series]).items():
        if value is not None:
            log_dict[f"time-series-wise/{key}"] = value

    return log_dict


def calculate_mean_evaluation_metrics(evaluation_dict: dict) -> dict:
    """
    Calculate the mean evaluation metrics for a dictionary of evaluation metrics.

    Args:
        evaluation_dict (Dict[str, EvaluationMetrics]): A dictionary of evaluation metrics,
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
        log_dict[f"step-wise/step/{conflict_type}"] = s
        step_wise_log_dict = generate_wandb_step_wise_log_dict(
            log_dict, step_wise_evaluation, step
        )
        wandb.log(step_wise_log_dict)

    for month in month_wise_evaluation.keys():
        m = int(re.search(r"\d+", month).group())
        log_dict = {}
        log_dict[f"month-wise/month/{conflict_type}"] = m
        month_wise_log_dict = generate_wandb_month_wise_log_dict(
            log_dict, month_wise_evaluation, month
        )
        wandb.log(month_wise_log_dict)

    for time_series in time_series_wise_evaluation.keys():
        ts = int(re.search(r"\d+", time_series).group())
        log_dict = {}
        log_dict[f"time-series-wise/time-series/{conflict_type}"] = ts
        ts_wise_log_dict = generate_wandb_time_series_wise_log_dict(
            log_dict, time_series_wise_evaluation, time_series
        )
        wandb.log(ts_wise_log_dict)

    # Calculate and log the mean evaluation metrics
    mean_step_wise = calculate_mean_evaluation_metrics(step_wise_evaluation)
    mean_month_wise = calculate_mean_evaluation_metrics(month_wise_evaluation)
    mean_time_series_wise = calculate_mean_evaluation_metrics(
        time_series_wise_evaluation
    )

    for key, value in mean_step_wise.items():
        wandb.log({f"step_wise_{conflict_type}_{key.lower()}_mean": value})

    for key, value in mean_month_wise.items():
        wandb.log({f"month_wise_{conflict_type}_{key.lower()}_mean": value})

    for key, value in mean_time_series_wise.items():
        wandb.log({f"time_series_wise_{conflict_type}_{key.lower()}_mean": value})
