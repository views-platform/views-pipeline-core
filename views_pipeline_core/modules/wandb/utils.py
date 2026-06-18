from typing import Union
from statistics import mean
import re
from dataclasses import asdict
import wandb
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _safe_wandb_log(data: dict) -> None:
    """Log to WandB with error suppression. Never crash the pipeline for a logging failure."""
    try:
        wandb.log(data)
    except Exception as e:
        logger.error(f"Failed to log to WandB: {e}")


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


def calculate_mean_evaluation_metrics(evaluation_dict: dict) -> dict:
    """
    Calculate the mean evaluation metrics for a dictionary of evaluation metrics.

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

    # Compute the mean for each metric, skipping metrics with None values
    for key in metric_names:
        valid_values = [
            value
            for value in (
                (item.get(key) if isinstance(item, dict) else vars(item).get(key))
                for item in evaluation_dict.values()
            )
            if value is not None
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
        if key.startswith("_"):
            key = key[1:]

        if key == "timestamp":
            continue
        elif key == "runtime":
            # convert seconds to hours, minutes, and seconds
            if isinstance(value, (int, float)):
                hours, remainder = divmod(int(value), 3600)
                formatted_dict[key] = f"{hours}h {remainder // 60}m {remainder % 60}s"
            else:
                formatted_dict[key] = value
        # elif isinstance(value, wandb.old.summary.SummarySubDict):
        #     continue
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

# Substring wandb embeds in the ValueError raised by ``Api().runs(...)`` when the
# project does not exist (the normal state for WANDB_MODE=offline models). This is
# wandb's undocumented, human-readable message text — verified against wandb 0.18.x
# (pyproject: wandb = "^0.18.7"). If a wandb bump rewords this, get_latest_run stops
# recognising project-not-found and re-raises instead of returning None — see
# risk register C-179. The WARNING log on the re-raise path makes that drift visible.
_PROJECT_NOT_FOUND_MARKER = "could not find project"


def get_latest_run(
    entity: str, model_name: str, run_type: str
) -> Optional['wandb.apis.public.runs.Run']:
    """
    Retrieve the latest finished, metrics-bearing WandB run for a model.

    Queries the WandB project ``f"{entity}/{model_name}_{run_type}"`` and returns
    the most recently created run whose state is ``"finished"`` and whose summary
    carries more than one key (i.e. it holds metrics).

    Contract (see GitHub issue #177):
        Returns the newest qualifying run, or ``None`` when the run is *genuinely
        absent* — either the project does not exist (the normal case for models
        run with ``WANDB_MODE=offline``, which never create a cloud project) or
        the project exists but holds no finished, metrics-bearing run. "No run
        available" is a normal state, not an error.

        *Transient* failures (network/communication errors, or any other
        unexpected WandB/API error) are propagated unchanged, so callers can
        distinguish "this model is not in the cloud" (``None`` -> surface as
        missing) from "WandB hiccupped" (exception -> retry or mark degraded).

    Returns:
        Optional[wandb.Run]: The latest qualifying run, or ``None`` if genuinely
        absent.

    Raises:
        Exception: Transient WandB/API errors are propagated. Project-not-found
            is NOT raised — it is reported as ``None``.
    """
    from wandb import Api

    project_path = f"{entity}/{model_name}_{run_type}"
    api = Api()
    try:
        wandb_runs = sorted(
            api.runs(project_path, include_sweeps=False),
            key=lambda run: run.created_at,
            reverse=True,
        )
    except ValueError as e:
        # WandB raises ValueError("Could not find project ...") when the project
        # does not exist — routine for offline-only models. Treat as "no run".
        # Any other ValueError is unexpected and propagates (logged so a wandb
        # message-text change that breaks the match above is observable — C-179).
        if _PROJECT_NOT_FOUND_MARKER in str(e).lower():
            logger.info(
                f"WandB project '{project_path}' not found; treating as no run "
                f"available (e.g. model run with WANDB_MODE=offline)."
            )
            return None
        logger.warning(
            f"Unexpected ValueError from WandB for '{project_path}' "
            f"(not recognised as project-not-found): {e}. Re-raising."
        )
        raise

    # Pick the latest successfully finished run that carries metrics. Returns
    # None (not StopIteration) when no run qualifies — this is a normal state.
    latest_run = next(
        (
            run
            for run in wandb_runs
            if run.state == "finished" and len(dict(run.summary)) > 1
        ),
        None,
    )
    if latest_run is None:
        logger.info(
            f"No finished, metrics-bearing WandB run found for '{project_path}'."
        )
    return latest_run