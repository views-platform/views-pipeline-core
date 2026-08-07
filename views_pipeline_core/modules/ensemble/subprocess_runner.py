"""Ensemble subprocess delegation — shared helper.

Extracted from ``DataFrameEnsembleManager._execute_shell_script`` and
``PredictionFrameEnsembleManager._execute_shell_script`` (which were
byte-identical) per the C-2 audit decision: "Look for common logic
that could also be moved into /modules".

Both ensemble managers now delegate to :func:`execute_model_subprocess`
instead of duplicating the subprocess.run + timeout + exception-wrapping
logic.
"""
from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

from views_pipeline_core.exceptions import PipelineException

if TYPE_CHECKING:  # pragma: no cover — annotation-only
    from views_pipeline_core.cli import ForecastingModelArgs
    from views_pipeline_core.data.model_path import ModelPathManager
    from views_pipeline_core.modules.wandb import WandBModule

logger = logging.getLogger(__name__)

#: Default subprocess timeout (2 hours). Matches the historical value
#: baked into both ensemble managers' ``_execute_shell_script`` methods.
DEFAULT_TIMEOUT_SECONDS = 7200


def execute_model_subprocess(
    model_path: "ModelPathManager",
    model_name: str,
    model_args: "ForecastingModelArgs",
    wandb_module: "WandBModule | None" = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Run a model's shell command via subprocess.

    Args:
        model_path: Path manager for the model being executed.
        model_name: Model name (for error messages).
        model_args: Argument object with ``to_shell_command(model_path)``.
        wandb_module: Optional WandB module for alerting on failure.
        timeout: Subprocess timeout in seconds (default 7200 = 2h).

    Raises:
        PipelineException: If the subprocess times out or fails.
    """
    try:
        shell_command = model_args.to_shell_command(model_path)
        logger.info(f"Executing shell command: {' '.join(shell_command)}")
        subprocess.run(shell_command, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error(
            f"Shell command timed out for model {model_name} after {timeout}s",
        )
        raise PipelineException(
            f"Shell command timed out for model {model_name} after {timeout}s. "
            "Consider increasing the timeout or investigating the model script.",
            wandb_module=wandb_module,
        )
    except Exception as e:
        logger.error(
            f"Error during shell command execution for model {model_name}: {e}",
            exc_info=True,
        )
        raise PipelineException(
            f"Error during shell command execution for model {model_name}: {e}",
            wandb_module=wandb_module,
        )