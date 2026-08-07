"""Centralized filename generation for all pipeline outputs.

Consolidates all filename generation logic that was previously scattered
across files/utils.py and modules/data/file_namer.py into one location.

Two interfaces:
  * :class:`FilenameModule` — static methods for one-off filename generation.
  * :class:`PredictionFileNamer` — instance-based namer that holds
    ``(run_type, timestamp, file_extension)`` state and produces
    prediction/evaluation filenames. Preserved for backward compat.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FilenameModule:
    """Centralized filename generation (static methods).

    All filename generation for predictions, evaluations, input data,
    and reports happens here.

    Usage:
        >>> FilenameModule.model_artifact("calibration", ".pt")
        'calibration_model_20241105.pt'
    """

    @staticmethod
    def model_artifact(run_type: str, file_extension: str = ".pt") -> str:
        """Generate model artifact filename: {run_type}_model_{timestamp}{ext}."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{run_type}_model_{timestamp}{file_extension}"

    @staticmethod
    def prediction(
        run_type: str,
        timestamp: str,
        file_extension: str = ".parquet",
        target_identifier: str = "",
        sequence_number: Optional[int] = None,
    ) -> str:
        """Generate prediction filename."""
        name = f"predictions_{run_type}_{timestamp}"
        if sequence_number is not None:
            name += f"_{sequence_number:02d}"
        if target_identifier:
            name += f"_{target_identifier}"
        return name + file_extension

    @staticmethod
    def evaluation(
        run_type: str,
        timestamp: str,
        file_extension: str = ".csv",
    ) -> str:
        """Generate evaluation filename: metrics_{run_type}_{timestamp}{ext}."""
        return f"metrics_{run_type}_{timestamp}{file_extension}"

    @staticmethod
    def evaluation_report(
        run_type: str,
        timestamp: str,
        file_extension: str = ".html",
    ) -> str:
        """Generate evaluation report filename."""
        return f"evaluation_report_{run_type}_{timestamp}{file_extension}"

    @staticmethod
    def raw_data(
        run_type: str,
        source: str,
        timestamp: str,
        file_extension: str = ".parquet",
    ) -> str:
        """Generate raw data filename: {run_type}_{source}_df_{timestamp}{ext}."""
        return f"{run_type}_{source}_df_{timestamp}{file_extension}"


class PredictionFileNamer:
    """Instance-based filename namer for predictions and evaluations.

    Holds ``(run_type, timestamp, file_extension)`` state and produces
    prediction/evaluation filenames via instance methods.

    Usage:
        >>> namer = PredictionFileNamer("calibration", "20260407", ".parquet")
        >>> namer.prediction_name(sequence_number=3)
        'predictions_calibration_20260407_03.parquet'
        >>> namer.evaluation_name("step", "ged_sb_best")
        'eval_calibration_ged_sb_best_step_20260407.parquet'
    """

    def __init__(
        self,
        run_type: str,
        timestamp: str,
        file_extension: str = ".parquet",
    ) -> None:
        self.run_type = run_type
        self.timestamp = timestamp
        self.file_extension = file_extension

    def prediction_name(
        self,
        sequence_number: Optional[int] = None,
        target_identifier: str = "",
    ) -> str:
        """Generate prediction filename.

        Format: ``predictions_{run_type}_{timestamp}[_{target}][_{seq}]{ext}``
        """
        name = f"predictions_{self.run_type}_{self.timestamp}"
        if target_identifier:
            name += f"_{target_identifier}"
        if sequence_number is not None:
            name += f"_{sequence_number:02d}"
        return name + self.file_extension

    def evaluation_name(
        self,
        eval_type: str,
        target: str,
    ) -> str:
        """Generate evaluation filename.

        Format: ``eval_{run_type}_{target}_{eval_type}_{timestamp}{ext}``
        """
        return f"eval_{self.run_type}_{target}_{eval_type}_{self.timestamp}{self.file_extension}"


# Backward-compat module-level functions (r2darts2 imports generate_model_file_name)
def generate_model_file_name(run_type: str, file_extension: str) -> str:
    """Deprecated. Use FilenameModule.model_artifact()."""
    return FilenameModule.model_artifact(run_type, file_extension)


def generate_output_file_name(
    run_type: str,
    timestamp: str,
    file_extension: str = ".parquet",
    target_identifier: str = "",
) -> str:
    """Deprecated. Use FilenameModule.prediction()."""
    return FilenameModule.prediction(run_type, timestamp, file_extension, target_identifier)


def generate_evaluation_file_name(
    run_type: str,
    timestamp: str,
    file_extension: str = ".csv",
) -> str:
    """Deprecated. Use FilenameModule.evaluation()."""
    return FilenameModule.evaluation(run_type, timestamp, file_extension)


def generate_evaluation_report_name(
    run_type: str,
    timestamp: str,
    file_extension: str = ".html",
) -> str:
    """Deprecated. Use FilenameModule.evaluation_report()."""
    return FilenameModule.evaluation_report(run_type, timestamp, file_extension)