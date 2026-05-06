"""Canonical filename generation for predictions and evaluations.

Extracted from inline calls in PredictionIOManager (SRP). Each method
returns a filename string — no I/O, no side effects.
"""
from typing import Optional

from views_pipeline_core.files.utils import (
    generate_evaluation_file_name,
    generate_output_file_name,
)


class PredictionFileNamer:
    """Generates canonical filenames for prediction and evaluation outputs.

    Consolidates filename generation that was previously inline in
    PredictionIOManager.save_predictions() and save_evaluations().
    """

    def __init__(
        self, run_type: str, timestamp: str, file_extension: str = ".parquet"
    ):
        self._run_type = run_type
        self._timestamp = timestamp
        self._file_extension = file_extension

    def prediction_name(
        self,
        sequence_number: Optional[int] = None,
        target_identifier: Optional[str] = None,
    ) -> str:
        """Generate a prediction output filename.

        Args:
            sequence_number: Rolling-origin sequence index. None for forecasting.
            target_identifier: Target name for multi-target models.

        Returns:
            Filename like ``predictions_calibration_20260407_03.parquet``.
        """
        return generate_output_file_name(
            "predictions",
            self._run_type,
            self._timestamp,
            sequence_number,
            self._file_extension,
            target_identifier=target_identifier,
        )

    def evaluation_name(
        self, evaluation_type: str, target_identifier: str
    ) -> str:
        """Generate an evaluation metrics filename.

        Args:
            evaluation_type: One of ``"step"``, ``"ts"``, ``"month"``.
            target_identifier: Target name (e.g. ``"ged_sb_best"``).

        Returns:
            Filename like ``eval_calibration_ged_sb_best_step_20260407.parquet``.
        """
        return generate_evaluation_file_name(
            evaluation_type,
            target_identifier,
            self._run_type,
            self._timestamp,
            self._file_extension,
        )
