"""
Centralised file naming for all pipeline artifacts.

Every generated filename in the pipeline — model files, predictions,
evaluations, evaluation reports — should be created through this module
so that naming conventions are consistent and discoverable from one place.
"""

from datetime import datetime

__all__ = [
    "ArtifactNaming",
]


class ArtifactNaming:
    """
    Centralised naming authority for all pipeline artifact files.

    All methods are static/class-level — no instance state required.
    Call these instead of constructing filenames by hand.

    Examples::

        name = ArtifactNaming.model_artifact("calibration", ".pt")
        # => "calibration_model_20260228_143022.pt"

        name = ArtifactNaming.prediction("calibration", "20260228_143022", 2, ".parquet")
        # => "predictions_calibration_20260228_143022_02.parquet"
    """

    # ------------------------------------------------------------------
    # Timestamp helpers
    # ------------------------------------------------------------------

    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

    @classmethod
    def now_timestamp(cls) -> str:
        """Return the current UTC-naive timestamp in pipeline format."""
        return datetime.now().strftime(cls.TIMESTAMP_FORMAT)

    @classmethod
    def parse_timestamp(cls, ts: str) -> datetime:
        """Parse a pipeline-format timestamp string to ``datetime``."""
        return datetime.strptime(ts, cls.TIMESTAMP_FORMAT)

    # ------------------------------------------------------------------
    # Filename generators
    # ------------------------------------------------------------------

    @staticmethod
    def model_artifact(run_type: str, file_extension: str) -> str:
        """
        Generate filename for a trained model artifact.

        Pattern: ``{run_type}_model_{timestamp}{ext}``

        Args:
            run_type: calibration | validation | forecasting
            file_extension: e.g. ".pt", ".pkl", ".cbm"

        Returns:
            Filename string (no directory component).
        """
        timestamp = datetime.now().strftime(ArtifactNaming.TIMESTAMP_FORMAT)
        return f"{run_type}_model_{timestamp}{file_extension}"

    @staticmethod
    def prediction(
        run_type: str,
        timestamp: str,
        sequence_number: int = None,
        file_extension: str = ".parquet",
    ) -> str:
        """
        Generate filename for a prediction output file.

        Pattern: ``predictions_{run_type}_{timestamp}[_{seq}]{ext}``

        Args:
            run_type: calibration | validation | forecasting
            timestamp: Pipeline-format timestamp string.
            sequence_number: Evaluation sequence index, or None for forecasts.
            file_extension: e.g. ".parquet"

        Returns:
            Filename string.
        """
        if sequence_number is not None:
            return (
                f"predictions_{run_type}_{timestamp}"
                f"_{str(sequence_number).zfill(2)}{file_extension}"
            )
        return f"predictions_{run_type}_{timestamp}{file_extension}"

    @staticmethod
    def evaluation(
        evaluation_type: str,
        target_identifier: str,
        run_type: str,
        timestamp: str,
        file_extension: str = ".parquet",
    ) -> str:
        """
        Generate filename for an evaluation metrics file.

        Pattern: ``eval_{run_type}_{target}_{eval_type}_{timestamp}{ext}``

        Args:
            evaluation_type: "step", "ts", or "month".
            target_identifier: Target name, e.g. "ged_sb".
            run_type: calibration | validation | forecasting
            timestamp: Pipeline-format timestamp string.
            file_extension: e.g. ".parquet"

        Returns:
            Filename string.
        """
        return (
            f"eval_{run_type}_{target_identifier}"
            f"_{evaluation_type}_{timestamp}{file_extension}"
        )

    @staticmethod
    def evaluation_report(
        run_type: str,
        target_identifier: str,
        timestamp: str,
        file_extension: str = ".h",
    ) -> str:
        """
        Generate filename for an evaluation report.

        Pattern: ``eval_{run_type}_{target_identifier}_{timestamp}{ext}``

        Args:
            run_type: calibration | validation | forecasting
            target_identifier: Target name, e.g. "ged_sb", "ged_ns".
            timestamp: Pipeline-format timestamp string.
            file_extension: e.g. ".json"

        Returns:
            Filename string.
        """
        return f"eval_{run_type}_{target_identifier}_{timestamp}{file_extension}"

    @staticmethod
    def raw_data(run_type: str, file_extension: str = ".parquet") -> str:
        """
        Generate filename for a raw viewser data file.

        Pattern: ``{run_type}_viewser_df{ext}``

        Args:
            run_type: calibration | validation | forecasting
            file_extension: e.g. ".parquet"

        Returns:
            Filename string.
        """
        return f"{run_type}_viewser_df{file_extension}"

    @staticmethod
    def data_fetch_log(run_type: str) -> str:
        """Filename for the data-fetch log: ``{run_type}_data_fetch_log.txt``."""
        return f"{run_type}_data_fetch_log.txt"

    @staticmethod
    def run_log(run_type: str) -> str:
        """Filename for the pipeline run log: ``{run_type}_log.txt``."""
        return f"{run_type}_log.txt"
