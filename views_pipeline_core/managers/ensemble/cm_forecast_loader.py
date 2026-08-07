"""Load a country (cm) model's forecast as a views_frames PredictionFrame.

Simplified: uses PredictionFrameConverter.from_prediction_df() instead
of inline numpy vstack. The combined parquet is the unified on-disk
format — both DF and PF tracks write the same file.
"""
from __future__ import annotations

from views_frames import PredictionFrame

from views_pipeline_core.files.utils import read_dataframe
from views_pipeline_core.managers.ensemble.ensemble import EnsemblePathManager

PREDICTION_COLUMN_PREFIX = "pred_"


def _prediction_column(target: str) -> str:
    """The pred_* column for a target (idempotent if already prefixed)."""
    if target.startswith(PREDICTION_COLUMN_PREFIX):
        return target
    return f"{PREDICTION_COLUMN_PREFIX}{target}"


def load_cm_frame(cm_model: str, target: str, run_type: str) -> PredictionFrame:
    """Locate cm_model's latest local forecast and return its pred_{target} as a cm frame.

    Args:
        cm_model: the country model named by reconcile_with.
        target: the prediction target (with or without pred_ prefix).
        run_type: e.g. "forecasting" — selects the generated-predictions file set.

    Raises:
        ValueError: no local forecast for cm_model, or it lacks the target column.
    """
    try:
        paths = EnsemblePathManager(cm_model).get_generated_predictions_data_file_paths(
            run_type=run_type
        )
    except OSError as e:
        raise ValueError(
            f"Cannot reconcile: could not locate a forecast for country model '{cm_model}' "
            f"(run_type={run_type}): {e}. Run the country model before reconciling."
        ) from e
    if not paths:
        raise ValueError(
            f"Cannot reconcile: no local forecast found for country model '{cm_model}' "
            f"(run_type={run_type}). Run the country model before reconciling."
        )
    df = read_dataframe(paths[0])

    column = _prediction_column(target)
    if column not in df.columns:
        available = sorted(c for c in df.columns if c.startswith(PREDICTION_COLUMN_PREFIX))
        raise ValueError(
            f"Country model '{cm_model}' forecast has no '{column}' column; "
            f"available prediction columns: {available}."
        )

    # Use the converter's from_prediction_df for the DF→PF conversion
    # instead of inline numpy vstack. Strip the pred_ prefix so the
    # converter looks for the correct column name.
    bare_target = target[len(PREDICTION_COLUMN_PREFIX):] if target.startswith(PREDICTION_COLUMN_PREFIX) else target
    from views_pipeline_core.modules.frames.prediction_frame_converter import (
        PredictionFrameConverter,
    )
    converter = PredictionFrameConverter()
    return converter.from_prediction_df(df[[column]], bare_target, "cm")
