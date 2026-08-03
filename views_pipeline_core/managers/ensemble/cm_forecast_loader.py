"""Load a country (cm) model's forecast as a views_frames PredictionFrame (#235, epic #233).

`PredictionFrameEnsembleManager` reconciles a grid ensemble to a country model named by the
`reconcile_with` config (wiring is S3/#236). That country forecast is read here at the
**ingest edge** (pandas) and handed to the frames-native reconciler as a cm `PredictionFrame`.

A free function, not a manager method: it keeps the reconciliation ingest testable without
constructing the (large) PFE, and keeps that class from growing (epic #233 design — external
functions the manager calls).

PFE is a **local-disk** manager (it loads constituent forecasts from `data_generated`), so the
country forecast is located the same way. The DataFrame `EnsembleManager`'s prediction-store
path is intentionally out of scope here: it needs a generated pred-store name + run creation
(`_get_pred_store_name`), which PFE has no counterpart for. Today's cm models are **point**
DataFrame forecasts, so the parquet's `pred_{target}` column is stacked into a `(N, S)` array
(S=1 for a point model, S>1 for a draws-emitting one) and wrapped as a cm frame; the
reconciler broadcasts a point cm across the grid's draws natively (views-frames ≥1.8).

Missing rows are **not** filled (unlike `_CDataset` preprocessing): a country absent from the
forecast must surface as the reconciler's fail-loud "missing country", not a silent zero total.
"""
from __future__ import annotations

import numpy as np
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

from views_pipeline_core.files.utils import read_dataframe
from views_pipeline_core.managers.ensemble.ensemble import EnsemblePathManager

#: Prefix marking a DataFrame's prediction columns (e.g. ``pred_sb``).
PREDICTION_COLUMN_PREFIX = "pred_"


def _prediction_column(target: str) -> str:
    """The `pred_*` column for a target (idempotent if already prefixed)."""
    if target.startswith(PREDICTION_COLUMN_PREFIX):
        return target
    return f"{PREDICTION_COLUMN_PREFIX}{target}"


def load_cm_frame(cm_model: str, target: str, run_type: str) -> PredictionFrame:
    """Locate `cm_model`'s latest local forecast and return its `pred_{target}` as a cm frame.

    Args:
        cm_model: the country model named by `reconcile_with`.
        target: the prediction target (with or without the `pred_` prefix).
        run_type: e.g. ``"forecasting"`` — selects the generated-predictions file set.

    Raises:
        ValueError: no local forecast for `cm_model`, or it lacks the target column. Both are
            fail-loud — reconciliation cannot proceed without the country totals.
    """
    # Locating the forecast can raise before we reach the `not paths` guard: constructing
    # EnsemblePathManager validates the model dir (FileNotFoundError if absent), and the
    # path scan does an unguarded `data_generated.iterdir()` (FileNotFoundError if the model
    # has never produced a forecast). Wrap both so the failure is loud AND actionable.
    try:
        paths = EnsemblePathManager(cm_model)._get_generated_predictions_data_file_paths(
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

    # Stack the per-row prediction cells (scalars for point, sample arrays for draws) into a
    # dense (N, S) float32 array — mirrors the cell contract in reconciliation/adapter.py.
    cells = df[column].to_numpy()
    rows = [np.atleast_1d(np.asarray(v, dtype=np.float32)).ravel() for v in cells]
    y_pred = np.vstack(rows) if rows else np.empty((0, 1), dtype=np.float32)
    index = SpatioTemporalIndex(
        time=df.index.get_level_values(0).to_numpy(dtype=np.int64),
        unit=df.index.get_level_values(1).to_numpy(dtype=np.int64),
        level=SpatialLevel.CM,
    )
    return PredictionFrame(y_pred, index)
