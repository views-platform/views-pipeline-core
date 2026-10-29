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

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.files.utils import read_dataframe
from views_pipeline_core.managers.ensemble.ensemble import EnsemblePathManager
from views_pipeline_core.managers.prediction.prediction_frame_io import load_pf

logger = logging.getLogger(__name__)

#: Prefix marking a DataFrame's prediction columns (e.g. ``pred_sb``).
PREDICTION_COLUMN_PREFIX = "pred_"


def _prediction_column(target: str) -> str:
    """The `pred_*` column for a target (idempotent if already prefixed)."""
    if target.startswith(PREDICTION_COLUMN_PREFIX):
        return target
    return f"{PREDICTION_COLUMN_PREFIX}{target}"


def _target_directory(target: str) -> str:
    """The on-disk target directory for a target — `_prediction_column`'s inverse.

    PF runs name their per-target directories with the bare internal target name
    (`lr_sb_best`); `pred_` is a DataFrame *column* convention. Accepting either
    spelling is what `load_cm_frame`'s docstring has always promised.
    """
    if target.startswith(PREDICTION_COLUMN_PREFIX):
        return target[len(PREDICTION_COLUMN_PREFIX) :]
    return target


def _locate_forecast(cm_model: str, run_type: str) -> Tuple[Path, bool]:
    """`cm_model`'s newest local forecast artefact, and whether it is a PF directory.

    Both layouts are searched because the partner's declared `prediction_format` is not
    knowable from here — `load_cm_frame` is handed a *foreign* model name and has no
    context for it (ADR-069). The filesystem is the only authority on what that model
    actually wrote, and an empty census of one layout is not evidence of absence.

    The PF directory wins when both are present. That is not a preference: a PF *model*
    partner writes per-target parquets, and the DataFrame finder ignores the requested
    target, so the directory is the only selection that is correct for such a partner.

    Returns:
        `(path, is_prediction_frame)` — a run directory, or a parquet file.

    Raises:
        ValueError: neither layout matched, naming the directory searched and both
            patterns; or the location itself failed (missing model dir).
    """
    # Locating can raise before we reach the census: constructing EnsemblePathManager
    # validates the model dir (FileNotFoundError if absent), and the DataFrame scan does
    # an unguarded `data_generated.iterdir()` (FileNotFoundError if the model has never
    # produced anything). Wrap both so the failure is loud AND actionable.
    try:
        path_manager = EnsemblePathManager(cm_model)
        pf_directories = path_manager._get_generated_pf_prediction_paths(run_type=run_type)
        dataframe_paths = path_manager._get_generated_predictions_data_file_paths(
            run_type=run_type
        )
    except OSError as e:
        raise ValueError(
            f"Cannot reconcile: could not locate a forecast for country model '{cm_model}' "
            f"(run_type={run_type}): {e}. Run the country model before reconciling."
        ) from e

    if pf_directories:
        return pf_directories[0], True
    if dataframe_paths:
        return dataframe_paths[0], False

    raise ValueError(
        f"Cannot reconcile: no local forecast found for country model '{cm_model}' "
        f"(run_type={run_type}). Searched {path_manager.data_generated} for both layouts: "
        f"PredictionFrame directories 'predictions_{run_type}_*/' and DataFrame files "
        f"'predictions_{run_type}_*{PipelineConfig.dataframe_format}'. Neither matched. "
        f"Run the country model before reconciling."
    )


def load_cm_frame(cm_model: str, target: str, run_type: str) -> PredictionFrame:
    """Locate `cm_model`'s latest local forecast and return its `pred_{target}` as a cm frame.

    The partner may have written either tier. A PF run writes
    ``predictions_{run_type}_{ts}/{target}/`` directories; a DataFrame run writes
    ``predictions_{run_type}_{ts}*.parquet``. Both are searched and the PF directory is
    preferred — see `_locate_forecast` and ADR-069.

    Args:
        cm_model: the country model named by `reconcile_with`.
        target: the prediction target (with or without the `pred_` prefix).
        run_type: e.g. ``"forecasting"`` — selects the generated-predictions file set.

    Raises:
        ValueError: no local forecast for `cm_model`, or the located forecast lacks the
            target (no such target directory, or no such prediction column). All are
            fail-loud — reconciliation cannot proceed without the country totals.
    """
    path, is_prediction_frame = _locate_forecast(cm_model, run_type)

    if is_prediction_frame:
        target_directory = path / _target_directory(target)
        try:
            frame = load_pf(target_directory, SpatialLevel.CM.value)
        except FileNotFoundError as e:
            # The run exists but does not carry this target. Refuse here rather than
            # scanning older runs for one that does — substituting a stale forecast is
            # the silent-wrong-answer failure this loader exists to avoid.
            available = sorted(d.name for d in path.iterdir() if d.is_dir())
            raise ValueError(
                f"Country model '{cm_model}' forecast at {path} has no "
                f"'{_target_directory(target)}' target directory; "
                f"available targets: {available}."
            ) from e
        logger.info(
            f"Loaded cm forecast for '{cm_model}' target '{target}' from "
            f"PredictionFrame directory {target_directory}."
        )
        return frame

    logger.info(
        f"Loaded cm forecast for '{cm_model}' target '{target}' from DataFrame file {path}."
    )
    df = read_dataframe(path)

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
