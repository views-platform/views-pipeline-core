"""Dataset↔frame adapter for injected reconciliation (#195, epic #193).

The ensemble managers hold pipeline-core datasets (`_CDataset`/`_PGDataset`, pandas
`pred_{target}` columns whose cells are per-row sample arrays). The injected
`Reconciler` port is **frames-native** (`reconcile(cm_frame, pgm_frame) -> PredictionFrame`).
This module is the only place that bridges the two: it converts the datasets to
`views_frames.PredictionFrame`s, calls the injected reconciler once per shared target,
and writes the reconciled grid values back into a fresh DataFrame matching the input
pg dataset (de-mutated, register C-182).

Single responsibility: dataset↔frame conversion + per-target orchestration. It imports
**only views-frames** for the frame types — never `views_reporting` or
`views_postprocessing` (ADP: no cross-repo cycle). Geography (the country↔grid mapping)
lives in the injected reconciler, not here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

if TYPE_CHECKING:  # avoid importing the dataset god-class at module load (SDP)
    from views_pipeline_core.data.handlers import _CDataset, _PGDataset
    from views_pipeline_core.modules.reconciliation.port import Reconciler


def _stack_cells(column: pd.Series) -> np.ndarray:
    """Stack a `pred_{target}` object column (per-row sample arrays / scalars) into
    a dense ``(N, S)`` float32 array — the inverse of the list-in-cell layout."""
    rows = [np.atleast_1d(np.asarray(v, dtype=np.float32)).ravel() for v in column.to_numpy()]
    return np.vstack(rows) if rows else np.empty((0, 1), dtype=np.float32)


def _frame_from_dataset(dataset: Any, column: str, level: SpatialLevel) -> PredictionFrame:
    """Build a `PredictionFrame` for one target column from a dataset's dataframe.

    Rows follow the dataframe's own order. The reconciled frame is realigned to the
    dataframe **by `(time, unit)` index** on write-back (see `reconcile_datasets`), so
    a reconciler that reorders rows cannot silently corrupt the result.
    """
    df = dataset.dataframe
    y_pred = _stack_cells(df[column])
    index = SpatioTemporalIndex(
        time=df.index.get_level_values(0).to_numpy(dtype=np.int64),
        unit=df.index.get_level_values(1).to_numpy(dtype=np.int64),
        level=level,
    )
    return PredictionFrame(y_pred, index)


def _align_to_dataframe(reconciled: PredictionFrame, result_df: pd.DataFrame, column: str) -> list:
    """Realign a reconciled frame's rows to `result_df`'s `(time, unit)` index.

    The injected `Reconciler` is free to return rows in any order (e.g. grouped by
    country); `PredictionFrame` preserves whatever order it is handed, so we must NOT
    assume positional correspondence. We key both sides by `(time, unit)` and reorder
    the reconciled values to the dataframe's row order — failing loud on a missing or
    duplicated key rather than silently scattering values to the wrong grid cell.
    """
    values = reconciled.values
    rec_keys = list(
        zip(
            np.asarray(reconciled.index.time).tolist(),
            np.asarray(reconciled.index.unit).tolist(),
        )
    )
    if values.shape[0] != len(result_df):
        raise ValueError(
            f"Reconciled frame for '{column}' has {values.shape[0]} rows but the pg "
            f"dataframe has {len(result_df)} — row alignment broken."
        )
    pos_by_key = {key: i for i, key in enumerate(rec_keys)}
    if len(pos_by_key) != len(rec_keys):
        raise ValueError(
            f"Reconciled frame for '{column}' has duplicate (time, unit) keys — "
            "cannot align reconciled values back to the pg dataframe."
        )
    df_keys = zip(
        result_df.index.get_level_values(0).tolist(),
        result_df.index.get_level_values(1).tolist(),
    )
    try:
        order = [pos_by_key[key] for key in df_keys]
    except KeyError as e:
        raise ValueError(
            f"Reconciled frame for '{column}' is missing grid cell {e.args[0]} present "
            "in the pg dataframe — cannot align reconciled values back (row alignment broken)."
        ) from e
    return list(values[order])


def reconcile_datasets(
    reconciler: "Reconciler",
    c_dataset: "_CDataset",
    pg_dataset: "_PGDataset",
) -> pd.DataFrame:
    """Reconcile a grid (pg) dataset to country (cm) totals via the injected port.

    For each target present in both datasets: build the cm/pgm frames, call
    ``reconciler.reconcile(cm_frame, pgm_frame)``, and write the reconciled grid
    values into a fresh copy of the pg dataframe (the pg dataset is left untouched).

    Returns a DataFrame with the pg dataset's index ``(month_id, priogrid_id)`` and
    its ``pred_{target}`` columns — the same contract the legacy
    ``ReconciliationModule.reconcile()`` returned.
    """
    targets = sorted(set(c_dataset.targets) & set(pg_dataset.targets))
    if not targets:
        raise ValueError(
            "Reconciliation found no common targets between the country and grid "
            f"datasets (cm={sorted(c_dataset.targets)}, pgm={sorted(pg_dataset.targets)})."
        )

    result_df = pg_dataset.dataframe.copy()
    for column in targets:
        cm_frame = _frame_from_dataset(c_dataset, column, SpatialLevel.CM)
        pgm_frame = _frame_from_dataset(pg_dataset, column, SpatialLevel.PGM)
        reconciled = reconciler.reconcile(cm_frame, pgm_frame)

        # Realign by (time, unit) — never trust positional row order from the port.
        # Write back as per-row sample arrays (object cells), matching the layout the
        # legacy reconcile() produced and that downstream consumers expect.
        result_df[column] = _align_to_dataframe(reconciled, result_df, column)

    return result_df