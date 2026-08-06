"""
PredictionFrameConverter — PredictionFrame ↔ DataFrame/parquet I/O adapter and
structural auditing for the ADR-042 PredictionFrame path.

Forward (PF → DataFrame/Arrow): converts PredictionFrame objects to the
list-in-cell parquet format required for disk persistence and downstream
consumption by the ensemble manager. This format is a permanent cross-repo
contract: the ensemble reads pred_{target} columns with list-in-cell cells from
every model's saved predictions.

Reverse (DataFrame/parquet → PF): the boundary converter that normalises a model
that emits list-in-cell DataFrames (prediction_format="dataframe") into the
numpy-native PredictionFrame the rest of the pipeline uses, so pandas never
leaks past the boundary. ``from_parquet`` reads straight from disk through
pyarrow (column projection + batch iteration) so the object-dtype list-in-cell
frame is never materialised in memory.
"""
import json
import logging
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

# Column-name constants for the Arrow parquet format.
# Uses priogrid_id (post-rename convention, ADR-034). The _PGDataset rename
# boundary has already been crossed before predictions reach this converter.
_TIME_COL = "month_id"
_LEVEL_TO_ENTITY_COL: dict = {
    "cm":  "country_id",
    "pgm": "priogrid_id",
}

logger = logging.getLogger(__name__)


class PredictionFrameConverter:
    """
    Converts PredictionFrame objects to the list-in-cell DataFrame format
    required for disk persistence, and audits the structural integrity of those
    conversions.

    The disk format (pred_{target} columns, list-in-cell cells, MultiIndex time/unit)
    is a permanent cross-repo contract consumed by the ensemble manager. This class
    owns the PF→DataFrame I/O adapter layer.

    All public methods are stateless; the class exists for cohesion and to
    provide clean patch points in tests.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_prediction_df(
        self,
        pf: Any,     # PredictionFrame (duck-typed)
        target: str,
    ) -> pd.DataFrame:
        """
        Convert a single PredictionFrame to the list-in-cell DataFrame format
        required for disk persistence and ensemble consumption.

        Produces the permanent I/O format: MultiIndex (time, unit) with one column
        'pred_{target}' where each cell is a list of S sample floats. This format
        is a cross-repo contract — the ensemble manager reads pred_{target} columns
        from every model's saved parquet files.

        The natural unit of work: 1 PF = 1 target = 1 DataFrame.

        Args:
            pf:     PredictionFrame to convert.
            target: Target variable name (column becomes 'pred_{target}').

        Returns:
            DataFrame with MultiIndex (time, unit) and column 'pred_{target}'.
        """
        idx = pd.MultiIndex.from_arrays(
            [pf.identifiers["time"], pf.identifiers["unit"]]
        )
        return pd.DataFrame(
            {f"pred_{target}": [list(row) for row in pf.values]},
            index=idx,
        )

    def from_prediction_df(
        self,
        df: pd.DataFrame,
        target: str,
        level: Union[str, SpatialLevel],
    ) -> PredictionFrame:
        """Convert a single list-in-cell DataFrame to a PredictionFrame.

        The inverse of :meth:`to_prediction_df`: reads the MultiIndex (time, unit)
        and the ``pred_{target}`` column (cells are per-row sample lists for draws,
        or scalars for a point forecast) and stacks them into a dense ``(N, S)``
        float32 array. This is the boundary normaliser for a model that emits
        ``prediction_format="dataframe"``.

        Args:
            df:     DataFrame with a MultiIndex (time, unit) and a ``pred_{target}``
                    column.
            target: Target variable name (column read is ``pred_{target}``).
            level:  Spatial level — ``"cm"``/``"pgm"`` or a ``SpatialLevel``.

        Returns:
            A PredictionFrame with ``(N, S)`` values and a SpatioTemporalIndex.

        Raises:
            ValueError: the ``pred_{target}`` column is absent or the per-row
                sample widths are ragged.
        """
        level = self._coerce_level(level)
        pred_col = f"pred_{target}"
        if pred_col not in df.columns:
            raise ValueError(
                f"DF→PF conversion: expected column '{pred_col}' not found "
                f"(columns: {list(df.columns)})."
            )
        cells = df[pred_col].to_numpy()
        values = self._stack_cells(cells)
        index = SpatioTemporalIndex(
            time=df.index.get_level_values(0).to_numpy(dtype=np.int64),
            unit=df.index.get_level_values(1).to_numpy(dtype=np.int64),
            level=level,
        )
        return PredictionFrame(values, index)

    def from_legacy_dfs(
        self,
        dfs: List[pd.DataFrame],
        target: str,
        level: Union[str, SpatialLevel],
    ) -> List[PredictionFrame]:
        """Convert a list of list-in-cell DataFrames to PredictionFrames.

        The inverse of :meth:`to_legacy_dfs` — the boundary normaliser for the
        evaluation path where a model emits one DataFrame per rolling-origin
        sequence (``prediction_format="dataframe"``).

        Args:
            dfs:    List of list-in-cell DataFrames (one per evaluation sequence).
            target: Target variable name (``pred_{target}`` column).
            level:  Spatial level — ``"cm"``/``"pgm"`` or a ``SpatialLevel``.

        Returns:
            One PredictionFrame per input DataFrame, in order.
        """
        return [self.from_prediction_df(df, target, level) for df in dfs]

    def from_parquet(
        self,
        path: Union[str, Path],
        target: str,
        level: Union[str, SpatialLevel],
    ) -> PredictionFrame:
        """Stream a list-in-cell parquet file into a PredictionFrame via pyarrow.

        Reads only the ``(time, unit, pred_{target})`` columns and iterates row-group
        batches, filling a pre-allocated ``(N, S)`` float32 array. The object-dtype
        list-in-cell DataFrame is never built (no ``pd.read_parquet``, no per-cell
        Python lists) — the measured memory blow-up of that format is avoided on the
        way in as well as out.

        Handles both on-disk variants: the Arrow-written layout (explicit
        ``month_id``/entity/``pred_{target}`` columns) and the pandas-written layout
        (a MultiIndex persisted via ``df.to_parquet``). The ``pred_{target}`` column
        may be a ``list<float>`` (draws) or a primitive (a point forecast, ``S=1``).

        Args:
            path:   Path to the parquet file.
            target: Target variable name (``pred_{target}`` column).
            level:  Spatial level — ``"cm"``/``"pgm"`` or a ``SpatialLevel``.

        Returns:
            A PredictionFrame with ``(N, S)`` values and a SpatioTemporalIndex.

        Raises:
            ValueError: the ``pred_{target}`` column is absent or per-row sample
                widths are ragged.
        """
        level = self._coerce_level(level)
        pred_col = f"pred_{target}"
        parquet = pq.ParquetFile(str(path))
        schema = parquet.schema_arrow
        if pred_col not in schema.names:
            available = sorted(c for c in schema.names if c.startswith("pred_"))
            raise ValueError(
                f"parquet at {path} has no '{pred_col}' column; "
                f"available prediction columns: {available}."
            )
        time_col, unit_col = self._index_columns(schema, level)

        n_rows = parquet.metadata.num_rows
        time = np.empty(n_rows, dtype=np.int64)
        unit = np.empty(n_rows, dtype=np.int64)
        values: np.ndarray | None = None
        pos = 0
        for batch in parquet.iter_batches(columns=[time_col, unit_col, pred_col]):
            block = self._stack_arrow_cells(batch.column(pred_col))
            if values is None:
                values = np.empty((n_rows, block.shape[1]), dtype=np.float32)
            n = batch.num_rows
            values[pos : pos + n] = block
            time[pos : pos + n] = batch.column(time_col).to_numpy(zero_copy_only=False)
            unit[pos : pos + n] = batch.column(unit_col).to_numpy(zero_copy_only=False)
            pos += n
        if values is None:
            values = np.empty((0, 1), dtype=np.float32)

        index = SpatioTemporalIndex(time=time, unit=unit, level=level)
        return PredictionFrame(values, index)

    @staticmethod
    def _coerce_level(level: Union[str, SpatialLevel]) -> SpatialLevel:
        """Accept a ``"cm"``/``"pgm"`` string or a ``SpatialLevel`` and return the enum."""
        return level if isinstance(level, SpatialLevel) else SpatialLevel(level)

    @staticmethod
    def _stack_cells(cells: np.ndarray) -> np.ndarray:
        """Stack an object array of per-row sample cells into ``(N, S)`` float32."""
        if len(cells) == 0:
            return np.empty((0, 1), dtype=np.float32)
        rows = [np.atleast_1d(np.asarray(v, dtype=np.float32)).ravel() for v in cells]
        widths = {r.shape[0] for r in rows}
        if len(widths) != 1:
            raise ValueError(
                f"DF→PF conversion: ragged pred cells — every row must have the "
                f"same sample count, got widths {sorted(widths)}."
            )
        return np.vstack(rows)

    @staticmethod
    def _stack_arrow_cells(arr: pa.Array) -> np.ndarray:
        """Convert an Arrow ``pred`` column (list<float> or primitive) to ``(N, S)`` float32."""
        if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
            n = len(arr)
            if n == 0:
                return np.empty((0, 1), dtype=np.float32)
            lengths = pc.list_value_length(arr).to_numpy(zero_copy_only=False)
            s = int(lengths[0])
            if not np.all(lengths == s):
                raise ValueError(
                    "DF→PF conversion: ragged pred cells in parquet — every row "
                    "must have the same sample count."
                )
            flat = arr.flatten().to_numpy(zero_copy_only=False).astype(
                np.float32, copy=False
            )
            return flat.reshape(n, s)
        # A point forecast persisted as a scalar column → (N, 1).
        flat = arr.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        return flat.reshape(-1, 1)

    @staticmethod
    def _index_columns(schema: pa.Schema, level: SpatialLevel) -> tuple[str, str]:
        """Resolve the (time, unit) column names for either on-disk parquet variant."""
        names = set(schema.names)
        time_name, unit_name = level.index_names
        if time_name in names and unit_name in names:
            return time_name, unit_name
        meta = schema.metadata or {}
        if b"pandas" in meta:
            info = json.loads(meta[b"pandas"].decode())
            index_columns = [c for c in info.get("index_columns", []) if isinstance(c, str)]
            if len(index_columns) >= 2:
                return index_columns[0], index_columns[1]
        return "__index_level_0__", "__index_level_1__"

    def to_legacy_dfs(
        self,
        predictions: List[Any],  # List[PredictionFrame]
        target: str,
    ) -> List[pd.DataFrame]:
        """
        Convert List[PredictionFrame] to the list-in-cell DataFrame format
        that EvaluationAdapter.from_dataframes() expects.

        Each output DataFrame has:
        - MultiIndex (time, unit) derived from PredictionFrame.identifiers.
        - A single column 'pred_{target}' where each cell is a list of sample floats.

        PARITY-BRIDGE ONLY — remove when the DataFrame path is retired and
        from_dataframes() / from_prediction_frames() are no longer compared.

        # DoD #3 removal target: retire when from_dataframes() path is removed.

        Args:
            predictions: List of PredictionFrame objects.
            target:      Target variable name (used to construct the column name).

        Returns:
            List of DataFrames, one per input PredictionFrame.
        """
        return [self.to_prediction_df(pf, target) for pf in predictions]

    def audit_parity_ef(
        self,
        ef_pf: Any,   # EvaluationFrame (duck-typed)
        ef_leg: Any,  # EvaluationFrame
        target: str,
    ) -> None:
        """
        Compare two EvaluationFrame objects for bit-wise parity.

        Used during the ADR-042 Strangler Fig transition to verify that the
        PredictionFrame adapter path produces numerically identical output to
        the legacy DataFrame adapter path for the same underlying predictions.

        # DoD #3 removal target: retire when from_dataframes() path is removed.

        Args:
            ef_pf:   EvaluationFrame built from the PredictionFrame path.
            ef_leg:  EvaluationFrame built from the legacy DataFrame path.
            target:  Target column name (for logging only).

        Raises:
            ValueError: If any array comparison fails — message begins with
                        "Parity Failure".
        """
        logger.info("AUDITING EF PARITY for target: %s", target)

        try:
            np.testing.assert_allclose(ef_pf.y_pred, ef_leg.y_pred, rtol=1e-5, atol=1e-8)
        except AssertionError as e:
            raise ValueError(f"Parity Failure (y_pred): {e}")

        try:
            np.testing.assert_allclose(ef_pf.y_true, ef_leg.y_true, rtol=1e-5, atol=1e-8)
        except AssertionError as e:
            raise ValueError(f"Parity Failure (y_true): {e}")

        for key in ("time", "unit", "origin", "step"):
            try:
                np.testing.assert_array_equal(
                    ef_pf.identifiers[key], ef_leg.identifiers[key]
                )
            except AssertionError as e:
                raise ValueError(f"Parity Failure (identifiers['{key}']): {e}")

        logger.info("EF PARITY CONFIRMED for %s", target.upper())

    def audit_prediction_structure(
        self,
        pf: Any,           # PredictionFrame (duck-typed)
        df: pd.DataFrame,
        target: str,
    ) -> None:
        """
        Structural audit after PF→DF conversion.

        Verifies that the legacy list-in-cell DataFrame produced by
        to_legacy_dfs() has the correct row count and column name relative
        to the originating PredictionFrame.  Used in the forecasting-partition
        path where no actuals are available for a full EF-level parity check.

        Note: "prediction" in the method name refers to PredictionFrame, not
        the forecasting data partition.

        Args:
            pf:     The source PredictionFrame.
            df:     The converted DataFrame (output of to_legacy_dfs).
            target: Target variable name (used to check column 'pred_{target}').

        Raises:
            ValueError: "PF→DF conversion ..." if row count or column name mismatch.
        """
        pf_rows = len(pf.identifiers["time"])
        df_rows = len(df)
        if pf_rows != df_rows:
            raise ValueError(
                f"PF→DF conversion: PF has {pf_rows} rows but converted DF has {df_rows} rows."
            )
        if f"pred_{target}" not in df.columns:
            raise ValueError(
                f"PF→DF conversion: expected column 'pred_{target}' "
                f"not found in converted DF (columns: {list(df.columns)})."
            )
        logger.info("PF STRUCTURAL INTEGRITY OK for %s", target.upper())

    def to_arrow_table(
        self,
        pf: Any,      # PredictionFrame (duck-typed)
        target: str,
        level: str,
    ) -> pa.Table:
        """
        Convert a single PredictionFrame to a pa.Table without Python list
        materialisation (Fix A — zero-copy Arrow write).

        Column layout (flat, no MultiIndex):
            month_id          int64
            country_id        int64   (level='cm')
          OR
            priogrid_id       int64   (level='pgm')
            pred_{target}     List<float32>

        The written parquet is backward-compatible: pd.read_parquet() reads
        pred_{target} as object dtype (list/ndarray cells), matching the format
        produced by to_prediction_df() + df.to_parquet().

        Args:
            pf:     PredictionFrame to convert.
            target: Target variable name (column becomes 'pred_{target}').
            level:  Spatial level — 'cm' (country-month) or 'pgm' (PRIO-grid-month).

        Returns:
            pa.Table with columns [month_id, {entity_col}, pred_{target}].

        Raises:
            ValueError: If level is not 'cm' or 'pgm'.
        """
        if level not in _LEVEL_TO_ENTITY_COL:
            raise ValueError(
                f"Unsupported level '{level}'. "
                f"Expected one of {sorted(_LEVEL_TO_ENTITY_COL)}"
            )
        entity_col = _LEVEL_TO_ENTITY_COL[level]

        n_rows, n_samples = pf.values.shape

        # Build ListArray from flat numpy — zero-copy, no Python list objects
        flat_values = pa.array(pf.values.reshape(-1), type=pa.float32())
        offsets = pa.array(
            np.arange(0, (n_rows + 1) * n_samples, n_samples, dtype=np.int32)
        )
        list_array = pa.ListArray.from_arrays(offsets, flat_values)

        return pa.table(
            {
                _TIME_COL:         pf.identifiers["time"],
                entity_col:        pf.identifiers["unit"],
                f"pred_{target}":  list_array,
            }
        )
