"""
PredictionFrameConverter — PF-to-DataFrame I/O adapter and structural auditing
for the ADR-033 PredictionFrame path.

Converts PredictionFrame objects to the list-in-cell parquet format required for
disk persistence and downstream consumption by the ensemble manager. This format
is a permanent cross-repo contract: the ensemble reads pred_{target} columns with
list-in-cell cells from every model's saved predictions.
"""
import logging
import numpy as np
import pyarrow as pa
from typing import Any, List

import pandas as pd

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
            {f"pred_{target}": [list(row) for row in pf.y_pred]},
            index=idx,
        )

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

        Used during the ADR-033 Strangler Fig transition to verify that the
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

        n_rows, n_samples = pf.y_pred.shape

        # Build ListArray from flat numpy — zero-copy, no Python list objects
        flat_values = pa.array(pf.y_pred.reshape(-1), type=pa.float32())
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
