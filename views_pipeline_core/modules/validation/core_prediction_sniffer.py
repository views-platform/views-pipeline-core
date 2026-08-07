# LEGACY DataFrame tier — pandas by design; retires with roadmap G5–G7 (#313/#307). See C-226.
"""
CorePredictionSniffer: Audit-only validator for prediction DataFrames.
Called after model inference, before evaluation or storage.
Fail Loud and Proud. Follows the hydranet DataSniffer pattern.
"""
from __future__ import annotations
import logging
from typing import Union

import pandas as pd

from views_pipeline_core.modules.validation.core_data_sniffer import _check_multiindex

logger = logging.getLogger(__name__)


class CorePredictionSniffer:
    """
    Audits prediction DataFrame output before evaluation or storage. Read-only throughout.
    Fail Loud and Proud. Follows the hydranet DataSniffer pattern.

    PredictionFrame is self-validating at construction and requires no external sniffer.
    """

    def __init__(self, level: str) -> None:
        self._level = level

    def sniff_predictions(
        self,
        df: pd.DataFrame,
        targets: Union[str, list],
    ) -> None:
        """Audit suite for prediction DataFrame output before evaluation or storage."""
        self._check_not_empty(df)
        self._check_targets_type(targets)
        self._check_prediction_columns(df, targets)
        self._check_multiindex_structure(df)
        logger.info("CorePredictionSniffer: prediction DataFrame audited.")

    def _check_not_empty(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("CorePredictionSniffer: Prediction DataFrame is empty.")

    def _check_targets_type(self, targets: object) -> None:
        if not isinstance(targets, (str, list)):
            raise ValueError(
                f"CorePredictionSniffer: Invalid targets type: {type(targets)}. "
                f"Expected str or list."
            )

    def _check_prediction_columns(
        self, df: pd.DataFrame, targets: Union[str, list]
    ) -> None:
        required = {
            f"pred_{t}" for t in ([targets] if isinstance(targets, str) else targets)
        }
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"CorePredictionSniffer: Missing prediction columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

    def _check_multiindex_structure(self, df: pd.DataFrame) -> None:
        _check_multiindex(df, self._level, self.__class__.__name__)