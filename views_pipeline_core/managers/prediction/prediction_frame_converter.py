"""Backward-compat re-export shim.

The ``PredictionFrameConverter`` class was relocated to
:mod:`views_pipeline_core.modules.frames.prediction_frame_converter`
(audit decision: data-related prediction helpers belong in ``modules/data/``,
not in ``managers/prediction/``).

This shim preserves the historical import path
``from views_pipeline_core.managers.prediction.prediction_frame_converter import PredictionFrameConverter``
so existing callers (including r2darts2) continue to work.
"""
from views_pipeline_core.modules.frames.prediction_frame_converter import (  # noqa: F401
    PredictionFrameConverter,
)

__all__ = ["PredictionFrameConverter"]