"""Frames-native code — PredictionFrame I/O, conversion, and helpers.

This package centralizes all views-frames-related code that was
previously scattered across managers/prediction/, managers/ensemble/,
and modules/data/.
"""
from typing import TYPE_CHECKING

from views_pipeline_core._lazy import lazy_attach

if TYPE_CHECKING:  # pragma: no cover — static-analysis convenience only
    from .prediction_frame_converter import PredictionFrameConverter  # noqa: F401
    from .prediction_frame_io import load_pf, save_pf  # noqa: F401

_LAZY_EXPORTS = {
    "PredictionFrameConverter": "prediction_frame_converter",
    "load_pf": "prediction_frame_io",
    "save_pf": "prediction_frame_io",
}
_LAZY_SUBMODULES = {
    "prediction_frame_converter",
    "prediction_frame_io",
}
__all__ = sorted(_LAZY_EXPORTS)
__getattr__, __dir__ = lazy_attach(__name__, _LAZY_EXPORTS, _LAZY_SUBMODULES)
