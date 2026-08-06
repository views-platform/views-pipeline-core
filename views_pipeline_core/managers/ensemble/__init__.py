"""Lazy ensemble-manager facade (PEP 562, #320 / C-223).

This package used to eagerly import every ensemble manager at init,
which welded the whole family's transitive closures together: importing
ANY ensemble manager (Python initializes the parent package first)
loaded the DataFrame-era ensemble modules — and with them pandas — into
every engine process. The frame-native path (epic #300) requires the
base manager to import pandas-free, so the facade is lazy: each name
resolves on first access via `_lazy.lazy_attach` (the #288 machinery).
Every existing `from views_pipeline_core.managers.ensemble import X`
call site behaves identically, just without the eager fan-out.

Guard: tests/test_import_purity.py (C-225).
"""
from typing import TYPE_CHECKING

from views_pipeline_core._lazy import lazy_attach

if TYPE_CHECKING:  # pragma: no cover — static-analysis convenience only
    from .dataframe_ensemble import DataFrameEnsembleManager  # noqa: F401
    from .ensemble import EnsembleManager, EnsemblePathManager  # noqa: F401
    from .prediction_frame_ensemble import PredictionFrameEnsembleManager  # noqa: F401

#: name → defining submodule. Single source of truth for __all__/__getattr__/__dir__.
_LAZY_EXPORTS = {
    "DataFrameEnsembleManager": "dataframe_ensemble",
    "EnsembleManager": "ensemble",
    "EnsemblePathManager": "ensemble",
    "PredictionFrameEnsembleManager": "prediction_frame_ensemble",
}
_LAZY_SUBMODULES = {
    "cm_forecast_loader",
    "dataframe_ensemble",
    "ensemble",
    "prediction_frame_ensemble",
    "sampled_forecast_publisher",
}
__all__ = sorted(_LAZY_EXPORTS)
__getattr__, __dir__ = lazy_attach(__name__, _LAZY_EXPORTS, _LAZY_SUBMODULES)
