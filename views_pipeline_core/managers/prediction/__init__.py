"""Lazy prediction-package facade (PEP 562, #320 / C-223).

`io.py` (PredictionIOManager, the legacy DataFrame delivery) imports pandas at
module scope by design; the base manager imports `prediction_frame_io` from
this package, so an eager init here would put pandas back on the frame-native
import chain. Names resolve on first access via `_lazy.lazy_attach` (#288);
every existing `from views_pipeline_core.managers.prediction import X` call
site behaves identically. Guard: tests/test_import_purity.py (C-225).
"""
from views_pipeline_core._lazy import lazy_attach

_LAZY_EXPORTS = {
    "PredictionFileNamer": "file_namer",
    "PredictionIOManager": "io",
    "AppwriteSaver": "savers",
    "LocalParquetSaver": "savers",
    "NpzSaver": "savers",
    "PredictionMetadata": "savers",
    "PredictionSaver": "savers",
    "ViewsForecastsSaver": "savers",
}
_LAZY_SUBMODULES = {"file_namer", "io", "savers", "prediction_frame_io"}
__all__ = sorted(_LAZY_EXPORTS)
__getattr__, __dir__ = lazy_attach(__name__, _LAZY_EXPORTS, _LAZY_SUBMODULES)
