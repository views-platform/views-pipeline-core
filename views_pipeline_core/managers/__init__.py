"""Lazy manager-family facade (PEP 562, #320 / C-223).

This package used to eagerly import every manager at init, which welded the
whole family's transitive closures together: importing ANY manager (Python
initializes the parent package first) loaded the DataFrame-era ensemble and
prediction modules — and with them pandas — into every engine process. The
frame-native path (epic #300) requires the base manager to import pandas-free,
so the facade is lazy: each name resolves on first access via `_lazy.lazy_attach`
(the #288 machinery). Every existing `from views_pipeline_core.managers import X`
call site behaves identically, just without the eager fan-out.
Guard: tests/test_import_purity.py (C-225).
"""
from views_pipeline_core._lazy import lazy_attach

_LAZY_EXPORTS = {
    "ConfigurationManager": "configuration.configuration",
    "ExtractorManager": "extractor.extractor",
    "ExtractorPathManager": "extractor.extractor",
    "ModelManager": "model.model",
    "ModelPathManager": "model.model",
    "ForecastingModelManager": "model.model",
    "PostprocessorManager": "postprocessor.postprocessor",
    "PostprocessorPathManager": "postprocessor.postprocessor",
    "EnsembleManager": "ensemble.ensemble",
    "EnsemblePathManager": "ensemble.ensemble",
    "PackageManager": "package.package",
}
_LAZY_SUBMODULES = {
    "configuration",
    "extractor",
    "model",
    "postprocessor",
    "ensemble",
    "package",
}
__all__ = sorted(_LAZY_EXPORTS)
__getattr__, __dir__ = lazy_attach(__name__, _LAZY_EXPORTS, _LAZY_SUBMODULES)
