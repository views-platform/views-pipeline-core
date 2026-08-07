"""Validation package (ADR-041 sniffers).

Re-exports are lazy (PEP 562, shared machinery in views_pipeline_core/_lazy.py)
so the import-light frame sniffer can be imported without paying for the
pandas-based sniffers (#288).
"""
from typing import TYPE_CHECKING

from views_pipeline_core._lazy import lazy_attach

if TYPE_CHECKING:  # pragma: no cover — static-analysis convenience only
    from .core_config_sniffer import CoreConfigSniffer  # noqa: F401
    from .core_data_sniffer import CoreDataSniffer  # noqa: F401
    from .core_frame_sniffer import CoreFrameSniffer  # noqa: F401
    from .core_prediction_sniffer import CorePredictionSniffer  # noqa: F401

#: name → defining submodule. Single source of truth for __all__/__getattr__/__dir__.
_LAZY_EXPORTS = {
    "CoreConfigSniffer": "core_config_sniffer",
    "CoreDataSniffer": "core_data_sniffer",
    "CoreFrameSniffer": "core_frame_sniffer",
    "CorePredictionSniffer": "core_prediction_sniffer",
}
_LAZY_SUBMODULES = {
    "core_config_sniffer",
    "core_data_sniffer",
    "core_frame_sniffer",
    "core_prediction_sniffer",
    "ensemble",
}

__all__ = sorted(_LAZY_EXPORTS)

__getattr__, __dir__ = lazy_attach(__name__, _LAZY_EXPORTS, _LAZY_SUBMODULES)