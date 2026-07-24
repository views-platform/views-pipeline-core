"""Data layer package.

Constants are re-exported eagerly (stdlib-pure). Everything else is lazy
(PEP 562, shared machinery in views_pipeline_core/_lazy.py) so import-light
modules can depend on ``data.constants``/``data.partitions`` without paying for
pandas via ``handlers``/``model_path`` (#286/#288). Attribute access, star
imports (via __all__), and submodule access behave as before — resolution
happens on first touch and is cached.
"""
from typing import TYPE_CHECKING

from views_pipeline_core._lazy import lazy_attach

from .constants import CACHE_FILENAME_TEMPLATE as CACHE_FILENAME_TEMPLATE
from .constants import CACHE_SOURCES as CACHE_SOURCES

if TYPE_CHECKING:  # pragma: no cover — static-analysis convenience only
    from .handlers import _ViewsDataset  # noqa: F401
    from .model_path import ModelPathManager  # noqa: F401
    from .utils import (  # noqa: F401
        convert_json_to_list_of_dicts,
        download_json,
        ensure_float64,
        replace_nan_values,
    )

#: name → defining submodule. Single source of truth for __all__/__getattr__/__dir__.
_LAZY_EXPORTS = {
    "_ViewsDataset": "handlers",
    "ModelPathManager": "model_path",
    "ensure_float64": "utils",
    "download_json": "utils",
    "convert_json_to_list_of_dicts": "utils",
    "replace_nan_values": "utils",
}
_LAZY_SUBMODULES = {
    "constants",
    "frame_invariants",
    "handlers",
    "model_path",
    "partitions",
    "utils",
}

__all__ = sorted({"CACHE_FILENAME_TEMPLATE", "CACHE_SOURCES", *_LAZY_EXPORTS})

__getattr__, __dir__ = lazy_attach(__name__, _LAZY_EXPORTS, _LAZY_SUBMODULES)
