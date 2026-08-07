"""Data layer package.

Constants are re-exported eagerly (stdlib-pure). Everything else is lazy
(PEP 562, shared machinery in views_pipeline_core/_lazy.py) so import-light
modules can depend on ``data.constants``/``data.partitions`` without paying for
pandas via ``handlers``/``model_path`` (#286/#288). Attribute access, star
imports (via __all__), and submodule access behave as before — resolution
happens on first touch and is cached.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from views_pipeline_core._lazy import lazy_attach

from views_pipeline_core.constants.data import (
    CACHE_FILENAME_TEMPLATE as CACHE_FILENAME_TEMPLATE,
)
from views_pipeline_core.constants.data import CACHE_SOURCES as CACHE_SOURCES

if TYPE_CHECKING:  # pragma: no cover — static-analysis convenience only
    from .handlers import _ViewsDataset  # noqa: F401
    from .utils import ensure_float64  # noqa: F401
    from views_pipeline_core.managers.model.path import ModelPathManager  # noqa: F401
    from views_pipeline_core.managers.path import PathManager  # noqa: F401

#: name → defining submodule. Single source of truth for __all__/__getattr__/__dir__.
_LAZY_EXPORTS = {
    "_ViewsDataset": "handlers",
    "ensure_float64": "utils",
}
_LAZY_SUBMODULES = {
    "frame_invariants",
    "handlers",
    "partitions",
    "utils",
}

__all__ = sorted({"CACHE_FILENAME_TEMPLATE", "CACHE_SOURCES", *_LAZY_EXPORTS})

_lazy_getattr, _lazy_dir = lazy_attach(__name__, _LAZY_EXPORTS, _LAZY_SUBMODULES)


def __getattr__(name: str):
    if name == "PathManager":
        from views_pipeline_core.managers.path import PathManager

        return PathManager
    if name == "ModelPathManager":
        from views_pipeline_core.managers.model.path import ModelPathManager

        return ModelPathManager
    return _lazy_getattr(name)


def __dir__():
    return sorted(set(_lazy_dir()) | {"PathManager", "ModelPathManager"})