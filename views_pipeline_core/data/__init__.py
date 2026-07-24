"""Data layer package.

Constants are re-exported eagerly (stdlib-pure). Everything else is lazy
(PEP 562) so import-light modules can depend on ``data.constants`` without
paying for pandas via ``handlers``/``model_path`` (#286). Attribute access
(``from views_pipeline_core.data import _ViewsDataset``, star-imports via
``__all__``, and submodule access like ``data.model_path``) behaves as before —
resolution happens on first touch and is cached into module globals. Same idiom
as ``views_pipeline_core/modules/dataloaders``.
"""
import importlib
from typing import TYPE_CHECKING

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
_LAZY_SUBMODULES = {"constants", "handlers", "model_path", "utils"}

__all__ = sorted(
    {"CACHE_FILENAME_TEMPLATE", "CACHE_SOURCES", *_LAZY_EXPORTS}
)


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        value = getattr(
            importlib.import_module(f".{_LAZY_EXPORTS[name]}", __name__), name
        )
    elif name in _LAZY_SUBMODULES:
        value = importlib.import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # cache: later accesses skip __getattr__ entirely
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | _LAZY_SUBMODULES)
