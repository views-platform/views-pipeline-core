"""Data loaders package.

Re-exports are lazy (PEP 562) so import-light submodules (fetch_context, and the
FeatureFrame path, #289) can be imported without paying for the heavy legacy
loader stack (viewser, ingester3, transformation library) that ``dataloaders``
pulls in at module level (#286). Same idiom as ``views_pipeline_core/data``:
name→submodule dict, resolved on first access, cached into module globals.
"""
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — static-analysis convenience only
    from .dataloaders import UpdateViewser, ViewsDataLoader  # noqa: F401

#: name → defining submodule. Single source of truth for __all__/__getattr__/__dir__.
_LAZY_EXPORTS = {
    "ViewsDataLoader": "dataloaders",
    "UpdateViewser": "dataloaders",
}
_LAZY_SUBMODULES = {"dataloaders", "fetch_context"}

__all__ = sorted(_LAZY_EXPORTS)


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
