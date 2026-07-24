"""Data loaders package.

Re-exports are lazy (PEP 562, shared machinery in views_pipeline_core/_lazy.py)
so import-light submodules (fetch_context, frame_cache, and the FeatureFrame
path, #289) can be imported without paying for the heavy legacy loader stack
(viewser, ingester3, transformation library) that ``dataloaders`` pulls in at
module level (#286).
"""
from typing import TYPE_CHECKING

from views_pipeline_core._lazy import lazy_attach

if TYPE_CHECKING:  # pragma: no cover — static-analysis convenience only
    from .dataloaders import UpdateViewser, ViewsDataLoader  # noqa: F401

#: name → defining submodule. Single source of truth for __all__/__getattr__/__dir__.
_LAZY_EXPORTS = {
    "ViewsDataLoader": "dataloaders",
    "UpdateViewser": "dataloaders",
}
_LAZY_SUBMODULES = {"dataloaders", "fetch_context", "frame_cache"}

__all__ = sorted(_LAZY_EXPORTS)

__getattr__, __dir__ = lazy_attach(__name__, _LAZY_EXPORTS, _LAZY_SUBMODULES)
