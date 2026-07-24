"""PEP 562 lazy re-export machinery shared by package __init__ files (#288).

Extracted at copy-count three (data/, modules/dataloaders/, modules/validation/ —
WET-before-DRY threshold met): one implementation of the export/submodule
dispatch, first-access caching, and dir() surface. stdlib-pure by construction —
the import-light packages depend on this module.

Usage in a package __init__::

    _LAZY_EXPORTS = {"Name": "submodule", ...}
    _LAZY_SUBMODULES = {"submodule", ...}
    __all__ = sorted(_LAZY_EXPORTS)
    __getattr__, __dir__ = lazy_attach(__name__, _LAZY_EXPORTS, _LAZY_SUBMODULES)
"""
from __future__ import annotations

import importlib
import sys
from typing import Callable, Dict, Set, Tuple


def lazy_attach(
    package: str,
    exports: Dict[str, str],
    submodules: Set[str],
) -> Tuple[Callable, Callable]:
    """Build (__getattr__, __dir__) for a lazily re-exporting package.

    ``exports`` maps public name → defining submodule; ``submodules`` are
    importable as attributes too. Resolved values are cached into the package's
    globals so later accesses bypass __getattr__ entirely.
    """

    def __getattr__(name: str):
        if name in exports:
            value = getattr(
                importlib.import_module(f".{exports[name]}", package), name
            )
        elif name in submodules:
            value = importlib.import_module(f".{name}", package)
        else:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        setattr(sys.modules[package], name, value)  # cache for later accesses
        return value

    def __dir__():
        return sorted(set(vars(sys.modules[package])) | set(exports) | submodules)

    return __getattr__, __dir__
