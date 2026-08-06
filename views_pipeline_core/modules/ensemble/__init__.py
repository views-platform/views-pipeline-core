"""Ensemble-related shared helpers.

Currently provides:
  * :mod:`subprocess_runner` — :func:`execute_model_subprocess` for
    delegating model execution to a shell subprocess (extracted from
    the byte-identical ``_execute_shell_script`` methods on
    ``DataFrameEnsembleManager`` and ``PredictionFrameEnsembleManager``).
"""
from typing import TYPE_CHECKING

from views_pipeline_core._lazy import lazy_attach

if TYPE_CHECKING:  # pragma: no cover — static-analysis convenience only
    from .subprocess_runner import (  # noqa: F401
        DEFAULT_TIMEOUT_SECONDS,
        execute_model_subprocess,
    )

_LAZY_EXPORTS = {
    "DEFAULT_TIMEOUT_SECONDS": "subprocess_runner",
    "execute_model_subprocess": "subprocess_runner",
}
_LAZY_SUBMODULES = {"subprocess_runner"}
__all__ = sorted(_LAZY_EXPORTS)
__getattr__, __dir__ = lazy_attach(__name__, _LAZY_EXPORTS, _LAZY_SUBMODULES)
