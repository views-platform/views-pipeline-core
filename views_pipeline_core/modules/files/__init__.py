"""File naming and I/O utilities.

  * :mod:`naming` — :class:`FilenameModule` for centralized filename generation.
"""
from typing import TYPE_CHECKING

from views_pipeline_core._lazy import lazy_attach

if TYPE_CHECKING:  # pragma: no cover
    from .naming import (  # noqa: F401
        FilenameModule,
        generate_evaluation_file_name,
        generate_evaluation_report_name,
        generate_model_file_name,
        generate_output_file_name,
    )

_LAZY_EXPORTS = {
    "FilenameModule": "naming",
    "generate_evaluation_file_name": "naming",
    "generate_evaluation_report_name": "naming",
    "generate_model_file_name": "naming",
    "generate_output_file_name": "naming",
}
_LAZY_SUBMODULES = {"naming"}
__all__ = sorted(_LAZY_EXPORTS)
__getattr__, __dir__ = lazy_attach(__name__, _LAZY_EXPORTS, _LAZY_SUBMODULES)
