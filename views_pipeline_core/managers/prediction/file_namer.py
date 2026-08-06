"""Backward-compat re-export — use modules.files.naming.

The file naming logic has been consolidated into
:mod:`views_pipeline_core.modules.files.naming`.
"""
from views_pipeline_core.modules.files.naming import (  # noqa: F401
    FilenameModule,
    PredictionFileNamer,
)

__all__ = ["FilenameModule", "PredictionFileNamer"]
