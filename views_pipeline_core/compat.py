"""Backward-compat import paths — deprecated.

This module consolidates all backward-compat re-export shims into one
visible location (M-2 audit decision).

Historical import paths that are now redirected:

  from views_pipeline_core.data.model_path import ModelPathManager
      -> from views_pipeline_core.managers.model.model import ModelPathManager

  from views_pipeline_core.data.path import PathManager
      -> from views_pipeline_core.managers.path import PathManager

  from views_pipeline_core.modules.data import PredictionFrameConverter
      -> from views_pipeline_core.modules.frames import PredictionFrameConverter

  from views_pipeline_core.modules.data import save_pf, load_pf
      -> from views_pipeline_core.modules.frames import save_pf, load_pf

  from views_pipeline_core.managers.prediction.prediction_frame_converter import PredictionFrameConverter
      -> from views_pipeline_core.modules.frames import PredictionFrameConverter

  from views_pipeline_core.managers.prediction.prediction_frame_io import save_pf, load_pf
      -> from views_pipeline_core.modules.frames import save_pf, load_pf

  from views_pipeline_core.modules.appwrite.file import AppWriteFileModule
      -> from views_pipeline_core.modules.appwrite.storage import AppWriteFileModule
"""
from __future__ import annotations

from views_pipeline_core.managers.model.model import ModelPathManager  # noqa: F401
from views_pipeline_core.managers.path import PathManager  # noqa: F401
from views_pipeline_core.modules.frames.prediction_frame_converter import (  # noqa: F401
    PredictionFrameConverter,
)
from views_pipeline_core.modules.frames.prediction_frame_io import (  # noqa: F401
    load_pf,
    save_pf,
)
from views_pipeline_core.modules.files.naming import (  # noqa: F401
    FilenameModule,
    generate_evaluation_file_name,
    generate_evaluation_report_name,
    generate_model_file_name,
    generate_output_file_name,
)
from views_pipeline_core.modules.appwrite.storage import AppWriteFileModule  # noqa: F401

__all__ = [
    "AppWriteFileModule",
    "FilenameModule",
    "ModelPathManager",
    "PathManager",
    "PredictionFrameConverter",
    "generate_evaluation_file_name",
    "generate_evaluation_report_name",
    "generate_model_file_name",
    "generate_output_file_name",
    "load_pf",
    "save_pf",
]
