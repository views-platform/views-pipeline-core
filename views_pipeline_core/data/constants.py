"""Backward-compat re-export from constants/data.py.

Constants have been centralized in views_pipeline_core/constants/data.py
per the user's directive to "scan the codebase for all constants and
organise them in one centralised spot."
"""
from views_pipeline_core.constants.data import (  # noqa: F401
    CACHE_FILENAME_TEMPLATE,
    CACHE_SOURCES,
    FRAME_CACHE_DIRNAME_TEMPLATE,
    PARTITION_TEST,
    PARTITION_TRAIN,
    RUN_TYPE_CALIBRATION,
    RUN_TYPE_FORECASTING,
    RUN_TYPE_VALIDATION,
    TRAINING_RUN_TYPES,
)