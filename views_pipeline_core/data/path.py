"""Backward-compat re-export.

PathManager has been moved to managers/path.py per the user's directive.
This shim preserves the historical import path for lower-layer callers.
"""
from views_pipeline_core.managers.path import PathManager  # noqa: F401

__all__ = ["PathManager"]
