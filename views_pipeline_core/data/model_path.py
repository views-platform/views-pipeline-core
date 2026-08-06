"""Backward-compat re-export.

ModelPathManager has been moved to managers/model/model.py per the
user's directive. This shim preserves the historical import path
``from views_pipeline_core.data.model_path import ModelPathManager``
for lower-layer callers (data/, modules/) that cannot import from
managers/ due to ADR-002 layer dependency rules.
"""
from views_pipeline_core.managers.model.model import ModelPathManager  # noqa: F401

__all__ = ["ModelPathManager"]
