"""Backward-compat re-export shim.

The ``save_pf`` / ``load_pf`` functions were relocated to
:mod:`views_pipeline_core.modules.frames.prediction_frame_io`
(audit decision: data-related prediction helpers belong in ``modules/data/``,
not in ``managers/prediction/``).

This shim preserves the historical import path
``from views_pipeline_core.managers.prediction.prediction_frame_io import save_pf, load_pf``
so existing callers (including r2darts2) continue to work.
"""
from views_pipeline_core.modules.frames.prediction_frame_io import (  # noqa: F401
    load_pf,
    save_pf,
)

__all__ = ["load_pf", "save_pf"]
