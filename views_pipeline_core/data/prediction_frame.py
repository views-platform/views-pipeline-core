"""pipeline-core's ``PredictionFrame`` IS the published leaf type (views-frames).

Issue #188 retired the local duplicate in favour of
``views_frames.PredictionFrame`` (one canonical, numpy-only frame across the
platform). Existing imports of
``views_pipeline_core.data.prediction_frame.PredictionFrame`` keep working via
this re-export.

Two things that used to live on the local class moved:
  * on-disk persistence in pipeline-core's ``y_pred.npy`` layout →
    ``views_pipeline_core.managers.prediction.prediction_frame_io``
    (``save_pf`` / ``load_pf``) — kept as the cross-repo contract views-reporting
    reads (the leaf's own ``save`` uses a different ``values.npy`` layout);
  * sample-axis reduction (the old ``collapse``) → ``views_frames_summarize``
    (e.g. ``collapse(frame, np.mean)``).
"""
from views_frames import PredictionFrame

__all__ = ["PredictionFrame"]
