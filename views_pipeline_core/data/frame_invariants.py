"""Shared FeatureFrame invariants (#288).

One definition of "empty" for the frame path — consumed by the cache boundary
(``frame_cache``) and the audit boundary (``CoreFrameSniffer``) so the two can
never disagree about what they refuse. (``prediction_frame_io.load_pf`` carries
the PredictionFrame-shaped analogue; unifying across frame kinds is a views-frames
question, tracked as C-212.)

Import-light: views-frames + stdlib only.
"""
from __future__ import annotations

from views_frames import FeatureFrame


def frame_is_empty(frame: FeatureFrame) -> bool:
    """True when the frame carries no rows or no features."""
    return frame.n_rows == 0 or frame.n_features == 0


def assert_frame_nonempty(frame: FeatureFrame, context: str) -> None:
    """Fail Loud and Proud on an empty frame; ``context`` names the boundary."""
    if frame_is_empty(frame):
        raise ValueError(
            f"Empty FeatureFrame ({frame.n_rows} rows, {frame.n_features} features) "
            f"{context}. An empty frame must never sit in a cache or reach a model."
        )
