"""On-disk persistence for the leaf ``views_frames.PredictionFrame`` in
pipeline-core's ``y_pred.npy`` layout.

The leaf's own ``PredictionFrame.save`` writes ``values.npy`` + ``header.json``;
pipeline-core deliberately keeps the **``y_pred.npy`` + ``identifiers.npz``**
layout because it is a cross-repo contract — views-reporting's
``PredictionFrameLoader`` reads ``y_pred.npy`` directly (issue #188; it does NOT
call the leaf's ``load``). These helpers serialize/deserialize a leaf frame in
that layout. ``level`` is supplied by the caller (from ``config["level"]``)
because the layout does not persist it.
"""
from pathlib import Path

import numpy as np
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex


def save_pf(frame: PredictionFrame, directory: Path) -> None:
    """Write *frame* to *directory* as ``y_pred.npy`` + ``identifiers.npz``.

    Layout-compatible with the pre-#188 ``PredictionFrame.save`` and with
    views-reporting's reader. Overwrites cleanly; creates *directory* if absent.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "y_pred.npy", frame.values)
    np.savez(directory / "identifiers.npz", **frame.identifiers)


def load_pf(directory: Path, level: str, mmap: bool = False) -> PredictionFrame:
    """Read a leaf ``PredictionFrame`` from a ``y_pred.npy`` directory.

    Args:
        directory: holds ``y_pred.npy`` and ``identifiers.npz``.
        level: spatial level string (``"cm"``/``"pgm"``) — the layout does not
            store it, so it is supplied here to build the ``SpatioTemporalIndex``.
        mmap: memory-map ``y_pred`` (read-only) — bounds peak RAM to the working
            set during the metrics reload phase.
    """
    directory = Path(directory)
    mmap_mode = "r" if mmap else None
    y_pred = np.load(directory / "y_pred.npy", mmap_mode=mmap_mode)
    # Fail loud on an empty/corrupted saved frame at the load boundary. The leaf
    # PredictionFrame accepts 0-row / 0-sample arrays (it lets numpy reject them
    # downstream); pipeline-core's retired local class rejected them at construction
    # (n_rows>0, sample_count>=1). Re-establish that guarantee here rather than let
    # an empty frame propagate silently into eval/ensemble (Fail Loud and Proud).
    if y_pred.ndim != 2 or y_pred.shape[0] == 0 or y_pred.shape[1] == 0:
        raise ValueError(
            f"Loaded PredictionFrame at {directory} has invalid shape {y_pred.shape}; "
            f"expected a non-empty 2D (N>0, S>0) y_pred.npy."
        )
    with np.load(directory / "identifiers.npz") as f:
        ids = dict(f)
    index = SpatioTemporalIndex(
        time=np.asarray(ids["time"], dtype=np.int64),
        unit=np.asarray(ids["unit"], dtype=np.int64),
        level=SpatialLevel(level),
    )
    return PredictionFrame(y_pred, index)