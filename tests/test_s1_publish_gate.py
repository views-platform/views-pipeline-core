"""Epic #261 S1 — the encoded publish-gate acceptance contract (resolves C-208).

The S1 proof (real-scale PFE ensemble evaluation) passes ONLY if all of the
following hold. Criteria 1–3 are executable here against the run's output;
criteria 4–5 are procedural and recorded by the runner in the run log:

  1. Pooled sample_count == Σ constituent samples   (checked here)
  2. All pooled prediction values are finite         (checked here)
  3. A MetricFrame is persisted per target           (checked here)
  4. The run exits 0 (no OOM kill, no exception)     (runner-recorded)
  5. Peak RSS recorded and < box RAM                 (runner-recorded)

Run after the S1 evaluation completes:

    S1_ENSEMBLE_PRED_DIR=/path/to/ensembles/<name>/data/generated/predictions_calibration_<ts> \\
    S1_METRICFRAME_ROOT=/path/to/ensembles/<name>/data/generated \\
    S1_EXPECTED_SAMPLES=1024 \\
    pytest tests/test_s1_publish_gate.py -v

Skipped entirely (CI-safe) when the env vars are absent.
"""
import os
from pathlib import Path

import numpy as np
import pytest

_PRED_DIR = os.environ.get("S1_ENSEMBLE_PRED_DIR")
_MF_ROOT = os.environ.get("S1_METRICFRAME_ROOT")
_EXPECTED_SAMPLES = os.environ.get("S1_EXPECTED_SAMPLES")

pytestmark = pytest.mark.skipif(
    not (_PRED_DIR and _EXPECTED_SAMPLES),
    reason="S1 publish-gate contract — run post-S1 with S1_ENSEMBLE_PRED_DIR "
    "and S1_EXPECTED_SAMPLES set (see module docstring).",
)


def _pooled_ypred_files():
    files = sorted(Path(_PRED_DIR).rglob("y_pred.npy"))
    assert files, f"No y_pred.npy under {_PRED_DIR} — the run produced no pooled output."
    return files


def test_c1_pooled_sample_count_equals_sum_of_constituents():
    expected = int(_EXPECTED_SAMPLES)
    for f in _pooled_ypred_files():
        y = np.load(f, mmap_mode="r")
        assert y.ndim == 2 and y.shape[1] == expected, (
            f"{f}: pooled sample_count {y.shape[1]} != Σ constituent samples "
            f"{expected} — pooling dropped or duplicated draws."
        )


def test_c2_all_pooled_values_finite():
    for f in _pooled_ypred_files():
        y = np.load(f, mmap_mode="r")
        # chunked scan — files can be ~2 GB
        step = max(1, y.shape[0] // 16)
        for lo in range(0, y.shape[0], step):
            assert np.isfinite(y[lo : lo + step]).all(), (
                f"{f}: non-finite values in rows [{lo}, {lo + step})."
            )


def test_c3_metricframe_persisted_per_target():
    assert _MF_ROOT, "Set S1_METRICFRAME_ROOT to check MetricFrame persistence."
    mf_dirs = sorted(Path(_MF_ROOT).rglob("metricframe_*"))
    assert mf_dirs, (
        f"No metricframe_<target> directory under {_MF_ROOT} — the evaluation "
        f"of record was not persisted."
    )
    targets_with_preds = {p.parent.name for p in _pooled_ypred_files()}
    mf_targets = {d.name.removeprefix("metricframe_") for d in mf_dirs}
    missing = targets_with_preds - mf_targets
    assert not missing, f"Targets with predictions but no MetricFrame: {sorted(missing)}"
