"""Falsification stubs: Epic #261 S1 readiness audit (2026-07-05).

The claim "we are ready to start the real-scale PFE proof" was FALSIFIED
(2 hard, 2 soft). Each finding below carries a stub. The code-level stub
(#160 guard) is xfail-until-fixed; the environment-level probes only run
when RUN_S1_READINESS=1 so CI stays green while the register/issues track
the operational fixes.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

from views_pipeline_core.managers.ensemble.prediction_frame_ensemble import (
    _aggregate_prediction_frames,
)

_ENV_GATED = pytest.mark.skipif(
    os.environ.get("RUN_S1_READINESS") != "1",
    reason="S1 operational readiness probe — run with RUN_S1_READINESS=1",
)


def _pf(n_rows, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    idx = SpatioTemporalIndex(
        np.repeat(np.int64(500), n_rows),
        np.arange(n_rows, dtype=np.int64),
        SpatialLevel.PGM,
    )
    return PredictionFrame(
        rng.uniform(0, 5, size=(n_rows, n_samples)).astype(np.float32), idx
    )


# --- RESOLVED (issue #160, probe P3, register C-205): guard landed ---
def test_pfe_concat_rejects_heterogeneous_sample_counts():
    frames = [_pf(10, 128, seed=1), _pf(10, 8, seed=2)]
    with pytest.raises(ValueError, match="sample_count mismatch"):
        _aggregate_prediction_frames(frames, "concat")


def test_pfe_mean_rejects_heterogeneous_sample_counts():
    # The mean path previously crashed cryptically inside np.stack; now it fails
    # with the same clear guard message as concat.
    frames = [_pf(10, 16, seed=3), _pf(10, 8, seed=4)]
    with pytest.raises(ValueError, match="sample_count mismatch"):
        _aggregate_prediction_frames(frames, "arithmetic_mean")


def test_pfe_concat_homogeneous_still_pools():
    frames = [_pf(10, 8, seed=5), _pf(10, 8, seed=6), _pf(10, 8, seed=7)]
    out = _aggregate_prediction_frames(frames, "concat")
    assert out.sample_count == 24  # 3 × 8, nothing dropped


# --- HARD #1 (probe P4): disk headroom for the S1 run ---
@_ENV_GATED
def test_s1_disk_headroom_for_rusty_bucket_at_declared_samples():
    """rusty_bucket @ n_samples=128: ~151 GB constituent evals + ~151 GB ensemble
    output ≈ 300 GB. Fails while the volume holds < the required headroom."""
    required_gb = 320
    free_gb = shutil.disk_usage("/home/simon/Documents/scripts/views_platform").free / 1e9
    assert free_gb >= required_gb, (
        f"S1 disk headroom: {free_gb:.0f} GB free < {required_gb} GB required "
        f"(reduce n_samples / constituent count, or free space)."
    )


# --- HARD #2 (probe P1): proof must run engine code slated for publication ---
@_ENV_GATED
def test_s1_hydranet_worktree_on_development():
    """The publish-gate proof is invalid if the editable views-hydranet runs an
    experimental branch (feat/zinb-distributional-head, 62 ahead / 2 behind)."""
    out = subprocess.run(
        ["git", "-C", "/home/simon/Documents/scripts/views_platform/views-hydranet",
         "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "development", (
        f"views-hydranet editable worktree is on '{out}', not 'development' — "
        f"the S1 proof would validate unpublished experimental engine code."
    )


# --- RESOLVED (probe P7, register C-208): the contract is encoded ---
def test_s1_acceptance_criteria_are_encoded():
    """The publish-gate acceptance contract lives in tests/test_s1_publish_gate.py
    (executable, env-gated); criteria 4-5 (exit 0, peak RSS) are runner-recorded."""
    contract = os.path.join(os.path.dirname(__file__), "test_s1_publish_gate.py")
    assert os.path.exists(contract), "S1 acceptance contract file is missing."
