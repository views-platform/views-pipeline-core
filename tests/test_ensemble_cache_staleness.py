"""S2 (sprint: kill silent sample-count failures) — config-aware forecast cache.

The ensemble cached each constituent forecast as y_pred.npy in a directory keyed
on the sample-agnostic MODEL ARTIFACT timestamp; editing the sample count never
changed the artifact, so `_load_or_generate_pf` reloaded the stale forecast at
the old count and the config change vanished (register C-85, the 2026-07-20 FAO
delivery). S1 made that fail loud AFTER the fact; S2 makes the cache
config-aware so it REGENERATES instead of reusing.

The check reads the cached array's ACTUAL sample width (which cannot lie about
itself) rather than a self-reported sidecar, and compares it to the ensemble's
expected_samples_per_model.
"""
import numpy as np
from unittest.mock import MagicMock
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

from views_pipeline_core.managers.ensemble.prediction_frame_ensemble import (
    PredictionFrameEnsembleManager,
    _cached_sample_count,
)
from views_pipeline_core.modules.frames.prediction_frame_io import save_pf


def _write_pf(pf_dir, n_rows, n_samples, seed=0):
    pf_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = SpatioTemporalIndex(
        np.arange(n_rows, dtype=np.int64),
        np.arange(1000, 1000 + n_rows, dtype=np.int64),
        SpatialLevel.PGM,
    )
    save_pf(
        PredictionFrame(rng.uniform(0, 5, (n_rows, n_samples)).astype(np.float32), idx),
        pf_dir,
    )


# --- the peek helper (reads reality, no sidecar) --------------------------

def test_cached_sample_count_reads_actual_width(tmp_path):
    _write_pf(tmp_path / "d", 8, 16)
    assert _cached_sample_count(tmp_path / "d") == 16


def test_cached_sample_count_none_when_absent(tmp_path):
    assert _cached_sample_count(tmp_path / "missing") is None


def test_cached_sample_count_none_for_nonstandard_array(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    np.save(d / "y_pred.npy", np.arange(5))  # 1-D, not (N, S)
    assert _cached_sample_count(d) is None


# --- the config-aware load: regenerate on mismatch ------------------------

def _bare_manager():
    mgr = object.__new__(PredictionFrameEnsembleManager)
    mgr._wandb_module = MagicMock()
    mgr._create_model_args = MagicMock(return_value=MagicMock())
    return mgr


def _ctx(expected):
    ctx = MagicMock()
    ctx.configs = {"level": "pgm"}
    ctx.expected_samples_per_model = expected
    return ctx


def test_reuses_cache_when_width_matches_expected(tmp_path):
    pf_dir = tmp_path / "predictions_forecasting_x" / "lr_sb_best"
    _write_pf(pf_dir, 8, 16)
    mgr = _bare_manager()
    mgr._execute_shell_script = MagicMock()  # must NOT be called

    out = mgr._load_or_generate_pf(
        model_path=MagicMock(), model_name="m", pf_dir=pf_dir,
        ctx=_ctx(16), forecast=True,
    )
    assert out.sample_count == 16
    mgr._execute_shell_script.assert_not_called()


def test_regenerates_when_cached_width_is_stale(tmp_path):
    # tonight's bug: cache is S=128, config now expects 16 → must regenerate.
    pf_dir = tmp_path / "predictions_forecasting_x" / "lr_sb_best"
    _write_pf(pf_dir, 8, 128)

    mgr = _bare_manager()

    def fake_shell(model_path, model_name, model_args):
        _write_pf(pf_dir, 8, 16)  # the constituent regenerates fresh at 16

    mgr._execute_shell_script = MagicMock(side_effect=fake_shell)

    out = mgr._load_or_generate_pf(
        model_path=MagicMock(), model_name="m", pf_dir=pf_dir,
        ctx=_ctx(16), forecast=True,
    )
    mgr._execute_shell_script.assert_called_once()  # did NOT blindly reuse
    assert out.sample_count == 16  # returns the fresh frame


def test_reuses_cache_when_no_expectation_declared(tmp_path):
    # ensembles without expected_samples_per_model keep the old load behaviour.
    pf_dir = tmp_path / "predictions_forecasting_x" / "lr_sb_best"
    _write_pf(pf_dir, 8, 64)
    mgr = _bare_manager()
    mgr._execute_shell_script = MagicMock()

    out = mgr._load_or_generate_pf(
        model_path=MagicMock(), model_name="m", pf_dir=pf_dir,
        ctx=_ctx(None), forecast=True,
    )
    assert out.sample_count == 64
    mgr._execute_shell_script.assert_not_called()