"""S1 (sprint: kill silent sample-count failures) — the fail-loud guard.

The #160 balance guard (`_aggregate_prediction_frames`) rejects constituents
whose sample counts DIFFER from each other. It cannot catch the case that cost
the 2026-07-20 FAO delivery ~4 hours: every constituent reloaded a *stale*
cached forecast at the SAME wrong count (all-equal-but-wrong), so the config's
sample-count change was silently discarded with no error, log, or test.

`_assert_expected_sample_count` closes that gap: it compares each constituent's
PRODUCED `pf.sample_count` against the ensemble config's declared
`expected_samples_per_model` and fails loud on mismatch, while returning the
produced counts for forecast-time logging. Register C-85.
"""
import numpy as np
import pytest
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

from views_pipeline_core.managers.ensemble.prediction_frame_ensemble import (
    _assert_expected_sample_count,
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


def _results(models, sample_count, targets=("lr_sb_best",)):
    """A model_results dict: {model: {target: PF}} all at `sample_count`."""
    return {
        m: {t: _pf(6, sample_count, seed=i) for t in targets}
        for i, m in enumerate(models)
    }


MODELS = ["a", "b", "c"]
TARGETS = ["lr_sb_best"]


def test_returns_produced_counts_when_all_match():
    results = _results(MODELS, 16)
    produced = _assert_expected_sample_count(results, 16, MODELS, TARGETS)
    assert produced == {"a": 16, "b": 16, "c": 16}


def test_raises_when_all_equal_but_wrong():
    # tonight's exact bug: config expects 16, stale cache served 128 everywhere.
    # #160 sees them equal and passes; this guard must fail loud.
    results = _results(MODELS, 128)
    with pytest.raises(ValueError, match="expects 16"):
        _assert_expected_sample_count(results, 16, MODELS, TARGETS)


def test_error_names_the_constituent_and_the_remedy():
    results = _results(MODELS, 128)
    with pytest.raises(ValueError) as exc:
        _assert_expected_sample_count(results, 16, MODELS, TARGETS)
    msg = str(exc.value)
    assert "'a'" in msg and "128" in msg
    assert "predictions_" in msg  # points at the stale-cache remedy


def test_no_expectation_declared_skips_enforcement_but_still_reports():
    # An ensemble that declares no expected_samples_per_model must not crash;
    # the produced counts are still returned for logging.
    results = _results(MODELS, 64)
    produced = _assert_expected_sample_count(results, None, MODELS, TARGETS)
    assert produced == {"a": 64, "b": 64, "c": 64}


def test_catches_a_single_divergent_constituent():
    # mixed on disk (some regenerated, some stale) — must fire on the wrong one.
    results = _results(["a", "b"], 16)
    results["c"] = {"lr_sb_best": _pf(6, 128, seed=9)}
    with pytest.raises(ValueError, match="'c'.*128|128.*expects 16"):
        _assert_expected_sample_count(results, 16, MODELS, TARGETS)