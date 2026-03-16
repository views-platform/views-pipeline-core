"""
Tests for ModelManager._evaluate_model_artifact_streaming() and the
Track A (computation format) persistence pattern used in the PF evaluation path.

The streaming hook emits one pf_dict per rolling origin by calling
origin_sink(origin_idx, pf_dict). The default implementation wraps the
existing batch _evaluate_model_artifact() for backward compatibility.

Subclasses override _evaluate_model_artifact_streaming() to emit one
origin at a time without accumulating all origins in memory first.
"""

import gc
import tempfile
import weakref
from pathlib import Path

import numpy as np

from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.managers.model.model import ForecastingModelManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pf(n_rows: int = 4, n_samples: int = 2) -> PredictionFrame:
    return PredictionFrame(
        y_pred=np.ones((n_rows, n_samples), dtype=np.float32),
        identifiers={
            "time": np.arange(n_rows, dtype=np.int64),
            "unit": np.arange(n_rows, dtype=np.int64) + 100,
        },
    )


class _StreamingStub(ForecastingModelManager):
    """Minimal concrete stub that bypasses the real pipeline stack."""

    def _train_model_artifact(self, *a, **kw): pass
    def _evaluate_sweep(self, *a, **kw): return []
    def _forecast_model_artifact(self, *a, **kw): return {}

    def _evaluate_model_artifact(self, eval_type, artifact_name):
        return getattr(self, "_batch_return", {})


def _make_manager(batch_return: dict) -> _StreamingStub:
    """
    Build a _StreamingStub wired only with the attributes that
    _evaluate_model_artifact_streaming() reads.
    """
    m = object.__new__(_StreamingStub)
    m._batch_return = batch_return
    return m


# ---------------------------------------------------------------------------
# Default streaming hook — wraps batch method correctly
# ---------------------------------------------------------------------------

class TestStreamingHookDefault:

    def test_emits_one_dict_per_origin(self):
        """Default hook calls origin_sink once per rolling origin."""
        batch = {
            "fatalities": [_make_pf() for _ in range(3)],
            "violence":   [_make_pf() for _ in range(3)],
        }
        manager = _make_manager(batch)
        emitted = []
        manager._evaluate_model_artifact_streaming(
            "standard", "artifact.pt",
            origin_sink=lambda i, d: emitted.append(i),
        )
        assert len(emitted) == 3

    def test_emits_correct_origin_indices(self):
        """origin_idx argument is 0-based and sequential."""
        batch = {"target": [_make_pf() for _ in range(4)]}
        manager = _make_manager(batch)
        indices = []
        manager._evaluate_model_artifact_streaming(
            "standard", "artifact.pt",
            origin_sink=lambda i, d: indices.append(i),
        )
        assert indices == [0, 1, 2, 3]

    def test_each_pf_dict_has_all_target_keys(self):
        """Every emitted pf_dict contains all targets."""
        batch = {
            "fatalities": [_make_pf() for _ in range(2)],
            "violence":   [_make_pf() for _ in range(2)],
            "extent":     [_make_pf() for _ in range(2)],
        }
        manager = _make_manager(batch)
        received_keys = []
        manager._evaluate_model_artifact_streaming(
            "standard", "artifact.pt",
            origin_sink=lambda i, d: received_keys.append(set(d.keys())),
        )
        assert all(k == {"fatalities", "violence", "extent"} for k in received_keys)

    def test_each_value_is_a_single_predictionframe_not_a_list(self):
        """pf_dict values are PredictionFrame instances, not lists."""
        batch = {"target": [_make_pf(), _make_pf()]}
        manager = _make_manager(batch)
        value_types = []
        manager._evaluate_model_artifact_streaming(
            "standard", "artifact.pt",
            origin_sink=lambda i, d: value_types.append(type(d["target"]).__name__),
        )
        assert value_types == ["PredictionFrame", "PredictionFrame"]

    def test_emitted_pf_matches_batch_entry_for_same_origin(self):
        """The PF emitted for origin i is the same object as batch[target][i]."""
        pf0 = _make_pf()
        pf1 = _make_pf()
        batch = {"target": [pf0, pf1]}
        manager = _make_manager(batch)
        emitted = []
        manager._evaluate_model_artifact_streaming(
            "standard", "artifact.pt",
            origin_sink=lambda i, d: emitted.append(d["target"]),
        )
        assert emitted[0] is pf0
        assert emitted[1] is pf1

    def test_single_origin_single_target(self):
        """Edge case: M=1 origin, T=1 target — exactly one sink call."""
        batch = {"only_target": [_make_pf()]}
        manager = _make_manager(batch)
        calls = []
        manager._evaluate_model_artifact_streaming(
            "standard", "artifact.pt",
            origin_sink=lambda i, d: calls.append((i, d)),
        )
        assert len(calls) == 1
        assert calls[0][0] == 0
        assert isinstance(calls[0][1]["only_target"], PredictionFrame)


# ---------------------------------------------------------------------------
# Subclass override — streaming without batch accumulation
# ---------------------------------------------------------------------------

class TestStreamingHookOverride:

    def test_subclass_override_is_called_instead_of_default(self):
        """A subclass that overrides streaming emits its own results."""

        class CustomStreamer(_StreamingStub):
            def _evaluate_model_artifact_streaming(self, eval_type, name, origin_sink):
                for i in range(5):
                    origin_sink(i, {"custom_target": _make_pf()})

        manager = object.__new__(CustomStreamer)
        indices = []
        manager._evaluate_model_artifact_streaming(
            "standard", "artifact.pt",
            origin_sink=lambda i, d: indices.append(i),
        )
        assert indices == [0, 1, 2, 3, 4]

    def test_subclass_override_frees_pf_after_sink(self):
        """
        A well-behaved streaming override passes the PF to the sink and then
        frees it — no PF survives after origin_sink returns.
        """
        weak_refs = []

        class ProperStreamer(_StreamingStub):
            def _evaluate_model_artifact_streaming(self, eval_type, name, origin_sink):
                for i in range(3):
                    pf_dict = {"target": _make_pf()}
                    weak_refs.append(weakref.ref(pf_dict["target"]))
                    origin_sink(i, pf_dict)
                    del pf_dict
                    gc.collect()

        manager = object.__new__(ProperStreamer)
        manager._evaluate_model_artifact_streaming(
            "standard", "artifact.pt",
            origin_sink=lambda i, d: None,  # sink does not hold a reference
        )
        gc.collect()
        assert all(r() is None for r in weak_refs), (
            "At least one PredictionFrame was not freed after origin_sink returned. "
            "The streaming override must del pf_dict after calling origin_sink."
        )


# ---------------------------------------------------------------------------
# Track A — the sink must write .npy staging files that PredictionFrame.load
# can reconstruct.  This is the "computation format" contract.
# ---------------------------------------------------------------------------

class TestTrackASinkPattern:
    """
    The origin_sink used in _execute_model_evaluation() writes Track A files.

    These tests verify that the pattern produces files PredictionFrame.load()
    can round-trip — independently of the full pipeline stack.
    """

    def test_sink_pattern_creates_y_pred_npy_per_origin_per_target(self):
        """Track A: sink must save y_pred.npy for each (origin, target) pair."""
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "_pf_staging"
            targets = ["fatalities", "violence"]
            pf = _make_pf()

            # Replicate the sink pattern that _execute_model_evaluation will use
            for origin_idx in range(3):
                for target in targets:
                    pf.save(staging / f"origin_{origin_idx}" / target)

            for origin_idx in range(3):
                for target in targets:
                    assert (staging / f"origin_{origin_idx}" / target / "y_pred.npy").exists()

    def test_sink_pattern_reload_with_mmap_is_valid_pf(self):
        """
        After sink writes Track A files, reloading with mmap=True produces
        valid PredictionFrames (right shape, right dtype).
        """
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "_pf_staging"
            targets = ["target_a"]
            n_origins = 2
            original = _make_pf(n_rows=6, n_samples=8)

            for i in range(n_origins):
                original.save(staging / f"origin_{i}" / targets[0])

            reloaded = [
                PredictionFrame.load(staging / f"origin_{i}" / targets[0], mmap=True)
                for i in range(n_origins)
            ]

        assert len(reloaded) == n_origins
        for pf in reloaded:
            assert pf.n_rows == original.n_rows
            assert pf.sample_count == original.sample_count
            assert isinstance(pf.y_pred, np.memmap)

    def test_staging_files_survive_the_sink_loop(self):
        """
        Track A files written by the sink persist until explicitly removed,
        allowing the metrics phase to reload them after PFs are freed.
        """
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "_pf_staging"
            pf_refs = []

            # Simulate the streaming loop: create PF, save Track A, free PF
            for i in range(3):
                pf = _make_pf()
                pf_refs.append(weakref.ref(pf))
                pf.save(staging / f"origin_{i}" / "target")
                del pf
                gc.collect()

            # PFs should be gone
            assert all(r() is None for r in pf_refs)

            # But staging files must still be on disk
            for i in range(3):
                assert (staging / f"origin_{i}" / "target" / "y_pred.npy").exists()
