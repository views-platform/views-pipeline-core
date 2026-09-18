"""PerModelMetricFrameSource looks each model up where the producer wrote it (#485).

No fixture that exercised the eval-report path before this file put a comparison model's
frame under a root different from the subject's — the stage test mocked the file source
outright and views-reporting's tests used one model or an in-memory double — so nothing
exercised production's per-model-root shape. These tests lay it out: each model's frame
under that model's own ``data/generated``. The real ``MetricFrameFileSource`` is used for the lookup
(the locked layout is theirs and must be honoured, not mocked); only the frame
deserialisation, ``MetricFrame.load``, is stubbed to record the directory it was handed.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from views_pipeline_core.managers.evaluation.stage import METRICFRAME_DIR_PREFIX
from views_pipeline_core.managers.reporting.metric_frame_source import (
    PerModelMetricFrameSource,
    model_data_generated,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("views_reporting.sources") is None,
    reason="views-reporting lacks the EvaluationSource consumer (#173)",
)

RUN_TYPE = "calibration"
TARGET = "lr_sb"


def _write_frame_dir(data_generated: Path, model: str) -> Path:
    """The producer's layout: <data_generated>/<model>/<run_type>/metricframe_<target>."""
    d = data_generated / model / RUN_TYPE / f"{METRICFRAME_DIR_PREFIX}{TARGET}"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def platform(tmp_path):
    """An ensemble subject and two models, each with its own data/generated — production's
    shape: ensembles/<e>/data/generated and models/<m>/data/generated."""
    roots = {
        "big_chungus": tmp_path / "ensembles" / "big_chungus" / "data" / "generated",
        "black_ranger": tmp_path / "models" / "black_ranger" / "data" / "generated",
        "average_cmbaseline": tmp_path / "models" / "average_cmbaseline" / "data" / "generated",
    }
    written = {m: _write_frame_dir(r, m) for m, r in roots.items()}
    return roots, written


def _source(roots, primary="big_chungus"):
    return PerModelMetricFrameSource(
        primary_model=primary,
        primary_root=roots[primary],
        run_type=RUN_TYPE,
        target=TARGET,
        root_of=lambda name: roots[name],
    )


def _loaded_dirs():
    """Patch MetricFrame.load to record the directory and return a sentinel per call."""
    return patch(
        "views_reporting.sources.metric_frame_file_source.MetricFrame.load",
        side_effect=lambda d: ("frame", Path(d)),
    )


def test_a_constituent_is_found_under_its_own_data_generated(platform):
    """The #485 case: the subject is an ensemble; the constituent's frame lives under the
    constituent's directory, not the ensemble's. Before this class the lookup probed
    <ensemble>/data/generated/black_ranger/... and returned None."""
    roots, written = platform
    with _loaded_dirs():
        frame = _source(roots).metric_frame("black_ranger")
    assert frame == ("frame", written["black_ranger"])
    assert not (roots["big_chungus"] / "black_ranger").exists(), "the wrong path must not exist for this test to mean anything"


def test_a_baseline_is_found_the_same_way(platform):
    roots, written = platform
    with _loaded_dirs():
        assert _source(roots).metric_frame("average_cmbaseline") == ("frame", written["average_cmbaseline"])


def test_the_subject_is_found_under_the_root_the_caller_supplied(platform):
    """An ensemble is not resolvable by name through ModelPathManager (it lives under
    ensembles/, not models/), so its root is supplied — and root_of is never asked for it."""
    roots, written = platform
    asked = []

    def root_of(name):
        asked.append(name)
        return roots[name]

    source = PerModelMetricFrameSource("big_chungus", roots["big_chungus"], RUN_TYPE, TARGET, root_of=root_of)
    with _loaded_dirs():
        assert source.metric_frame("big_chungus") == ("frame", written["big_chungus"])
    assert asked == []


def test_an_absent_model_is_none_and_names_the_root_probed(platform, caplog):
    """Absent stays absent (the port's contract: the report degrades-and-announces) — but
    the log now says WHERE it looked, which is what #485's operator had to reconstruct."""
    roots, _ = platform
    roots["missing_model"] = roots["black_ranger"].parent.parent.parent / "missing_model" / "data" / "generated"
    with caplog.at_level(logging.INFO), _loaded_dirs():
        assert _source(roots).metric_frame("missing_model") is None
    assert "missing_model" in caplog.text and str(roots["missing_model"]) in caplog.text


def test_provenance_comes_from_the_subjects_own_frame(platform):
    """provenance() reads the subject's frame; the frame stub here has no metadata, so the
    real MetricFrameFileSource.provenance() is replaced and only the routing is asserted."""
    roots, _ = platform
    source = _source(roots)
    routed = []
    with patch(
        "views_reporting.sources.metric_frame_file_source.MetricFrameFileSource.provenance",
        autospec=True,
        side_effect=lambda self: routed.append(self._root) or "provenance",
    ):
        assert source.provenance() == "provenance"
    assert routed == [roots["big_chungus"]]


def test_the_locked_layout_is_the_producers(platform):
    """Every per-model lookup still uses views-reporting's layout under the model's root —
    the locked cross-repo contract (C-202 / their C-192) is untouched; only the root moved."""
    roots, _ = platform
    source = _source(roots)
    for model in ("big_chungus", "black_ranger"):
        file_source = source._source_for(model)
        assert file_source._frame_dir(model) == roots[model] / model / RUN_TYPE / f"{METRICFRAME_DIR_PREFIX}{TARGET}"
        # Each file source is scoped to ITS model — provenance() on it would read that
        # model's frame, not the subject's (guard audit M7: inert today, so pinned here).
        assert file_source._primary_model == model
        assert source._source_for(model) is file_source  # built once per model, reused


def test_default_root_resolution_is_the_ensemble_managers_lookup():
    """The default root_of is ModelPathManager(name, validate=False).data_generated — the
    same resolution the ensemble managers use for a constituent — and it does not raise
    for a model that is not checked out (the lookup must be able to report absence)."""
    with patch("views_pipeline_core.managers.reporting.metric_frame_source.ModelPathManager") as mpm:
        mpm.return_value.data_generated = Path("/models/black_ranger/data/generated")
        assert model_data_generated("black_ranger") == Path("/models/black_ranger/data/generated")
    mpm.assert_called_once_with("black_ranger", validate=False)
