"""#227 (epic #224) — MetricFrame faithfulness gate.

Round-trip + parity (the ``group_id="mean"`` rows == the nanmean of the per-group legacy
metrics — the value views-reporting reads) + provenance-presence, against the REAL
views-evaluation / views-frames substrate, fully offline (a synthetic ``EvaluationReport`` —
no wandb / viewser). Capability-skipped where the optional substrate is absent (PyPI-only CI).
"""
import numpy as np
import pytest

pytest.importorskip("views_frames")
_er_mod = pytest.importorskip("views_evaluation.evaluation.evaluation_report")
_mf_mod = pytest.importorskip("views_evaluation.evaluation.metric_frame")

EvaluationReport = _er_mod.EvaluationReport
MetricFrame = _mf_mod.MetricFrame
MEAN_GROUP_ID = _mf_mod.MEAN_GROUP_ID
SCHEMA_TO_EVAL_TYPE = _mf_mod.SCHEMA_TO_EVAL_TYPE

if not hasattr(EvaluationReport, "to_metric_frame"):  # older views-evaluation
    pytest.skip("views-evaluation predates to_metric_frame", allow_module_level=True)

MONTH = SCHEMA_TO_EVAL_TYPE["month"]


def _report():
    # Two groups so the cross-group "mean" row is a non-trivial nanmean.
    results = {
        "month": {
            "g1": {"MSE": 2.0, "MAE": 1.0},
            "g2": {"MSE": 4.0, "MAE": 3.0},
        },
    }
    return EvaluationReport(
        target="lr_sb", task="regression", pred_type="point", results=results
    )


def _mf(**over):
    kw = dict(model_id="my_model", run_type="calibration", partition="(1, 2)", level="pgm")
    kw.update(over)
    return _report().to_metric_frame(**kw)


def _row_value(mf, *, eval_type, metric, group_id):
    ids = mf.identifiers
    mask = (
        (ids["eval_type"] == eval_type)
        & (ids["metric"] == metric)
        & (ids["group_id"] == group_id)
    )
    idx = np.flatnonzero(mask)
    assert idx.size == 1, f"expected exactly one row, got {idx.size}"
    return float(mf.values[idx[0], 0])


class TestParity:
    def test_mean_rows_equal_nanmean_of_groups(self):
        mf = _mf()
        # nanmean(2,4)=3 ; nanmean(1,3)=2 — the rows views-reporting matches on.
        assert _row_value(mf, eval_type=MONTH, metric="MSE", group_id=MEAN_GROUP_ID) == pytest.approx(3.0)
        assert _row_value(mf, eval_type=MONTH, metric="MAE", group_id=MEAN_GROUP_ID) == pytest.approx(2.0)

    def test_per_group_rows_preserved(self):
        mf = _mf()
        assert _row_value(mf, eval_type=MONTH, metric="MSE", group_id="g1") == pytest.approx(2.0)
        assert _row_value(mf, eval_type=MONTH, metric="MSE", group_id="g2") == pytest.approx(4.0)

    def test_values_are_float32_column_vector(self):
        mf = _mf()
        assert mf.values.dtype == np.float32
        assert mf.values.ndim == 2 and mf.values.shape[1] == 1


class TestRoundTrip:
    def test_save_load_preserves_values_identifiers_metadata(self, tmp_path):
        mf = _mf()
        directory = tmp_path / "mf"
        mf.save(directory)
        loaded = MetricFrame.load(directory)

        np.testing.assert_array_equal(loaded.values, mf.values)
        assert set(loaded.identifiers) == set(mf.identifiers)
        for axis in mf.identifiers:
            np.testing.assert_array_equal(loaded.identifiers[axis], mf.identifiers[axis])
        assert loaded.metadata.provenance.model == "my_model"
        assert loaded.metadata.provenance.run_type == "calibration"

    def test_round_trip_mean_value_stable(self, tmp_path):
        directory = tmp_path / "mf"
        _mf().save(directory)
        loaded = MetricFrame.load(directory)
        assert _row_value(loaded, eval_type=MONTH, metric="MSE", group_id=MEAN_GROUP_ID) == pytest.approx(3.0)


class TestCrossRepoPathContract:
    """C-202 durable guard: the producer's save path (pipeline-core) must be exactly where
    views-reporting's MetricFrameFileSource reads. Ties the producer prefix constant to the
    real consumer loader, so a prefix/layout change on either side fails this round-trip."""

    def test_producer_path_is_readable_by_metricframe_file_source(self, tmp_path):
        sources = pytest.importorskip("views_reporting.sources")
        from views_pipeline_core.managers.evaluation.stage import METRICFRAME_DIR_PREFIX

        model, run_type, target = "my_model", "calibration", "lr_sb"
        mf = _mf(model_id=model, run_type=run_type)
        # Mirror EvaluationStage._save_metric_frame's layout: root/<model>/<run_type>/<prefix><target>
        frame_dir = tmp_path / model / run_type / f"{METRICFRAME_DIR_PREFIX}{target}"
        mf.save(frame_dir)

        source = sources.MetricFrameFileSource(
            root=tmp_path, run_type=run_type, target=target, primary_model=model
        )
        loaded = source.metric_frame(model)
        assert loaded is not None, "consumer could not find the producer-written frame (path drift)"
        np.testing.assert_array_equal(loaded.values, mf.values)


class TestProvenance:
    def test_generic_provenance_present(self):
        p = _mf().metadata.provenance
        assert p.model == "my_model"
        assert p.run_type == "calibration"

    def test_partition_and_level_are_constant_columns(self):
        mf = _mf()
        assert set(mf.identifiers["partition"].tolist()) == {"(1, 2)"}
        assert set(mf.identifiers["level"].tolist()) == {"pgm"}

    def test_scoring_code_version_defaulted(self):
        assert _mf().metadata.scoring_code_version  # non-empty default (installed pkg version)

    def test_omitted_provenance_defaults_to_none(self):
        # to_metric_frame leaves run_id/data_version None when the caller omits them; the
        # eval stage now supplies both (S4/#228), but the producer default is still None.
        p = _mf().metadata.provenance
        assert p.run_id is None
        assert p.data_version is None