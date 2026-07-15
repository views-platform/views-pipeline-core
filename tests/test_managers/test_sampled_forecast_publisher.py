"""#269 / ADR-013 §3 — the Hop-A Track A publish leg.

Covers the amended #269 acceptance criteria: (a) §3.4 emission assert, (b) §3.3
golden-string names, (c) §10.2 injectable provenance + byte-pinned header, (d) §2 header
content incl. the §7a wire-target mapping, (e) round-trip identity (archive → load_pf),
(f) manifest-last commit protocol + torn-run abort, (g) flag gating on the PFE.
"""
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

from views_pipeline_core.managers.ensemble.prediction_frame_ensemble import (
    PredictionFrameEnsembleManager,
)
from views_pipeline_core.managers.ensemble.sampled_forecast_publisher import (
    INTERNAL_TO_WIRE_TARGET,
    MANIFEST_NAME_TEMPLATE,
    MANIFEST_TYPE,
    SHARD_NAME_TEMPLATE,
    SHARD_TYPE,
    WIRE_CONTRACT_VERSION,
    _assert_staged_emission,
    build_header,
    header_bytes,
    publish_sampled_forecast,
    wire_target,
)
from views_pipeline_core.managers.prediction.prediction_frame_io import load_pf, save_pf


def _pf(times, n_cells=4, n_samples=8, seed=0):
    rng = np.random.default_rng(seed)
    time = np.repeat(np.asarray(times, dtype=np.int64), n_cells)
    unit = np.tile(np.arange(1, n_cells + 1, dtype=np.int64), len(times))
    return PredictionFrame(
        rng.uniform(0, 5, size=(len(time), n_samples)).astype(np.float32),
        SpatioTemporalIndex(time, unit, SpatialLevel.PGM),
    )


class _FakeDatastore:
    """Records uploads in order; copies file bytes (the real tempdir dies after publish)."""

    def __init__(self, fail_on: str | None = None):
        self.uploads = []
        self._fail_on = fail_on

    def upload_data(self, *, file, filename, loa, name, type, targets, category,
                    description=None):
        if self._fail_on and self._fail_on in filename:
            return SimpleNamespace(success=False, error="synthetic failure", data={})
        self.uploads.append(
            {"filename": filename, "name": name, "type": type, "loa": loa,
             "targets": list(targets), "category": category,
             "bytes": Path(file).read_bytes()}
        )
        return SimpleNamespace(success=True, data={"file_id": f"fid-{len(self.uploads)}"})


def _publish(store, pf, **overrides):
    kwargs = dict(
        ensemble_name="rusty_bucket",
        internal_target="lr_sb_best",
        run_id="rusty_bucket_forecasting_20260715_000000",
        level="pgm",
        reconciled=True,
        generated_at="2026-07-15T00:00:00+00:00",
        pipeline_core_version="3.0.0",
    )
    kwargs.update(overrides)
    return publish_sampled_forecast(store, pf, **kwargs)


# ------------------------------------------------------------------ (b) golden strings


def test_shard_name_golden_string():
    assert (
        SHARD_NAME_TEMPLATE.format(run_id="r1", target="lr_ged_sb", time_id=543)
        == "r1__lr_ged_sb__m000543.tap.zip"
    )


def test_manifest_name_golden_string():
    assert (
        MANIFEST_NAME_TEMPLATE.format(run_id="r1", target="lr_ged_sb")
        == "r1__lr_ged_sb__manifest.json"
    )


# ------------------------------------------------------------------ (d) §7a wire mapping


def test_wire_target_mapping():
    assert wire_target("lr_sb_best") == "lr_ged_sb"
    assert wire_target("lr_ns_best") == "lr_ged_ns"
    assert wire_target("lr_os_best") == "lr_ged_os"
    assert set(INTERNAL_TO_WIRE_TARGET.values()) == {"lr_ged_sb", "lr_ged_ns", "lr_ged_os"}


def test_wire_target_unmapped_fails_loud():
    with pytest.raises(ValueError, match="wire-name mapping"):
        wire_target("synth_target")


# ------------------------------------------------------------------ (c) byte-pinned header


def test_header_bytes_are_stable_given_injected_provenance():
    header = build_header(
        sample_count=8, spatial_level="pgm", target_wire="lr_ged_sb", time_id=543,
        run_id="r1", generated_at="2026-07-15T00:00:00+00:00", ensemble_name="rusty_bucket",
        reconciled=True, shard_index=0, shard_count=2, pipeline_core_version="3.0.0",
    )
    expected = (
        "{\n"
        '  "contract_version": "' + WIRE_CONTRACT_VERSION + '",\n'
        '  "frame_type": "prediction",\n'
        '  "representation": "samples",\n'
        '  "sample_count": 8,\n'
        '  "dtype": "float32",\n'
        '  "spatial_level": "pgm",\n'
        '  "target": "lr_ged_sb",\n'
        '  "time_id": 543,\n'
        '  "run_id": "r1",\n'
        '  "generated_at": "2026-07-15T00:00:00+00:00",\n'
        '  "id_semantics": {\n'
        '    "time": "views_month_id",\n'
        '    "unit": "priogrid_id"\n'
        "  },\n"
        '  "provenance": {\n'
        '    "ensemble": "rusty_bucket",\n'
        '    "pipeline_core_version": "3.0.0",\n'
        '    "reconciled": true\n'
        "  },\n"
        '  "sharding": {\n'
        '    "scheme": "per_month",\n'
        '    "index": 0,\n'
        '    "count": 2\n'
        "  }\n"
        "}\n"
    ).encode("utf-8")
    assert header_bytes(header) == expected


# ------------------------------------------------------------------ (e)+(f) publish flow


def test_publish_uploads_shards_then_manifest_last():
    store = _FakeDatastore()
    pf = _pf(times=[543, 544], n_samples=8)
    manifest = _publish(store, pf)

    assert len(store.uploads) == 3  # 2 shards + 1 manifest
    assert [u["type"] for u in store.uploads] == [SHARD_TYPE, SHARD_TYPE, MANIFEST_TYPE]
    assert store.uploads[-1]["filename"].endswith("__manifest.json")
    # store-document fields (§3.1)
    for u in store.uploads:
        assert u["category"] == "forecast"
        assert u["loa"] == "pgm"
        assert u["targets"] == ["lr_ged_sb"]
    # manifest content (§3.2)
    assert manifest["contract_version"] == WIRE_CONTRACT_VERSION
    assert manifest["expected_months"] == [543, 544]
    assert manifest["expected_cell_count"] == 4
    assert manifest["sidecar_sha256"] is None
    assert [s["time_id"] for s in manifest["shards"]] == [543, 544]
    assert all(s["file_id"] for s in manifest["shards"])


def test_round_trip_identity_archive_to_load_pf(tmp_path):
    store = _FakeDatastore()
    pf = _pf(times=[543, 544], n_samples=8, seed=7)
    _publish(store, pf)

    shard = store.uploads[0]
    zpath = tmp_path / shard["filename"]
    zpath.write_bytes(shard["bytes"])
    out = tmp_path / "unpacked"
    with zipfile.ZipFile(zpath) as zf:
        assert sorted(zf.namelist()) == ["identifiers.npz", "metadata.json", "y_pred.npy"]
        zf.extractall(out)

    loaded = load_pf(out, level="pgm")
    times = np.asarray(pf.index.time)
    month_pf = pf.select(times == 543)
    np.testing.assert_array_equal(loaded.values, month_pf.values)
    np.testing.assert_array_equal(
        np.asarray(loaded.index.unit), np.asarray(month_pf.index.unit)
    )
    header = json.loads((out / "metadata.json").read_text())
    assert header["sample_count"] == 8
    assert header["sharding"] == {"scheme": "per_month", "index": 0, "count": 2}


def test_shard_failure_withholds_manifest():
    store = _FakeDatastore(fail_on="m000544")  # second shard fails
    pf = _pf(times=[543, 544])
    with pytest.raises(RuntimeError, match="manifest is withheld"):
        _publish(store, pf)
    assert all(u["type"] == SHARD_TYPE for u in store.uploads)  # no manifest committed


def test_ragged_months_fail_loud():
    # month 544 carries fewer cells than 543 → malformed run, no publish
    pf = _pf(times=[543, 544])
    times = np.asarray(pf.index.time)
    keep = ~((times == 544) & (np.asarray(pf.index.unit) == 1))
    ragged = pf.select(keep)
    with pytest.raises(ValueError, match="differing cell counts"):
        _publish(_FakeDatastore(), ragged)


# ------------------------------------------------------------------ (a) §3.4 assert


def test_emission_assert_trips_on_corrupted_stage(tmp_path):
    pf = _pf(times=[543])
    times = np.asarray(pf.index.time)
    month_pf = pf.select(times == 543)
    save_pf(month_pf, tmp_path)
    _assert_staged_emission(tmp_path, month_pf, agg_sample_count=8)  # clean passes

    np.save(tmp_path / "y_pred.npy", np.zeros((4, 8), dtype=np.float64))  # wrong dtype
    with pytest.raises(ValueError, match="dtype"):
        _assert_staged_emission(tmp_path, month_pf, agg_sample_count=8)

    np.save(tmp_path / "y_pred.npy", np.zeros((4, 5), dtype=np.float32))  # wrong S
    with pytest.raises(ValueError, match="sample_count"):
        _assert_staged_emission(tmp_path, month_pf, agg_sample_count=8)


# ------------------------------------------------------------------ (g) PFE wiring


def _bare_manager(use_store: bool) -> PredictionFrameEnsembleManager:
    m = object.__new__(PredictionFrameEnsembleManager)
    m._use_prediction_store = use_store
    m._datastore = _FakeDatastore()
    return m


def test_pfe_publish_method_routes_context(monkeypatch):
    calls = {}

    def fake_publish(datastore, agg_pf, **kwargs):
        calls.update(kwargs)
        return {}

    import views_pipeline_core.managers.ensemble.sampled_forecast_publisher as sfp
    monkeypatch.setattr(sfp, "publish_sampled_forecast", fake_publish)

    m = _bare_manager(use_store=True)
    ctx = SimpleNamespace(
        configs={"name": "rusty_bucket", "level": "pgm"},
        run_type="forecasting",
        timestamp="20260715_000000",
        reconciliation="pgm_cm",
    )
    m._publish_sampled_forecast(_pf(times=[543]), "lr_sb_best", ctx)

    assert calls["ensemble_name"] == "rusty_bucket"
    assert calls["internal_target"] == "lr_sb_best"
    assert calls["run_id"] == "rusty_bucket_forecasting_20260715_000000"
    assert calls["level"] == "pgm"
    assert calls["reconciled"] is True


def test_pfe_datastore_starts_unbuilt():
    # The leg is opt-in: no datastore is constructed at init (env credentials are only
    # required when use_prediction_store is actually exercised).
    import inspect
    src = inspect.getsource(PredictionFrameEnsembleManager.__init__)
    assert "_datastore = None" in src
