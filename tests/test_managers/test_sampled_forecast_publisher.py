"""#269 / ADR-013 §3 — the Hop-A Track A publish leg.

Covers the amended #269 acceptance criteria: (a) §3.4 emission assert, (b) §3.3
golden-string names, (c) §10.2 injectable provenance + byte-pinned header, (d) §2 header
content incl. the §7a wire-target mapping, (e) round-trip identity (archive → load_pf),
(f) manifest-last commit protocol + torn-run abort, (g) flag gating on the PFE.
"""
import json
import zipfile
from importlib.metadata import PackageNotFoundError
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


def test_wire_names_are_unique_so_two_targets_cannot_collide_on_one_column():
    """The mapping's VALUES must stay injective, and a set comparison cannot see that.

    `test_wire_target_mapping`'s `set(...values()) == {three}` stays GREEN if a fourth
    entry duplicates an existing wire name — a set collapses the duplicate. Since #536
    the keys are also the publish allowlist, so a colliding fourth entry (a copy-paste
    typo, or `{"cls_sb_best": "lr_ged_sb"}`) admits BOTH internal targets and uploads two
    shards and two manifests under byte-identical names: classification probabilities on
    the wire under the fatality target's name, which is the outcome #536's reasoning
    assumes deriving from the mapping prevents.

    The §7a extension procedure is exactly where this typo happens, and the run-time
    error message steers an operator toward it, so it is pinned here rather than trusted.
    """
    values = list(INTERNAL_TO_WIRE_TARGET.values())
    assert len(values) == len(set(values)), (
        f"two internal targets map to the same wire name: {sorted(values)} — they would "
        f"publish over each other under identical filenames"
    )


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
        "}"  # no trailing newline — §10 fixture canon
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


# ---------------------------------------------------------------------------
# #279 — the provenance field's authority, and what it must refuse to claim
# ---------------------------------------------------------------------------


class TestProvenanceVersionAuthority:
    """`pipeline_core_version` must never carry a number it did not establish.

    ADR-013 §2.2 told consumers to disregard this field until a real release existed.
    3.0.0 shipped on 2026-08-03, so for a consumer installing from PyPI the field is now
    authoritative — the metadata was written by the release that built the wheel.

    An editable install is the case that made this worth a test rather than a docstring
    edit. Its `.dist-info` records the version current when `pip install -e` was last run
    and never tracks the source again. Measured here on 2026-08-04: metadata `2.3.0`,
    `pyproject.toml` `3.0.0`. A developer run would have stamped `2.3.0` into a published
    artifact's provenance — a value the system never established, published as one it
    measured, which is the whole Cluster J shape.
    """

    def _version_module(self):
        from views_pipeline_core.managers.ensemble import sampled_forecast_publisher

        return sampled_forecast_publisher

    def test_an_editable_install_reports_unknown_not_its_stale_metadata(
        self, monkeypatch
    ):
        module = self._version_module()

        class _Editable:
            version = "2.3.0"  # what this repo's own dev env actually reported

            def read_text(self, name):
                assert name == "direct_url.json"
                return json.dumps(
                    {"dir_info": {"editable": True}, "url": "file:///somewhere"}
                )

        monkeypatch.setattr(
            module, "distributions_metadata", lambda _name: _Editable(), raising=True
        )
        assert module._pipeline_core_version() == "unknown", (
            "an editable install reported its stale metadata as the producing version"
        )

    def test_a_released_install_reports_its_real_version(self, monkeypatch):
        module = self._version_module()

        class _Released:
            version = "3.0.0"

            def read_text(self, name):
                return None  # a wheel from PyPI carries no direct_url.json

        monkeypatch.setattr(
            module, "distributions_metadata", lambda _name: _Released(), raising=True
        )
        assert module._pipeline_core_version() == "3.0.0"

    def test_a_non_editable_direct_url_still_reports_its_version(self, monkeypatch):
        """`pip install .` from a local path writes direct_url.json with editable false.

        That is a real build of a real version, so it is not the case this guard refuses.
        """
        module = self._version_module()

        class _LocalBuild:
            version = "3.0.0"

            def read_text(self, name):
                return json.dumps({"dir_info": {"editable": False}, "url": "file:///x"})

        monkeypatch.setattr(
            module, "distributions_metadata", lambda _name: _LocalBuild(), raising=True
        )
        assert module._pipeline_core_version() == "3.0.0"

    def test_unreadable_or_malformed_metadata_reports_unknown(self, monkeypatch):
        """Refusing to conclude is not the same as concluding a version."""
        module = self._version_module()

        class _Malformed:
            version = "3.0.0"

            def read_text(self, name):
                return "{not json"

        monkeypatch.setattr(
            module, "distributions_metadata", lambda _name: _Malformed(), raising=True
        )
        assert module._pipeline_core_version() == "unknown"

    def test_no_distribution_at_all_reports_unknown(self, monkeypatch):
        module = self._version_module()

        def _absent(_name):
            raise PackageNotFoundError("views_pipeline_core")

        monkeypatch.setattr(
            module, "distributions_metadata", _absent, raising=True
        )
        assert module._pipeline_core_version() == "unknown"


# ---------------------------------------------------------------------------
# #536 — the multi-target publish loop. Every test above publishes ONE
# (run, target); the loop over `ctx.targets` was covered nowhere, which is how a
# six-target ensemble came to commit three manifests to the partner-visible store
# and then raise on the fourth.
# ---------------------------------------------------------------------------

#: `rusty_bucket`'s declared set since PR #422 made `ctx.targets` = combined_targets:
#: three regression targets the wire serves, three classification targets it does not.
#: Deliberately classification-FIRST. With the regression names at the front, a
#: positional `[:3]` mutant publishes exactly the right three and every count-based
#: assertion here stays green — the "luck of ordering" the fix exists to rule out.
_SIX_TARGETS = [
    "by_sb_best", "by_ns_best", "by_os_best",
    "lr_sb_best", "lr_ns_best", "lr_os_best",
]

_WIRE_MANIFESTS = {
    "rusty_bucket_forecasting_20260715_000000__lr_ged_sb__manifest.json",
    "rusty_bucket_forecasting_20260715_000000__lr_ged_ns__manifest.json",
    "rusty_bucket_forecasting_20260715_000000__lr_ged_os__manifest.json",
}


def _forecast_manager(tmp_path, targets, monkeypatch, use_store=True):
    """A PFE manager wired to run `_forecast_ensemble` against an in-memory store.

    `save_pf` is REAL: the fix's central mitigation is that withheld targets are still
    saved locally, and stubbing the saver left that claim with no guard anywhere (a
    mutant that skipped `save_pf` for withheld targets passed all 2935 tests).
    `_forecast_model_artifact` keeps the real `(model_name, ctx)` parameter order, and
    `_build_datastore` is left alone so the tests still see regressions in it.
    """
    m = object.__new__(PredictionFrameEnsembleManager)
    m._use_prediction_store = use_store
    m._datastore = _FakeDatastore()
    m._ensemble_path = SimpleNamespace(data_generated=tmp_path / "generated")
    m._forecast_model_artifact = lambda model_name, ctx: {
        t: _pf(times=[543]) for t in targets
    }

    ctx = SimpleNamespace(
        configs={"name": "rusty_bucket", "level": "pgm"},
        models=["purple_alien"],
        targets=list(targets),
        aggregation="concat",
        reconciliation=None,
        run_type="forecasting",
        timestamp="20260715_000000",
        expected_samples_per_model=None,
    )
    return m, ctx


def _manifests_committed(store):
    return {u["filename"] for u in store.uploads if u["filename"].endswith("manifest.json")}


def _saved_target_dirs(tmp_path):
    root = tmp_path / "generated" / "predictions_forecasting_20260715_000000"
    return {d.name for d in root.iterdir()} if root.exists() else set()


def test_six_target_forecast_publishes_only_the_three_wire_targets(tmp_path, monkeypatch):
    """#536. The store is offered only the targets the wire serves — BY NAME.

    Before the fix this raised `No wire-name mapping for internal target 'by_sb_best'`
    with three manifests ALREADY committed to the partner-visible store; nothing is
    rolled back, by design (ADR-013 §3.2 is manifest-last, not two-phase).

    Asserted by name and never by count: FAO resolves a served name by tokenising, so
    `pred_lr_ged_sb` and `pred_cls_ged_sb` both resolve to the stem `sb` with nothing
    raising. A count assertion over a classification-first roster passes while a
    positional filter publishes the wrong three.
    """
    m, ctx = _forecast_manager(tmp_path, _SIX_TARGETS, monkeypatch)

    forecasts = m._forecast_ensemble(ctx)

    # The filter is on PUBLISHING: everything is still aggregated, returned and saved.
    assert sorted(forecasts) == sorted(_SIX_TARGETS)
    assert _saved_target_dirs(tmp_path) == set(_SIX_TARGETS)

    assert _manifests_committed(m._datastore) == _WIRE_MANIFESTS


def test_nothing_is_published_when_the_flag_is_off(tmp_path, monkeypatch):
    """The `--prediction_store` gate, asserted at the irreversible act.

    Every other test here runs with the flag ON, so nothing pinned that a plain forecast
    publishes nothing — and the only thing catching a hoisted `publishable` computation
    was an accident of another module's fixture using an unmapped target name.
    """
    m, ctx = _forecast_manager(tmp_path, _SIX_TARGETS, monkeypatch, use_store=False)

    m._forecast_ensemble(ctx)

    assert m._datastore.uploads == []
    assert _saved_target_dirs(tmp_path) == set(_SIX_TARGETS)


def test_a_target_outside_the_wire_vocabulary_is_never_offered_to_the_store(
    tmp_path, monkeypatch,
):
    """The filter must be DERIVED from the vocabulary, not restated as a shape.

    `lr_future_best` is the discriminator: same `lr_` prefix as the three served
    targets, not in the mapping. A derived filter withholds it; `startswith("lr_")`, a
    hand-copied list or a regex publishes it. That matters beyond tidiness — a wire
    target is added by adding a mapping entry, so a shape-keyed filter would publish a
    future `lr_*` target the wire does not serve: #536 again, one release later.

    Asserted as "the committed set is exactly the wire set", not as "no filename
    contains 'future'". Filenames are built from `wire_target()`, which can only return
    a served name or raise — so a substring assertion can never fire, and under the
    mutation it is written for the run raises inside `wire_target` with `lr_sb_best`
    already committed, never reaching the assertion at all.
    """
    m, ctx = _forecast_manager(
        tmp_path, _SIX_TARGETS + ["lr_future_best", "wildcard_target_best"], monkeypatch
    )

    m._forecast_ensemble(ctx)

    assert _manifests_committed(m._datastore) == _WIRE_MANIFESTS


def test_publishing_is_refused_when_a_served_column_has_no_target(tmp_path, monkeypatch):
    """A SHORT delivery is refused, not just an empty one — and nothing reaches the store.

    Checking only "publishable is empty" caught the all-miss case and let the realistic
    migration shape through: a roster part way through a rename supplies two of three,
    publishes a short set and exits 0. views-models#320 in partial dress.

    The live case: the mapping is keyed on INTERNAL names, and views-datafactory serves
    `ged_sb_best`/`ged_ns_best`/`ged_os_best` with no `lr_` prefix.
    """
    m, ctx = _forecast_manager(
        tmp_path, ["ged_sb_best", "lr_ns_best", "lr_os_best"], monkeypatch
    )

    with pytest.raises(ValueError, match=r"supplies no target for 1 of the 3"):
        m._forecast_ensemble(ctx)

    assert m._datastore.uploads == [], "nothing may reach the store on this path"


def test_publishing_is_refused_when_the_vocabulary_maps_none_of_the_targets(
    tmp_path, monkeypatch,
):
    """The all-miss case — a roster fully renamed while the mapping has not moved."""
    m, ctx = _forecast_manager(
        tmp_path, ["ged_sb_best", "ged_ns_best", "ged_os_best"], monkeypatch
    )

    with pytest.raises(ValueError) as info:
        m._forecast_ensemble(ctx)

    message = str(info.value)
    assert "3 of the 3" in message, message
    # The remedy must name the file that holds the mapping, and must not present the
    # mapping's KEYS as the served wire names — an operator following that would add
    # `"ged_sb_best": "ged_sb_best"` and publish an internal name onto the partner wire.
    assert "sampled_forecast_publisher.py" in message, message
    assert "lr_ged_sb" in message, message
    assert m._datastore.uploads == []


def test_the_filter_follows_the_mapping_when_targets_are_renamed(tmp_path, monkeypatch):
    """Rename-readiness: moving off the `lr_` prefix is a mapping entry, not a code change.

    The converse of the `lr_future_best` discriminator: there, an `lr_`-shaped unmapped
    name must be withheld; here, a mapped name that looks nothing like the current three
    must publish.
    """
    import views_pipeline_core.managers.ensemble.sampled_forecast_publisher as sfp

    monkeypatch.setattr(sfp, "INTERNAL_TO_WIRE_TARGET", {"ged_sb_best": "lr_ged_sb"})
    m, ctx = _forecast_manager(tmp_path, ["ged_sb_best", "by_sb_best"], monkeypatch)

    m._forecast_ensemble(ctx)

    assert _manifests_committed(m._datastore) == {
        "rusty_bucket_forecasting_20260715_000000__lr_ged_sb__manifest.json"
    }
    assert _saved_target_dirs(tmp_path) == {"ged_sb_best", "by_sb_best"}
