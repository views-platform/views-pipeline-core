"""ADR-013 §10 golden-fixture conformance — the producer's side (#269 follow-up).

Two layers, per §10.1:
1. **Vendored-bytes integrity** — our copy of the canonical fixture (vendored from
   views-postprocessing `tests/fixtures/wire_contract/`) is bit-identical to the canon:
   every per-file hash matches `SHA256SUMS`, and the ROOT hash (SHA-256 of `SHA256SUMS`
   itself) equals the pinned constant. On mismatch: re-vendor from views-postprocessing —
   the hash, not the bytes, is the cross-repo contract.
2. **Producer parity** — `publish_sampled_forecast`, given the fixture's declared inputs
   (§10.2 injected literals), emits a Track-A shard **byte-identical** to the canonical
   `fixture_run_0__lr_ged_sb__m000543.tap.zip`, and a Hop-A manifest whose overlapping
   fields match the canonical manifest (ours carries additive keys — §2.1 open schema).
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from views_pipeline_core.managers.ensemble.sampled_forecast_publisher import (
    publish_sampled_forecast,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "wire_contract"

#: §10.1 — THE pinned root hash (SHA-256 of SHA256SUMS), from the fixture README.
ROOT_HASH = "9658a6484cc9d975412e52624d52f328985f14cf58e3fc9fbdf3e64ab5a0564b"

SHARD_A = FIX / "fixture_run_0__lr_ged_sb__m000543.tap.zip"
MANIFEST_A = FIX / "fixture_run_0__lr_ged_sb__manifest.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------- 1. vendored integrity


def test_root_hash_is_canonical():
    assert _sha256((FIX / "SHA256SUMS").read_bytes()) == ROOT_HASH, (
        "Vendored SHA256SUMS differs from the pinned root hash — the contract fixture "
        "changed upstream. Re-vendor from views-postprocessing tests/fixtures/wire_contract/ "
        "and treat the change as a contract change (ADR-013 §10)."
    )


def test_every_vendored_file_matches_its_hash():
    for line in (FIX / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert _sha256((FIX / name).read_bytes()) == digest, (
            f"Vendored fixture drift: {name} — re-vendor from views-postprocessing."
        )


# ----------------------------------------------------------------- 2. producer parity


def _fixture_inputs():
    """The §10.2 declared fixture identity — mirrors views-postprocessing
    scripts/build_wire_fixture.py exactly."""
    gids = np.array([100001, 100002, 100003, 100004, 100005, 100006], dtype=np.int64)
    rng = np.random.default_rng(1305)
    values = rng.gamma(2.0, size=(6, 4)).astype(np.float32)
    values[0] = 0.0  # row 0 draw-degenerate on purpose (§6 per-row-zeros-legal pin)
    time = np.full(6, 543, dtype=np.int64)
    return values, time, gids


class _CapturingDatastore:
    def __init__(self):
        self.uploads = []

    def upload_data(self, *, file, filename, loa, name, type, targets, category,
                    description=None):
        self.uploads.append({"filename": filename, "type": type,
                             "bytes": Path(file).read_bytes()})
        return SimpleNamespace(success=True, data={"file_id": f"fid-{len(self.uploads)}"})


@pytest.fixture(scope="module")
def published():
    from views_frames import PredictionFrame, SpatialLevel, SpatioTemporalIndex

    values, time, gids = _fixture_inputs()
    pf = PredictionFrame(values, SpatioTemporalIndex(time, gids, SpatialLevel.PGM))
    store = _CapturingDatastore()
    manifest = publish_sampled_forecast(
        store,
        pf,
        ensemble_name="fixture_ensemble",
        internal_target="lr_sb_best",  # §7a maps to the fixture's wire name lr_ged_sb
        run_id="fixture_run_0",
        level="pgm",
        reconciled=False,
        generated_at="2026-07-15T00:00:00Z",
        pipeline_core_version="0.0.0-fixture",
    )
    return store, manifest


def test_producer_emits_the_canonical_shard_bytes(published):
    store, _ = published
    shard = store.uploads[0]
    assert shard["filename"] == SHARD_A.name  # §3.3 template identity
    assert shard["bytes"] == SHARD_A.read_bytes(), (
        "Producer no longer emits the golden fixture's canonical shard bytes — either the "
        "publisher's wire serialization drifted (a contract break: ADR-013 §10) or the "
        "fixture was legitimately regenerated upstream (then re-vendor + update ROOT_HASH)."
    )


def test_producer_manifest_matches_canonical_overlap(published):
    _, ours = published
    canon = json.loads(MANIFEST_A.read_text())
    # exact-match overlapping fields (§3.2); ours may carry ADDITIVE keys (§2.1 open schema)
    assert ours["contract_version"] == canon["contract_version"]
    assert ours["run_id"] == canon["run_id"]
    assert ours["target"] == canon["target"]
    assert ours["expected_months"] == canon["expected_months"]
    assert ours["expected_cell_count"] == canon["expected_cell_count"]
    assert ours["sidecar_sha256"] is None and canon["sidecar_sha256"] is None  # Erratum E1
    # shard entries agree on name + content hash (file_id/time_id are our additive keys)
    assert ours["shards"][0]["name"] == canon["shards"][0]["name"]
    assert ours["shards"][0]["sha256"] == canon["shards"][0]["sha256"]


# ----------------------------------------------------------------- 3. upstream drift
#
# Added 2026-08-11, after the vendored copy was found **eight weeks stale** and no test
# could see it.
#
# `ROOT_HASH` is the SHA-256 of *our own* `SHA256SUMS`. It detects someone editing our
# copy. Its failure message says "the contract fixture changed upstream" — which it is
# structurally incapable of detecting, because both sides of that comparison come from
# this repo.
#
# What had actually happened: views-postprocessing re-baselined the fixture to the
# production toolchain in their `e47db5d`, labelled "§10 contract-change event, maintainer
# Option A" — pyarrow 16.1.0 instead of 23.x, which changes parquet bytes (their C-72).
# Five of the seven files differed. Every test here stayed green, because the three that
# matter to the producer (`.tap.zip`, the shard manifest) were unaffected and the rest
# were only ever checked against our own SHA256SUMS.
#
# A self-consistent copy is not a conformed one.


_UPSTREAM = (
    Path(__file__).resolve().parents[2]
    / "views-postprocessing"
    / "tests"
    / "fixtures"
    / "wire_contract"
)


def test_the_vendored_fixture_is_byte_identical_to_upstream():
    """Three outcomes, and the third is the one that was missing.

    - sibling checkout absent → **skip**. CI has no views-postprocessing beside it, and
      no mechanism here can see another repo's files from a runner.
    - present and identical → **pass**.
    - present and differing → **fail**, naming the files.

    Enforced on a developer's machine rather than in CI, like
    `test_f4_deleting_sessionauth_must_not_orphan_the_seam_registry`. That is a real
    limit, not a defect to fix: whoever has both repos checked out is whoever is in a
    position to re-vendor.
    """
    if not _UPSTREAM.is_dir():  # pragma: no cover - sibling checkout may be absent
        pytest.skip("views-postprocessing not checked out beside this repo")

    ours = {p.name for p in FIX.iterdir() if p.is_file()}
    theirs = {p.name for p in _UPSTREAM.iterdir() if p.is_file()}
    assert ours, "no vendored fixture files found — this check would pass vacuously"

    assert ours == theirs, (
        f"the fixture's file SET differs from upstream. Only here: {sorted(ours - theirs)}. "
        f"Only upstream: {sorted(theirs - ours)}. Re-vendor the whole directory."
    )

    drifted = [
        name
        for name in sorted(ours)
        if (FIX / name).read_bytes() != (_UPSTREAM / name).read_bytes()
    ]
    assert not drifted, (
        f"vendored fixture is stale against views-postprocessing: {drifted}. A change to "
        f"this fixture is a change to the contract (ADR-013 §10) — re-vendor the whole "
        f"directory and update ROOT_HASH, and treat it as a contract change rather than a "
        f"refresh. Do NOT regenerate locally to make this pass: parquet bytes vary with "
        f"pyarrow version (their C-72), so regenerating substitutes your toolchain's "
        f"output for their canon."
    )


#: What pipeline-core actually produces, of the seven vendored files. Hop-B only.
_WE_PRODUCE = {
    "fixture_run_0__lr_ged_sb__m000543.tap.zip",
    "fixture_run_0__lr_ged_sb__manifest.json",
}


def test_it_is_stated_which_vendored_files_the_producer_is_checked_against():
    """The limit, as an assertion rather than an omission.

    Only two of the seven vendored files are compared against anything this repo emits.
    The rest — the Hop-A run manifest, the arrow parquet, the sidecar — are
    views-postprocessing's own output; pipeline-core does not produce them, so there is
    nothing here to hold them to. They are vendored as context.

    That is a good reason, and it was nowhere written. A vendored file that no producer
    test touches looks like coverage from the outside, and this file already had five
    such files silently stale for eight weeks. Now the drift check above covers all
    seven, and this states why parity covers two.
    """
    present = {p.name for p in FIX.iterdir() if p.is_file()}
    assert _WE_PRODUCE <= present, f"missing: {sorted(_WE_PRODUCE - present)}"
    assert SHARD_A.name in _WE_PRODUCE and MANIFEST_A.name in _WE_PRODUCE, (
        "the parity tests compare files not listed in _WE_PRODUCE — update the list and "
        "this docstring together."
    )
