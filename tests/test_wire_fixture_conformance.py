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
ROOT_HASH = "b1f3878df9ef74b25dce53a070e1711db39dfdf1c6ca3e1f5a716875ceb32f44"

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