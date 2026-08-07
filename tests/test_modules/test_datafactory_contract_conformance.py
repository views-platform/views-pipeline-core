"""#162 — consumer-side conformance to the datafactory contract (their ADR-050).

The vendored fixture (tests/fixtures/feature_frame_contract/) is views-datafactory's
committed, REAL ``FeatureFrame.save()`` output plus its language-neutral
``contract.json``. These tests are the pipeline-core half of the cross-repo contract:

- integrity: the vendored bytes match the pinned ``fixture_digest`` (tamper guard);
- layout conformance: our installed views-frames loads the real datafactory save()
  output and sees exactly the documented content (the C-30 contract test);
- vocabulary containment: every ``output_format`` string this repo sends is in the
  contract's vocabulary (kills the C-62 literal duplication silently drifting);
- freshness: where datafactory IS importable (dev envs; CI does not install it),
  the vendored contract must equal the installed package's exports — an upstream
  contract bump alarms here before it can bite at runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from views_frames import FeatureFrame

from views_pipeline_core.modules.dataloaders.dataloaders import _LOA_TO_OUTPUT_FORMAT

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "feature_frame_contract"
FRAME_DIR = FIXTURE_DIR / "frame"
CONTRACT_PATH = FIXTURE_DIR / "contract.json"

# The major contract line this repo is written against (upstream stability promise:
# member meanings never change; rename/removal = MAJOR, addition = MINOR).
SUPPORTED_CONTRACT_MAJOR = "1"

# Pinned fixture content: the shared canon lives in contract_canon.py (one
# source for this suite and test_frame_cache.py); shape/values stay local here.
from contract_canon import CONTRACT_CANON  # noqa: E402

EXPECTED_TIME = CONTRACT_CANON["time"]
EXPECTED_UNIT = CONTRACT_CANON["unit"]
EXPECTED_FEATURES = CONTRACT_CANON["features"]
EXPECTED_SHAPE = (6, 2, 1)
EXPECTED_VALUES_FLAT = [
    1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0,
]

# Digest scheme replicated from upstream (datafactory_provenance.compute_file_digest +
# scripts/generate_contract_fixture.py::fixture_digest) so CI needs no datafactory
# install: per-file sha256 hex truncated to 16, composed as "name:digest" lines,
# sha256 of the joined lines, truncated to 16.
_DIGEST_TRUNCATE = 16
_CHUNK = 65536


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()[:_DIGEST_TRUNCATE]


def _fixture_digest(frame_dir: Path) -> str:
    parts = [f"{p.name}:{_file_digest(p)}" for p in sorted(frame_dir.iterdir())]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:_DIGEST_TRUNCATE]


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


class TestVendoredFixtureIntegrity:
    """The vendored copy is byte-authentic against its own pinned digest."""

    def test_fixture_digest_matches_contract(self):
        assert _fixture_digest(FRAME_DIR) == _contract()["fixture_digest"], (
            "Vendored frame/ bytes do not match contract.json.fixture_digest. "
            "Either the vendored copy is corrupted/partial (re-vendor per the fixture "
            "README) or upstream changed the layout without a contract bump."
        )

    def test_layout_files_match_contract(self):
        assert sorted(p.name for p in FRAME_DIR.iterdir()) == sorted(
            _contract()["layout_files"]
        )


class TestLayoutConformance:
    """Our installed views-frames reads datafactory's real save() output (C-30)."""

    def test_views_frames_round_trips_the_datafactory_fixture(self):
        ff = FeatureFrame.load(FRAME_DIR)
        assert ff.values.shape == EXPECTED_SHAPE
        assert ff.values.dtype == np.float32, (
            "contract.json declares float32 as a contractual property"
        )
        np.testing.assert_array_equal(np.asarray(ff.index.time), EXPECTED_TIME)
        np.testing.assert_array_equal(np.asarray(ff.index.unit), EXPECTED_UNIT)
        assert list(ff.feature_names) == EXPECTED_FEATURES
        np.testing.assert_array_equal(ff.values.ravel(), EXPECTED_VALUES_FLAT)


class TestVocabularyConformance:
    """Every format string this repo sends is in the contract vocabulary (C-62)."""

    def test_loa_output_formats_are_within_contract(self):
        contract_formats = set(_contract()["output_formats"])
        ours = set(_LOA_TO_OUTPUT_FORMAT.values())
        assert ours <= contract_formats, (
            f"_LOA_TO_OUTPUT_FORMAT sends {sorted(ours - contract_formats)} which the "
            f"datafactory contract does not define ({sorted(contract_formats)}). "
            f"Reconcile with the vendored contract.json (and upstream OutputFormat)."
        )

    def test_contract_major_version_is_supported(self):
        major = _contract()["contract_version"].split(".")[0]
        assert major == SUPPORTED_CONTRACT_MAJOR, (
            f"Vendored contract is major version {major}; this repo is written "
            f"against major {SUPPORTED_CONTRACT_MAJOR}. Review upstream's MAJOR "
            f"changes (ADR-050 record) before re-pinning."
        )


class TestInstalledContractFreshness:
    """Vendored contract == installed datafactory exports (runs only where installed)."""

    def test_vendored_contract_matches_installed_package(self):
        dq = pytest.importorskip("datafactory_query")
        for export in ("OutputFormat", "CONTRACT_VERSION", "is_valid_output_format"):
            assert hasattr(dq, export), (
                f"Installed datafactory_query lacks the ADR-050 export '{export}' — "
                f"upgrade views-datafactory to >= 1.8.0."
            )
        contract = _contract()
        assert sorted(str(m) for m in dq.OutputFormat) == sorted(
            contract["output_formats"]
        ), "Installed OutputFormat vocabulary differs from the vendored contract — re-vendor."
        assert dq.CONTRACT_VERSION == contract["contract_version"], (
            f"Installed CONTRACT_VERSION={dq.CONTRACT_VERSION} != vendored "
            f"{contract['contract_version']} — upstream bumped the contract; review "
            f"the change and re-vendor per the fixture README."
        )