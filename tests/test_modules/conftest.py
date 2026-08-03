"""Shared fixtures for test_modules: contract-fixture canon + import-weight probe (#287).

The data lives in contract_canon.py (plain module, one canon); these fixtures
are the injection surface.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from contract_canon import CONTRACT_CANON, CONTRACT_FIXTURE_DIR, HEAVY_IMPORTS


@pytest.fixture(scope="session")
def contract_fixture_dir() -> Path:
    return CONTRACT_FIXTURE_DIR


@pytest.fixture(scope="session")
def contract_frame_dir() -> Path:
    return CONTRACT_FIXTURE_DIR / "frame"


@pytest.fixture(scope="session")
def contract_canon() -> dict:
    return CONTRACT_CANON


@pytest.fixture(scope="session")
def make_frame():
    """Factory for small synthetic FeatureFrames (shared by cache/sniffer suites)."""
    import numpy as np
    from views_frames import FeatureFrame, SpatialLevel, SpatioTemporalIndex

    def _make(months, level=SpatialLevel.PGM, n_features=1, n_units=3):
        time = np.repeat(np.asarray(list(months), dtype=np.int64), n_units)
        unit = np.tile(np.arange(1, n_units + 1, dtype=np.int64), len(list(months)))
        return FeatureFrame(
            y_features=np.ones((len(time), n_features, 1), dtype=np.float32),
            index=SpatioTemporalIndex(time=time, unit=unit, level=level),
            feature_names=[f"x{i}" for i in range(n_features)],
        )

    return _make


@pytest.fixture(scope="session")
def empty_frame():
    """A validly-constructed zero-row FeatureFrame (the poison-cache shape)."""
    import numpy as np
    from views_frames import FeatureFrame, SpatialLevel, SpatioTemporalIndex

    return FeatureFrame(
        y_features=np.zeros((0, 1, 1), dtype=np.float32),
        index=SpatioTemporalIndex(
            time=np.array([], dtype=np.int64),
            unit=np.array([], dtype=np.int64),
            level=SpatialLevel.PGM,
        ),
        feature_names=["x"],
    )


@pytest.fixture(scope="session")
def assert_module_import_light():
    """Callable: assert importing `module` pulls none of the heavy legacy deps."""

    def _check(module: str, heavy: tuple = HEAVY_IMPORTS) -> None:
        probe = (
            f"import sys; import {module}; "
            f"hits = [m for m in {heavy!r} if m in sys.modules]; "
            f"print(','.join(hits) or 'CLEAN')"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "CLEAN", (
            f"{module} transitively imports heavy modules: {out.stdout.strip()}"
        )

    return _check
