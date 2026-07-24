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
