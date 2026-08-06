"""Shared canon for the vendored datafactory-contract fixture (#162/#287).

One source for the pinned fixture content and the heavy-import denylist,
consumed by conftest.py, test_datafactory_contract_conformance.py and
test_frame_cache.py. Re-vendoring the fixture updates this file once.
"""
from pathlib import Path

CONTRACT_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "feature_frame_contract"

#: Pinned content of the vendored contract fixture (upstream fixture README).
CONTRACT_CANON = {
    "time": [541, 541, 541, 542, 542, 542],
    "unit": [149426, 150146, 150866, 149426, 150146, 150866],
    "features": ["ged_sb_best", "acled_fatalities"],
}

#: Modules the import-light guard forbids (falsify P4).
HEAVY_IMPORTS = ("pandas", "viewser", "ingester3")
