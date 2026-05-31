"""
Falsification audit: extraction splash zone (2026-05-30).

These tests encode findings from the falsification audit of the claim
"we 100% understand the consequences and the splash zone of what we
have done — the refactor."

Each test targets a specific hard or soft falsification. Tests are
expected to FAIL until the underlying issue is fixed.

Findings:
  H-1: 5 test files import removed packages (would break in clean CI)
  H-2: No downstream consumer inventory exists
  S-1: _ViewsDataset CIC §6 stale failure mode row
  S-2: 4 orphaned cache attributes in handlers.py
  S-3: 4 new TestReconcileCountryWorker failures (signature mismatch)
  S-4: ADR-054 Consequences text stale re: torch
  S-5: ForecastReconciler exception type changed (AssertionError → ValueError)
"""
import ast
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ── H-1: Test files must not import removed packages ─────────────────────────

REMOVED_PACKAGES = {"torch", "matplotlib", "geopandas", "plotly", "seaborn",
                     "properscoring", "scipy", "markdown", "shapely"}

# Files that should NOT have unconditional imports of removed packages
H1_TEST_FILES = [
    "tests/test_modules/test_reconciliation.py",
    "tests/test_modules/test_statistics.py",
    "tests/test_modules/test_mapping_characterization.py",
    "tests/test_modules/test_visualizations.py",
    "tests/test_modules/test_report.py",
]


def _get_unconditional_imports(filepath: pathlib.Path) -> set:
    """Parse a Python file and return top-level unconditional import names."""
    source = filepath.read_text()
    tree = ast.parse(source)
    imports = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


@pytest.mark.parametrize("rel_path", H1_TEST_FILES)
def test_h1_no_removed_package_imports(rel_path):
    """H-1: Test files must not unconditionally import removed packages."""
    filepath = REPO_ROOT / rel_path
    if not filepath.exists():
        pytest.skip(f"{rel_path} does not exist")
    imports = _get_unconditional_imports(filepath)
    violations = imports & REMOVED_PACKAGES
    assert not violations, (
        f"{rel_path} unconditionally imports removed packages: {violations}"
    )


# ── H-2: Downstream consumer inventory must exist ────────────────────────────

def test_h2_downstream_consumer_inventory_exists():
    """H-2: A downstream consumer inventory document must exist with actual data."""
    candidates = [
        "reports/downstream_consumer_inventory.md",
        "documentation/downstream_consumers.md",
        "reports/views_reporting_extraction/downstream_consumers.md",
    ]
    found = None
    for c in candidates:
        p = REPO_ROOT / c
        if p.exists():
            found = p
            break

    if found is None:
        pytest.fail(
            "No downstream consumer inventory document found. "
            "extraction_pr_plans.md contains grep templates but no recorded results. "
            "A file-to-symbol mapping of which downstream repos import extracted symbols "
            "is required to understand the splash zone."
        )
    else:
        text = found.read_text()
        assert len(text) > 200, f"Inventory at {found} appears empty/stub"


# ── S-1: CIC §6 must not reference extracted method behavior ─────────────────

def test_s1_cic_no_stale_failure_modes():
    """S-1: _ViewsDataset CIC failure mode table must not reference extracted behavior."""
    cic = REPO_ROOT / "documentation/CICs/_ViewsDataset.md"
    if not cic.exists():
        pytest.skip("CIC not found")
    text = cic.read_text()
    assert "Statistics on non-prediction data" not in text, (
        "CIC §6 still contains stale failure mode 'Statistics on non-prediction data' "
        "— this error is now raised by views_reporting.statistics, not handlers.py"
    )


# ── S-2: No orphaned cache attributes in handlers.py ─────────────────────────

ORPHANED_ATTRS = [
    "_entity_metadata_cache",
    "_country_to_grids_cache",
    "reconciled_dataframe",
]


def test_s2_no_orphaned_cache_attributes():
    """S-2: handlers.py must not initialize cache attrs whose populating methods were extracted."""
    handlers = REPO_ROOT / "views_pipeline_core/data/handlers.py"
    if not handlers.exists():
        pytest.skip("handlers.py not found")
    source = handlers.read_text()
    found = [attr for attr in ORPHANED_ATTRS if f"self.{attr}" in source]
    assert not found, (
        f"handlers.py initializes orphaned cache attributes: {found}. "
        "The methods that populated these were extracted to views-reporting."
    )


# ── S-3: TestReconcileCountryWorker signature must match current API ──────────

def test_s3_reconcile_worker_signature_matches():
    """S-3: test_reconciliation.py worker tests must use current _reconcile_country_worker signature."""
    test_file = REPO_ROOT / "tests/test_modules/test_reconciliation.py"
    if not test_file.exists():
        pytest.skip("test_reconciliation.py not found")
    source = test_file.read_text()
    # Check if test actually passes by looking for the 9-element args tuple
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 9:
            assert False, (
                "test_reconciliation.py still passes a 9-tuple to _reconcile_country_worker. "
                "The signature changed during extraction — lr, max_iters, tol are no longer "
                "separate positional args."
            )


# ── S-4: ADR-054 Consequences must not claim torch is still needed ────────────

def test_s4_adr054_consequences_no_stale_torch_claim():
    """S-4: ADR-054 Consequences section must not claim torch is still a dependency."""
    adr = REPO_ROOT / "documentation/ADRs/054_visualization_and_reporting_extraction.md"
    if not adr.exists():
        pytest.skip("ADR-054 not found")
    text = adr.read_text()
    # Find the Consequences section
    consequences_match = re.search(r"## Consequences\s*\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    if not consequences_match:
        pytest.skip("No Consequences section found")
    consequences = consequences_match.group(1)
    assert "Torch remains a top-level import" not in consequences, (
        "ADR-054 Consequences section still claims 'Torch remains a top-level import' — "
        "torch was fully removed from handlers.py and pyproject.toml in PRs 8+9"
    )


# ── S-5: ForecastReconciler sample count mismatch must raise ValueError ───────

def test_s5_reconciler_sample_mismatch_raises_value_error():
    """S-5: test_statistics.py must expect ValueError (not AssertionError) for sample count mismatch."""
    test_file = REPO_ROOT / "tests/test_modules/test_statistics.py"
    if not test_file.exists():
        pytest.skip("test_statistics.py not found")
    source = test_file.read_text()
    # Find the specific test
    if "test_reconcile_forecast_probabilistic_sample_count_mismatch" not in source:
        pytest.skip("Test not found")
    # Check if it expects AssertionError (the stale expectation)
    # Extract the relevant test method
    pattern = r"def test_reconcile_forecast_probabilistic_sample_count_mismatch.*?(?=\n    def |\nclass |\Z)"
    match = re.search(pattern, source, re.DOTALL)
    if match:
        test_body = match.group(0)
        assert "AssertionError" not in test_body and "AssertError" not in test_body, (
            "test_reconcile_forecast_probabilistic_sample_count_mismatch expects "
            "AssertionError but views-reporting now raises ValueError"
        )
