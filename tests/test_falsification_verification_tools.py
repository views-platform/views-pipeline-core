"""
Falsification audit 3: verification tool sufficiency (2026-05-31).

Attacks the claim: "We have the tools to 100% verify if this
refactor/export has been done with no regression."

Findings:
  V-2: No integration test exercises a full shim→views-reporting→pipeline-core path
  V-3: No downstream consumer verification exists
  V-4: No behavioral equivalence (pre/post) test exists
  V-5: importorskip guards never exercised in minimal environment
  V-6: Reconciliation parallelism entirely mocked — pickling untested
"""
import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ── V-2: At least one integration test must traverse the shim boundary ────────

def test_v2_integration_test_traverses_shim():
    """V-2: At least one test must call production code through a shim without mocking the shim layer."""
    test_dir = REPO_ROOT / "tests"
    shim_modules = [
        "views_pipeline_core.modules.reconciliation",
        "views_pipeline_core.modules.statistics",
        "views_pipeline_core.modules.visualizations",
        "views_pipeline_core.modules.mapping",
        "views_pipeline_core.modules.reports",
        "views_pipeline_core.templates.reports",
    ]

    found_unpatched_integration = False

    for test_file in test_dir.rglob("*.py"):
        if test_file.name.startswith("test_falsification"):
            continue
        source = test_file.read_text()
        # Look for tests that import from shim modules AND don't patch them
        for mod in shim_modules:
            if f"from {mod}" in source:
                # Check if ALL uses are behind @patch decorators
                patch_count = source.count(f'@patch("{mod}')
                patch_count += source.count(f"@patch('{mod}")
                import_count = source.count(f"from {mod}")
                # If there are imports but no patches, it might be a real integration test
                if import_count > 0 and patch_count == 0:
                    # But also check it's not just importing for type hints
                    if "def test_" in source:
                        found_unpatched_integration = True
                        break
        if found_unpatched_integration:
            break

    # The actual worker tests (TestReconcileCountryWorker) DO call through the shim
    # But ensemble-level integration (execute_single_run → reconcile) is fully mocked
    # This test checks for ensemble-level integration specifically
    ensemble_tests = REPO_ROOT / "tests/test_managers/test_ensemble_manager.py"
    if ensemble_tests.exists():
        source = ensemble_tests.read_text()
        # Actually check: is there a test that does NOT mock _reconcile_pg_with_c?
        tree = ast.parse(source)
        unmocked_reconcile_tests = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                # Check if this test's decorators include a patch of _reconcile_pg_with_c
                has_reconcile_patch = any(
                    "_reconcile_pg_with_c" in ast.dump(d)
                    for d in node.decorator_list
                )
                body_source = ast.get_source_segment(source, node) or ""
                calls_reconcile = "_reconcile_pg_with_c" in body_source
                if calls_reconcile and not has_reconcile_patch:
                    unmocked_reconcile_tests.append(node.name)

        assert unmocked_reconcile_tests, (
            "No test in test_ensemble_manager.py exercises _reconcile_pg_with_c "
            "without mocking it. The full integration path (ensemble → shim → "
            "views-reporting → pipeline-core datasets) is never tested."
        )


# ── V-3: Downstream consumer inventory must contain actual data ───────────────

def test_v3_downstream_consumer_data_exists():
    """V-3: A downstream consumer inventory with actual import mappings must exist."""
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
            "No downstream consumer inventory found. Without a file→symbol mapping "
            "of which downstream repos import extracted symbols, we cannot verify "
            "the extraction caused no import-path regressions in consumers."
        )

    text = found.read_text()
    # Must contain actual repo references with import paths
    has_views_models = "views-models" in text or "views_models" in text
    has_import_paths = "from views_pipeline_core" in text or "import views_pipeline_core" in text
    assert has_views_models and has_import_paths, (
        f"Inventory at {found} exists but does not contain actual import path data. "
        f"Must map downstream repos to specific symbols they import."
    )


# ── V-4: At least one behavioral equivalence test must exist ──────────────────

def test_v4_behavioral_equivalence_test_exists():
    """V-4: At least one test must verify pre-extraction behavior equals post-extraction behavior."""
    test_dir = REPO_ROOT / "tests"

    equivalence_indicators = [
        "assert_frame_equal",
        "assert_array_equal",
        "allclose",
        "snapshot",
        "golden",
        "baseline",
        "expected_output",
        "reference_output",
    ]

    # Look for tests that compare actual output against a saved/known baseline
    # from an extracted module (not just internal consistency checks)
    extracted_modules = [
        "ReconciliationModule",
        "ForecastReconciler",
        "PosteriorDistributionAnalyzer",
        "MappingModule",
        "ReportModule",
        "DatasetTransformationModule",
        "HistoricalLineGraph",
        "PlotDistribution",
    ]

    found_equivalence = False
    for test_file in test_dir.rglob("*.py"):
        source = test_file.read_text()
        uses_extracted = any(m in source for m in extracted_modules)
        uses_comparison = any(ind in source for ind in equivalence_indicators)
        if uses_extracted and uses_comparison:
            # Check if it's comparing against a saved fixture (not just internal checks)
            if "fixture" in source.lower() or "expected" in source.lower():
                found_equivalence = True
                break

    # This is deliberately strict: unit tests that assert shapes/types don't count.
    # We need tests that assert the SAME numerical outputs as before extraction.
    assert found_equivalence, (
        "No behavioral equivalence test found. The test suite verifies extracted "
        "modules work in isolation, but no test proves they produce identical outputs "
        "to the pre-extraction versions. A silent behavioral change during extraction "
        "would pass all current tests."
    )


# ── V-5: CI must have a minimal-environment test variant ──────────────────────

def test_v5_minimal_environment_ci_exists():
    """V-5: CI must include a test run without views-reporting to verify importorskip guards."""
    ci_dir = REPO_ROOT / ".github" / "workflows"
    if not ci_dir.exists():
        pytest.fail("No .github/workflows directory found")

    found_minimal = False
    for wf in ci_dir.glob("*.yml"):
        text = wf.read_text()
        # Look for a job/step that explicitly excludes views-reporting
        if ("without" in text.lower() and "reporting" in text.lower()) or \
           "minimal" in text.lower() or \
           "core-only" in text.lower() or \
           "pip uninstall views-reporting" in text:
            found_minimal = True
            break

    assert found_minimal, (
        "No CI workflow runs tests without views-reporting installed. "
        "The pytest.importorskip() guards in 5 test files have never been "
        "exercised in an automated environment. A broken guard would not be "
        "caught until a user installs pipeline-core without views-reporting."
    )


# ── V-6: At least one reconciliation test must use real ProcessPoolExecutor ───

def test_v6_real_concurrency_test_exists():
    """V-6: At least one test must exercise ReconciliationModule.reconcile() with real parallelism."""
    test_file = REPO_ROOT / "tests/test_modules/test_reconciliation.py"
    if not test_file.exists():
        pytest.skip("test_reconciliation.py not found")

    source = test_file.read_text()
    tree = ast.parse(source)

    # Find all test functions
    real_executor_tests = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            # Check if decorators include ProcessPoolExecutor patch
            decorator_src = "".join(
                ast.get_source_segment(source, d) or "" for d in node.decorator_list
            )
            if "ProcessPoolExecutor" in decorator_src:
                continue  # This test mocks the executor

            # Check if body calls reconcile() without prior patching
            body_src = ast.get_source_segment(source, node) or ""
            if ".reconcile(" in body_src and "ProcessPoolExecutor" not in body_src:
                real_executor_tests.append(node.name)

    assert real_executor_tests, (
        "All reconciliation orchestration tests mock ProcessPoolExecutor. "
        "No test verifies that _reconcile_country_worker can be pickled and "
        "executed in a subprocess post-extraction. Cross-package function "
        "serialization (views_reporting.reconciliation → subprocess) is untested."
    )
