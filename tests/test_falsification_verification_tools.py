"""
Falsification audit 3: verification tool sufficiency (2026-05-31).

Attacks the claim: "We have the tools to 100% verify if this
refactor/export has been done with no regression."

Findings:
  V-2: No integration test exercises a full shim->views-reporting->pipeline-core path
  V-3: No downstream consumer verification exists
  V-4: No behavioral equivalence (pre/post) test exists
  V-5: importorskip guards never exercised in minimal environment
  V-6: Reconciliation parallelism entirely mocked — pickling untested
"""
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# -- V-4: At least one behavioral equivalence test must exist ------------------

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
        if test_file.name.startswith("test_falsification"):
            continue
        source = test_file.read_text()
        uses_extracted = any(m in source for m in extracted_modules)
        uses_comparison = any(ind in source for ind in equivalence_indicators)
        if uses_extracted and uses_comparison:
            if "fixture" in source.lower() or "expected" in source.lower():
                found_equivalence = True
                break

    assert found_equivalence, (
        "No behavioral equivalence test found. The test suite verifies extracted "
        "modules work in isolation, but no test proves they produce identical outputs "
        "to the pre-extraction versions. A silent behavioral change during extraction "
        "would pass all current tests."
    )


# -- V-5: CI must have a minimal-environment test variant ----------------------

def test_v5_minimal_environment_ci_exists():
    """V-5: CI must include a test run without views-reporting to verify importorskip guards."""
    ci_dir = REPO_ROOT / ".github" / "workflows"
    if not ci_dir.exists():
        pytest.fail("No .github/workflows directory found")

    found_minimal = False
    for wf in ci_dir.glob("*.yml"):
        text = wf.read_text()
        if ("without" in text.lower() and "reporting" in text.lower()) or \
           "minimal" in text.lower() or \
           "core-only" in text.lower() or \
           "pip uninstall views-reporting" in text:
            found_minimal = True
            break

    assert found_minimal, (
        "No CI workflow runs tests without views-reporting installed. "
        "The pytest.importorskip() guards have never been exercised in an "
        "automated environment. A broken guard would not be caught until a "
        "user installs pipeline-core without views-reporting."
    )
