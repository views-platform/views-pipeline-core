"""
Falsification audit: extraction document sufficiency (2026-05-27).

Tests that the three documents in reports/views_reporting_extraction/ are
sufficient for a developer to execute the full extraction effort.
"""
from pathlib import Path

import pytest

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "views_reporting_extraction"
INVESTIGATION = REPORTS_DIR / "architectural_misplacement_investigation.md"
PR_PLANS = REPORTS_DIR / "extraction_pr_plans.md"
VPC = Path(__file__).resolve().parents[1] / "views_pipeline_core"
TESTS_DIR = Path(__file__).resolve().parents[1] / "tests"


# ---------------------------------------------------------------------------
# F-1: Circular install dependency must be documented with resolution
# ---------------------------------------------------------------------------

class TestF1CircularDependencyResolution:
    """If pipeline-core depends on views-reporting (re-export shims) AND
    views-reporting depends on pipeline-core (_ViewsDataset etc), the
    documents must specify how to resolve the circular install dependency."""

    def test_circular_dependency_acknowledged(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        has_circular_discussion = any(term in all_text.lower() for term in [
            "circular dependency",
            "circular install",
            "install order",
            "mutual dependency",
        ])

        pipeline_depends_on_reporting = (
            "views-reporting" in pr_text
            and "dependency in `pyproject.toml`" in pr_text
            and "pipeline-core needs it" in pr_text
        )
        reporting_depends_on_pipeline = (
            "Dependency on views-pipeline-core" in pr_text
        )

        if pipeline_depends_on_reporting and reporting_depends_on_pipeline:
            assert has_circular_discussion, (
                "PR plans create a circular dependency: pipeline-core → views-reporting "
                "(re-export shims, PR 9 line 972) AND views-reporting → pipeline-core "
                "(PR 0 line 75). No document discusses how to resolve this at install "
                "time. `pip install views-pipeline-core` would fail. Resolution options: "
                "(1) make shims use try/except ImportError, (2) use editable installs "
                "only, (3) don't add views-reporting as a hard dependency — only "
                "re-export shims need it and they're transitional."
            )


# ---------------------------------------------------------------------------
# F-2: Downstream consumer identification for PR 1
# ---------------------------------------------------------------------------

class TestF2DownstreamConsumerInventory:
    """PR 1 (transformations) has zero internal consumers. The documents must
    identify specific downstream repos and files that import it."""

    def test_downstream_consumers_identified(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        has_specific_downstream = any(pattern in all_text for pattern in [
            "views-models/",
            "views_models/",
            "views-hydranet/",
            "views_hydranet/",
            "views-baseline/",
            "views_baseline/",
        ])

        mentions_zero_internal = "zero internal consumers" in all_text.lower()

        assert not mentions_zero_internal or has_specific_downstream, (
            "Documents say DatasetTransformationModule has 'zero internal consumers' "
            "but never identify which downstream repos/files import it. A developer "
            "cannot verify PR 1 is complete without knowing which files in views-models, "
            "views-hydranet, or views-baseline need companion import updates."
        )


# ---------------------------------------------------------------------------
# F-3: tqdm removal claim vs actual usage
# ---------------------------------------------------------------------------

class TestF3TqdmRemovalAccuracy:
    """Documents must not claim tqdm should be removed from pipeline-core
    if tqdm is still imported by code that stays."""

    def test_tqdm_not_listed_as_removable_dep(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()

        staying_files_with_tqdm = []
        staying_dirs = [
            VPC / "managers",
            VPC / "cli",
            VPC / "configs",
        ]
        for d in staying_dirs:
            if d.exists():
                for py in d.rglob("*.py"):
                    if "__pycache__" in str(py):
                        continue
                    content = py.read_text()
                    if "import tqdm" in content or "from tqdm" in content:
                        staying_files_with_tqdm.append(
                            str(py.relative_to(VPC.parent))
                        )

        claims_tqdm_removable = (
            "remove torch/matplotlib/joblib/tqdm" in inv_text.lower()
            or "tqdm" in pr_text[pr_text.find("PR 9"):pr_text.find("PR 10")]
        )

        if staying_files_with_tqdm and claims_tqdm_removable:
            assert False, (
                f"Documents claim tqdm should be removed from pipeline-core, but "
                f"{len(staying_files_with_tqdm)} files that STAY still import tqdm: "
                f"{staying_files_with_tqdm}. tqdm must remain in pyproject.toml."
            )


# ---------------------------------------------------------------------------
# F-4: PR 8 function signature specification
# ---------------------------------------------------------------------------

class TestF4MethodExtractionSpecification:
    """PR 8 must specify function signatures for extracted methods, not just
    'replace self with dataset'."""

    def test_pr8_specifies_attribute_handling(self):
        pr_text = PR_PLANS.read_text()
        pr8_section = pr_text[pr_text.find("PR 8:"):pr_text.find("PR 9:")]

        attributes_accessed = [
            "is_prediction", "to_tensor", "targets", "get_subset_tensor",
            "dataframe", "num_entities", "sample_size", "_time_values",
            "_entity_values", "original_index",
        ]
        mentioned = [a for a in attributes_accessed if a in pr8_section]

        assert len(mentioned) >= 3, (
            f"PR 8 says 'Replace all self. references with dataset.' but doesn't "
            f"specify how to handle the {len(attributes_accessed)} distinct instance "
            f"attributes accessed by extracted methods. Only {len(mentioned)} are "
            f"mentioned: {mentioned}. A developer must design ~15 function signatures "
            f"with no guidance on which attributes to pass individually vs as the "
            f"full dataset object."
        )


# ---------------------------------------------------------------------------
# F-6: Test file migration inventory
# ---------------------------------------------------------------------------

class TestF6TestFileMigrationInventory:
    """Documents must identify test files that import from modules being
    extracted, so a developer knows which tests need import updates."""

    EXTRACTED_IMPORT_PATTERNS = [
        "from views_pipeline_core.modules.statistics",
        "from views_pipeline_core.modules.visualizations",
        "from views_pipeline_core.modules.mapping",
        "from views_pipeline_core.modules.reports",
        "from views_pipeline_core.modules.reconciliation",
        "from views_pipeline_core.modules.transformations",
    ]

    @pytest.mark.skip(reason="RETIRED (#183, 2026-07-24): the extraction is executed and finalized (ADR-054 update block); planning-doc sufficiency is a historical audit and its inventory shrank when the shim-era test file was deleted.")
    def test_affected_test_files_listed(self):
        affected_tests = []
        for py in TESTS_DIR.rglob("*.py"):
            if "__pycache__" in str(py):
                continue
            content = py.read_text()
            for pattern in self.EXTRACTED_IMPORT_PATTERNS:
                if pattern in content:
                    affected_tests.append(py.name)
                    break

        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        mentioned_tests = [t for t in affected_tests if t in all_text]

        assert len(mentioned_tests) >= len(affected_tests) // 2, (
            f"{len(affected_tests)} test files import from modules being extracted: "
            f"{affected_tests}. Only {len(mentioned_tests)} are mentioned in the "
            f"documents: {mentioned_tests}. No PR plan lists these in 'Files Changed'. "
            f"A developer won't know which tests need import updates or migration."
        )