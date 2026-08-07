"""
Falsification audit: extraction document file organization (2026-05-27).

Tests that the three documents in reports/views_reporting_extraction/
move the project toward clean file/folder organization: one main class
per file, clear folder responsibilities, no dumping grounds.
"""
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "views_reporting_extraction"
INVESTIGATION = REPORTS_DIR / "architectural_misplacement_investigation.md"
PR_PLANS = REPORTS_DIR / "extraction_pr_plans.md"
VPC = Path(__file__).resolve().parents[1] / "views_pipeline_core"


# ---------------------------------------------------------------------------
# F-1: statistics.py — 2 unrelated classes move as one file
# ---------------------------------------------------------------------------

class TestF1StatisticsFileSplitting:
    """statistics.py contains PosteriorDistributionAnalyzer (Bayesian
    posterior analysis) and ForecastReconciler (spatial hierarchy constraint
    optimization). These have distinct responsibilities and distinct consumers.
    The plan should either split them or justify keeping them together."""

    STATS_FILE = VPC / "modules" / "statistics" / "statistics.py"

    def test_statistics_splitting_or_justification(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        if not self.STATS_FILE.exists():
            return

        class_count = self.STATS_FILE.read_text().count("\nclass ")
        if class_count < 2:
            return

        has_splitting_discussion = any(term in all_text.lower() for term in [
            "split statistics",
            "separate file",
            "posterior_distribution",
            "forecast_reconciler.py",
            "one class per file",
            "single class per file",
            "two classes in statistics",
            "reconciler.*separate",
        ])

        has_justification = any(term in all_text.lower() for term in [
            "statistics.py contains both",
            "both classes share",
            "cohesive statistical",
            "same statistical domain",
            "keep together because",
        ])

        assert has_splitting_discussion or has_justification, (
            "statistics.py (769 LOC) contains 2 classes with distinct responsibilities: "
            "PosteriorDistributionAnalyzer (Bayesian HDI/MAP analysis, consumed by "
            "handlers.py) and ForecastReconciler (scipy optimization for spatial "
            "hierarchy constraints, consumed by reconciliation.py). PR 2 copies the "
            "file as-is ('Exact copy') without discussing whether to split them into "
            "separate files in views-reporting. The claim requires 'a file should "
            "usually contain one main class.' Either split into "
            "posterior_distribution.py and forecast_reconciler.py, or justify why "
            "they belong together."
        )


# ---------------------------------------------------------------------------
# F-2: handlers.py — 7-class inheritance tree stays in one file
# ---------------------------------------------------------------------------

class TestF2HandlersFileStructure:
    """After PR 8, handlers.py retains 7 classes (~953 LOC) in one file.
    The documents should discuss whether this inheritance tree should be
    split across files or justify keeping it together."""

    HANDLERS = VPC / "data" / "handlers.py"

    def test_residual_handlers_structure_discussed(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        if not self.HANDLERS.exists():
            return

        class_count = self.HANDLERS.read_text().count("\nclass ")
        if class_count < 5:
            return

        has_file_structure_discussion = any(term in all_text.lower() for term in [
            "split handlers",
            "handlers.py file structure",
            "separate file for",
            "one file per dataset",
            "pg_dataset.py",
            "c_dataset.py",
            "class per file",
            "7 class",
            "seven class",
            "inheritance tree",
            "inheritance hierarchy",
            "file structure after",
        ])

        has_justification = any(term in all_text.lower() for term in [
            "classes belong together",
            "tightly coupled hierarchy",
            "keep in one file because",
            "single file is appropriate",
            "inheritance tree in one file",
        ])

        assert has_file_structure_discussion or has_justification, (
            "After PR 8 extraction, handlers.py retains 7 classes in one file: "
            "_ViewsDataset (base), _PGDataset, PGMDataset, PGYDataset, _CDataset, "
            "CMDataset, CYDataset (~953 LOC). The investigation's 'Residual "
            "_ViewsDataset' section (lines 149-163) describes the LOC reduction but "
            "never addresses the 7-class inheritance tree structure. The claim says "
            "'inheritance-related classes may sometimes belong together, but even there "
            "we should be careful.' The documents should either propose splitting "
            "(e.g., base.py + pg_datasets.py + c_datasets.py) or justify why 7 classes "
            "in one file is the right structure for a 'lightweight typed DataFrame "
            "wrapper.'"
        )


# ---------------------------------------------------------------------------
# F-5: Multi-class files relocated without structural improvement
# ---------------------------------------------------------------------------

class TestF5MultiClassFileAwareness:
    """The extraction plan should acknowledge that some files being moved
    contain multiple classes and state whether this is intentional."""

    def test_multi_class_pattern_acknowledged(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        has_multi_class_awareness = any(term in all_text.lower() for term in [
            "multiple class",
            "multi-class",
            "two classes",
            "several classes",
            "classes in one file",
            "class per file",
            "one class per",
            "single class per",
            "classes share a file",
            "file contains both",
        ])

        assert has_multi_class_awareness, (
            "Extraction documents move statistics.py (2 classes) as 'Exact copy' and "
            "leave handlers.py (7 classes) in pipeline-core, without acknowledging "
            "the multi-class-per-file pattern anywhere. The claim requires 'a file "
            "should usually contain one main class' and 'multiple classes in the same "
            "file should be the exception, not the default.' The documents should "
            "state which multi-class files are intentional exceptions (e.g., tightly "
            "coupled inheritance hierarchies) and which should be split in a future "
            "cleanup."
        )


# ---------------------------------------------------------------------------
# F-6: File organization principles not articulated
# ---------------------------------------------------------------------------

class TestF6FileOrganizationPrinciples:
    """An extraction creating a new package should state file organization
    goals: one concept per file, clear folder names, navigability."""

    def test_file_organization_goals_stated(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        has_file_org_principles = any(term in all_text.lower() for term in [
            "one class per file",
            "one concept per file",
            "single responsibility per file",
            "file organization",
            "folder organization",
            "package layout",
            "navigable",
            "new developer",
            "dumping ground",
            "clear responsibilities",
            "obvious structure",
            "file structure principle",
            "file structure goal",
        ])

        assert has_file_org_principles, (
            "Extraction documents discuss 11 design principles (5 SOLID + 6 package) "
            "but never articulate file organization goals. The Design Principle "
            "Assessment covers class-level design (SOLID) and component-level coupling "
            "(REP/CCP/CRP/ADP/SDP/SAP) but skips the level between: how files and "
            "folders should be organized within each package. A new package is the "
            "ideal time to state goals like 'one main concept per file', 'folder names "
            "should convey responsibility', 'no utility dumping grounds.' Without "
            "these, the 'Exact copy' pattern may perpetuate existing file-level "
            "structural problems."
        )