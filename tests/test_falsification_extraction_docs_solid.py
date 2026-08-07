"""
Falsification audit: extraction document SOLID adherence (2026-05-27).

Tests that the extraction documents in reports/views_reporting_extraction/
move the project toward SOLID principles.
"""
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "views_reporting_extraction"
INVESTIGATION = REPORTS_DIR / "architectural_misplacement_investigation.md"
PR_PLANS = REPORTS_DIR / "extraction_pr_plans.md"
VPC = Path(__file__).resolve().parents[1] / "views_pipeline_core"


# ---------------------------------------------------------------------------
# F-2: LSP — Subclass-specific functions must use subclass type hints
# ---------------------------------------------------------------------------

class TestF2SubclassTypeHints:
    """PR 8 extracts functions from _PGDataset/_CDataset that access
    subclass-specific attributes. These must use subclass type hints,
    not the base _ViewsDataset."""

    PG_SPECIFIC_FUNCTIONS = [
        "reconcile_pg_dataset",
        "build_pg_metadata_cache",
        "detect_country_changes",
        "get_country_id",
        "get_subset_by_country_id",
    ]

    def test_pg_functions_specify_pg_type(self):
        pr_text = PR_PLANS.read_text()
        pr8_section = pr_text[pr_text.find("PR 8:"):pr_text.find("PR 9:")]

        for func_name in self.PG_SPECIFIC_FUNCTIONS:
            if func_name in pr8_section:
                func_context = pr8_section[
                    pr8_section.find(func_name):pr8_section.find(func_name) + 200
                ]
                has_pg_type = (
                    "_PGDataset" in func_context
                    or "pg_dataset" in func_context
                )
                assert has_pg_type, (
                    f"Function `{func_name}` accesses _PGDataset-specific attributes "
                    f"(_country_to_grids_cache, reconciled_dataframe) but its spec uses "
                    f"`_ViewsDataset` type hint. Calling with _CDataset would crash at "
                    f"runtime with AttributeError. Spec must use `_PGDataset` type hint."
                )


# ---------------------------------------------------------------------------
# F-3: DIP — At least one abstraction at the package boundary
# ---------------------------------------------------------------------------

class TestF3AbstractionAtBoundary:
    """The extraction creates a new package boundary between pipeline-core
    and views-reporting. At least one Protocol, ABC, or interface should
    be defined at this boundary."""

    def test_any_abstraction_mentioned(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        has_abstraction = any(term in all_text for term in [
            "Protocol",
            "ABC",
            "abstractmethod",
            "abstract base",
            "DatasetProtocol",
            "ReconciliationStrategy",
            "AggregationStrategy",
            "ReportTemplateProtocol",
        ])

        has_wet_before_dry = "WET-before-DRY" in all_text or "wet-before-dry" in all_text.lower()

        assert has_abstraction or has_wet_before_dry, (
            "Extraction creates a new package boundary (pipeline-core ↔ views-reporting) "
            "but no Protocol, ABC, or interface is introduced at that boundary. Ensemble "
            "managers depend on concrete ReconciliationModule/AggregationModule. "
            "ReportingStage depends on concrete template classes. Extracted functions "
            "take concrete _ViewsDataset. DIP requires high-level code to depend on "
            "abstractions, not concrete implementations."
        )


# ---------------------------------------------------------------------------
# F-4: OCP — ReportingStage extensibility
# ---------------------------------------------------------------------------

class TestF4ReportingStageExtensibility:
    """After extraction, adding a new report type should not require
    modifying ReportingStage source code."""

    def test_reporting_stage_uses_extensible_pattern(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        has_extensible_pattern = any(term in all_text.lower() for term in [
            "report registry",
            "plugin",
            "report protocol",
            "reporttemplate protocol",
            "template registry",
            "dispatch table",
            "report_types",
        ])

        stage_path = VPC / "managers" / "reporting" / "stage.py"
        if stage_path.exists():
            stage_code = stage_path.read_text()
            hardcoded_imports = stage_code.count("from views_pipeline_core.templates.reports")
            if hardcoded_imports >= 2:
                assert has_extensible_pattern, (
                    "ReportingStage has hardcoded deferred imports for each report type "
                    f"({hardcoded_imports} found). After extraction, these point to "
                    "views-reporting but the pattern is unchanged — adding a new report "
                    "type still requires editing stage.py. No registry, protocol, or "
                    "plugin pattern is proposed. OCP requires code to be open for "
                    "extension but closed for modification."
                )


# ---------------------------------------------------------------------------
# F-5: LSP — isinstance dispatches addressed, not just relocated
# ---------------------------------------------------------------------------

class TestF5IsinstanceDispatchesAddressed:
    """The extraction moves modules containing isinstance dispatches on
    _PGDataset/_CDataset. The plan should address these dispatches,
    not just relocate them."""

    def test_isinstance_dispatches_discussed(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        discusses_isinstance = any(term in all_text for term in [
            "isinstance",
            "type dispatch",
            "polymorphi",
            "replace type check",
            "visitor pattern",
        ])

        isinstance_files_move = any(f in all_text for f in [
            "mapping.py",
            "historical.py",
        ])

        if isinstance_files_move and not discusses_isinstance:
            actual_isinstance_count = 0
            for pattern_file in [
                VPC / "modules" / "mapping" / "mapping.py",
                VPC / "modules" / "visualizations" / "historical.py",
                VPC / "templates" / "reports" / "forecast.py",
            ]:
                if pattern_file.exists():
                    content = pattern_file.read_text()
                    actual_isinstance_count += content.count("isinstance")

            assert actual_isinstance_count == 0, (
                f"Extraction moves mapping.py and historical.py to views-reporting but "
                f"does not discuss the {actual_isinstance_count} isinstance dispatches "
                f"on _PGDataset/_CDataset in those files. These dispatches violate LSP "
                f"(a _ViewsDataset subclass cannot be substituted without type checking). "
                f"The plan should either: (1) introduce polymorphic methods on the "
                f"dataset classes, (2) use a strategy/visitor pattern, or (3) explicitly "
                f"document that LSP improvement is deferred to a future sprint."
            )


# ---------------------------------------------------------------------------
# F-6: Adequacy — SOLID principles acknowledged
# ---------------------------------------------------------------------------

class TestF6SolidPrinciplesAcknowledged:
    """An architectural extraction plan should explicitly state which
    design principles it addresses and which it defers."""

    def test_at_least_one_solid_principle_discussed(self):
        inv_text = INVESTIGATION.read_text()
        pr_text = PR_PLANS.read_text()
        all_text = inv_text + pr_text

        solid_terms = [
            "single responsibility",
            "open/closed", "open-closed",
            "liskov", "substitut",
            "interface segregation",
            "dependency inversion",
            "SRP", "OCP", "LSP", "ISP", "DIP",
            "SOLID",
        ]

        mentioned = [t for t in solid_terms if t.lower() in all_text.lower()]

        god_class_mentioned = "god class" in all_text.lower()
        misplacement_mentioned = "misplaced" in all_text.lower()

        assert len(mentioned) >= 1 or (god_class_mentioned and misplacement_mentioned), (
            "Extraction documents do not discuss any SOLID principle by name or formal "
            "equivalent. An architectural refactoring of ~8,285 LOC should state which "
            "design principles it addresses (SRP via god class decomposition, ISP via "
            "import cleanup) and which it defers (DIP — no protocols, OCP — no plugin "
            "pattern, LSP — isinstance dispatches persist)."
        )