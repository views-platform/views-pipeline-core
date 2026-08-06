"""
Falsification audit: extraction document package principles (2026-05-27).

Tests that the three documents in reports/views_reporting_extraction/
move the project toward package-level design principles (REP, CCP, CRP,
ADP, SDP, SAP).
"""
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "views_reporting_extraction"
INVESTIGATION = REPORTS_DIR / "architectural_misplacement_investigation.md"
PR_PLANS = REPORTS_DIR / "extraction_pr_plans.md"
VPC = Path(__file__).resolve().parents[1] / "views_pipeline_core"


# ---------------------------------------------------------------------------
# F-2: CRP — Ensemble managers forced to depend on entire views-reporting
# ---------------------------------------------------------------------------

class TestF2EnsembleDependencyProportionality:
    """After PR 7, ensemble managers import ReconciliationModule from
    views-reporting. CRP says consumers should not be forced to depend
    on things they don't use. The documents must acknowledge this
    disproportionate dependency or justify it."""

    def test_pr7_discusses_ensemble_dependency_scope(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        pr7_moves_reconciliation = "PR 7" in pr_text and "reconciliation" in pr_text

        if not pr7_moves_reconciliation:
            return

        has_proportionality_discussion = any(term in all_text.lower() for term in [
            "common reuse",
            "crp",
            "only need reconciliation",
            "only needs reconciliation",
            "disproportionate",
            "thin dependency",
            "lightweight import",
            "optional dependency",
            "extras_require",
            "reconciliation-only",
            "ensemble.*heavyweight",
        ])

        reconciliation_is_only_ensemble_need = (
            "optional per-ensemble" in all_text
            or "reconciliation is optional" in all_text.lower()
        )

        assert has_proportionality_discussion or not reconciliation_is_only_ensemble_need, (
            "PR 7 moves reconciliation (298 LOC) to views-reporting (~8,285 LOC total) "
            "but ensemble managers only need that one module. After extraction, ensemble "
            "managers would depend on the entire views-reporting install (torch, scipy, "
            "geopandas, seaborn, plotly, matplotlib — ~2.5 GB) for 298 LOC of optional "
            "reconciliation logic. This is the same heavyweight-dependency problem the "
            "extraction solves for pipeline-core, recreated in the other direction. "
            "Documents should discuss this CRP concern or propose a mitigation "
            "(e.g., optional dependency group, separate reconciliation package)."
        )


# ---------------------------------------------------------------------------
# F-3: ADP — Runtime import cycle via re-export shims
# ---------------------------------------------------------------------------

class TestF3RuntimeImportCycleAcknowledged:
    """Re-export shims create a runtime import cycle:
    pipeline-core.__init__ → views-reporting.X → pipeline-core.data.handlers.
    The documents must acknowledge this runtime cycle exists even though
    the install-time cycle is prevented."""

    def test_runtime_cycle_discussed(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        has_shims = "try:" in pr_text and "from views_reporting" in pr_text

        if not has_shims:
            return

        has_runtime_cycle_discussion = any(term in all_text.lower() for term in [
            "runtime cycle",
            "runtime import cycle",
            "import cycle",
            "runtime dependency cycle",
            "import order",
            "circular import",
            "acyclic dependencies",
            "adp",
            "bidirectional import",
        ])

        has_install_cycle_only = (
            "circular pip install" in all_text.lower()
            or "circular install" in all_text.lower()
        )

        if has_install_cycle_only and not has_runtime_cycle_discussion:
            assert False, (
                "Documents discuss the install-time circular dependency (PR 9 lines "
                "1082-1100) but not the runtime import cycle. When views-reporting IS "
                "installed, re-export shims create: pipeline-core.__init__.py → "
                "views_reporting.X → pipeline-core.data.handlers. Python can resolve "
                "this but it is fragile — import order matters, and developers may "
                "encounter confusing ImportError or AttributeError during development. "
                "Documents should acknowledge the runtime cycle exists and note that "
                "it resolves cleanly because shims are leaf-level __init__.py files."
            )


# ---------------------------------------------------------------------------
# F-5: REP — No version coordination strategy
# ---------------------------------------------------------------------------

class TestF5VersionCoordinationStrategy:
    """Two packages with a bidirectional dependency need a version
    compatibility strategy. The documents must specify how pipeline-core
    and views-reporting releases coordinate."""

    def test_version_compatibility_discussed(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        creates_two_packages = (
            "views-reporting" in all_text
            and "pyproject.toml" in all_text
            and "Dependency on views-pipeline-core" in all_text
        )

        if not creates_two_packages:
            return

        has_version_strategy = any(term in all_text.lower() for term in [
            "version pin",
            "version range",
            "version compatibility",
            "semver",
            "semantic version",
            "breaking change",
            "coordinated release",
            "release together",
            "version constraint",
            "lock-step",
            "lockstep",
            "compatible release",
        ])

        assert has_version_strategy, (
            "Extraction creates two packages with bidirectional dependency: "
            "views-reporting depends on pipeline-core (for _ViewsDataset), and "
            "pipeline-core's re-export shims depend on views-reporting. Documents "
            "cite 'independent versioning' as a benefit (investigation line 309) "
            "but never specify a version pinning strategy. A breaking change to "
            "_ViewsDataset (rename, remove attribute, change MultiIndex structure) "
            "would break all of views-reporting with no version constraint to "
            "prevent incompatible combinations. The pyproject.toml spec (PR 0 line "
            "101) says 'Dependency on views-pipeline-core' without a version range."
        )


# ---------------------------------------------------------------------------
# F-7: Adequacy — Package principles acknowledged
# ---------------------------------------------------------------------------

class TestF7PackagePrinciplesAcknowledged:
    """An architectural extraction creating two packages should acknowledge
    at least one package-level design principle by name or formal equivalent."""

    def test_at_least_one_package_principle_discussed(self):
        pr_text = PR_PLANS.read_text()
        inv_text = INVESTIGATION.read_text()
        all_text = pr_text + inv_text

        package_principles = [
            "reuse/release", "reuse-release",
            "common closure",
            "common reuse",
            "acyclic dependencies",
            "stable dependencies",
            "stable abstractions",
            "component cohesion", "package cohesion",
            "component coupling", "package coupling",
        ]

        acronyms_in_context = False
        for acronym in ["REP", "CCP", "CRP", "ADP", "SDP", "SAP"]:
            for i, line in enumerate(all_text.splitlines()):
                if f" {acronym} " in f" {line} " or line.strip().startswith(acronym):
                    surrounding = all_text[max(0, all_text.find(line)-200):
                                           all_text.find(line)+200].lower()
                    if any(kw in surrounding for kw in [
                        "principle", "package", "component", "cohesion", "coupling",
                    ]):
                        acronyms_in_context = True
                        break

        mentioned = [t for t in package_principles if t.lower() in all_text.lower()]

        has_implicit_discussion = (
            "change together" in all_text.lower()
            and "live together" in all_text.lower()
        ) or (
            "depend in the direction" in all_text.lower()
        ) or (
            "released together" in all_text.lower()
            and "reused together" in all_text.lower()
        )

        assert len(mentioned) >= 1 or acronyms_in_context or has_implicit_discussion, (
            "Extraction documents do not discuss any package-level design principle "
            "by name or formal equivalent. An architectural extraction of ~8,285 LOC "
            "creating two packages with bidirectional dependencies is the canonical "
            "domain for REP (reuse/release equivalence), CCP (common closure), CRP "
            "(common reuse), ADP (acyclic dependencies), SDP (stable dependencies), "
            "and SAP (stable abstractions). The documents make implicit decisions "
            "about all six but never name the framework — a reviewer cannot verify "
            "whether the package decomposition was evaluated against these criteria."
        )
