"""
Falsification audit: extraction document consistency (2026-05-27).

Tests that the three documents in reports/views_reporting_extraction/ are
internally and mutually consistent on quantitative facts.
"""
import re
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "views_reporting_extraction"
INVESTIGATION = REPORTS_DIR / "architectural_misplacement_investigation.md"
PR_PLANS = REPORTS_DIR / "extraction_pr_plans.md"
SPRINT_PLAN = REPORTS_DIR / "sprint_plan_c105_eval_report_scale_aware_graphs.md"


# ---------------------------------------------------------------------------
# F-1: handlers.py misplaced LOC — three contradictory figures
# ---------------------------------------------------------------------------

class TestF1HandlerMethodLOCConsistency:
    """The investigation must use ONE figure for handlers.py misplaced LOC."""

    def test_exec_summary_matches_census(self):
        text = INVESTIGATION.read_text()
        census_match = re.search(
            r"Subtotal misplaced in.*?handlers\.py.*?~?([\d,]+)\s*LOC", text
        )
        assert census_match, "Could not find census subtotal for handlers.py"
        census_loc = int(census_match.group(1).replace(",", ""))

        assert "~1,279 LOC" not in text or str(census_loc) in text.split("~1,279")[0], (
            f"Executive summary says ~1,279 LOC but census says ~{census_loc} LOC "
            f"for handlers.py misplaced methods. These must agree."
        )

    def test_complete_files_plus_methods_equals_grand_total(self):
        text = INVESTIGATION.read_text()
        assert "7,632" not in text or "6,943" not in text, (
            "Document contains both 7,632 (stale: 6,943 + old 689 estimate) and "
            "6,943 (current census subtotal). The 7,632 figure embeds the old C-36 "
            "estimate and must be updated to match the census."
        )


# ---------------------------------------------------------------------------
# F-2: Grand total LOC — irreconcilable totals
# ---------------------------------------------------------------------------

class TestF2GrandTotalConsistency:
    """Exec summary total and census grand total must agree within 100 LOC."""

    def test_exec_summary_total_matches_census_total(self):
        text = INVESTIGATION.read_text()
        grand_total_match = re.search(
            r"Total misplaced code.*?~?([\d,]+)\s*LOC", text
        )
        assert grand_total_match, "Could not find grand total LOC"
        grand_total = int(grand_total_match.group(1).replace(",", ""))

        assert "~8,900" not in text or abs(grand_total - 8900) < 100, (
            f"Exec summary says ~8,900 LOC but census grand total is ~{grand_total}. "
            f"Difference of {abs(grand_total - 8900)} LOC exceeds tolerance."
        )


# ---------------------------------------------------------------------------
# F-3: "13 source files" vs census count
# ---------------------------------------------------------------------------

class TestF3SourceFileCount:
    """Exec summary source file count must match census table item count."""

    def test_source_file_count_matches_census(self):
        text = INVESTIGATION.read_text()
        source_file_claims = re.findall(r"(\d+)\s+source files", text)
        assert source_file_claims, "No 'N source files' claim found"

        census_section = text[text.find("Files That Move as Complete Units"):
                              text.find("Methods That Move from")]
        census_py_files = len(re.findall(
            r"^\|\s*\d+\s*\|\s*`[^`]+\.py`", census_section, re.MULTILINE
        ))
        for claim in source_file_claims:
            assert int(claim) == census_py_files, (
                f"Document claims {claim} source files but census table has "
                f"{census_py_files} .py file entries."
            )


# ---------------------------------------------------------------------------
# F-4: handlers.py import line numbers vs ground truth
# ---------------------------------------------------------------------------

class TestF4HandlerLineNumbers:
    """Import line numbers cited in the investigation must match the actual file."""

    HANDLERS = (
        Path(__file__).resolve().parents[1]
        / "views_pipeline_core" / "data" / "handlers.py"
    )

    def _actual_line(self, pattern: str) -> int:
        for i, line in enumerate(self.HANDLERS.read_text().splitlines(), 1):
            if pattern in line:
                return i
        raise AssertionError(f"Pattern {pattern!r} not found in handlers.py")

    def test_matplotlib_line_number(self):
        actual = self._actual_line("import matplotlib")
        text = INVESTIGATION.read_text()
        assert f"handlers.py:{actual}" in text or "handlers.py:9" not in text, (
            f"Investigation cites handlers.py:9 for matplotlib but actual line is {actual}."
        )

    def test_joblib_line_number(self):
        actual = self._actual_line("from joblib")
        text = INVESTIGATION.read_text()
        assert f"handlers.py:{actual}" in text or "handlers.py:11" not in text, (
            f"Investigation cites handlers.py:11 for joblib but actual line is {actual}."
        )

    def test_torch_line_number(self):
        actual = self._actual_line("import torch")
        text = INVESTIGATION.read_text()
        assert f"handlers.py:{actual}" in text or "handlers.py:14" not in text, (
            f"Investigation cites handlers.py:14 for torch but actual line is {actual}."
        )


# ---------------------------------------------------------------------------
# F-5: Step count mismatch between investigation and PR plans
# ---------------------------------------------------------------------------

class TestF5StepCountAlignment:
    """Investigation extraction steps and PR plan numbers should map clearly."""

    def test_investigation_mentions_step_zero(self):
        text = INVESTIGATION.read_text()
        has_step_0 = bool(re.search(r"Step\s+0|step\s+0|\|\s*0\s*\|", text))
        pr_text = PR_PLANS.read_text()
        has_pr_0 = "PR 0" in pr_text

        assert has_step_0 == has_pr_0, (
            f"Investigation has step 0: {has_step_0}, PR plans has PR 0: {has_pr_0}. "
            "Documents disagree on whether the skeleton/ADR is a numbered step."
        )


# ---------------------------------------------------------------------------
# F-6: Sprint plan targets files that extraction will move
# ---------------------------------------------------------------------------

class TestF6SprintPlanFilePathValidity:
    """Sprint plan C-105 must not target file paths that the extraction
    investigation says will be deleted before Sprint 4 begins, without
    a clear superseded/stale warning header."""

    def test_evaluation_template_path_has_superseded_header(self):
        sprint = SPRINT_PLAN.read_text()

        sprint_targets_old_path = (
            "views_pipeline_core/templates/reports/evaluation.py" in sprint
        )
        has_superseded_header = (
            "SUPERSEDED" in sprint or "superseded" in sprint.lower()[:500]
        )

        assert not sprint_targets_old_path or has_superseded_header, (
            "Sprint plan C-105 targets views_pipeline_core/templates/reports/evaluation.py "
            "but the investigation says this file moves to views-reporting before Sprint 4. "
            "The sprint plan must include a SUPERSEDED header warning."
        )


# ---------------------------------------------------------------------------
# F-7: Dependency count inconsistency
# ---------------------------------------------------------------------------

class TestF7DependencyCountConsistency:
    """The investigation should use one consistent count for dependencies removed."""

    def test_heavyweight_dep_count_consistent(self):
        text = INVESTIGATION.read_text()
        counts = set()

        if "7 heavyweight" in text:
            counts.add(7)
        match_range = re.search(r"(\d)-(\d)\s+(?:deps|entries|dependencies)", text)
        if match_range:
            counts.add(int(match_range.group(1)))
            counts.add(int(match_range.group(2)))

        dep_list_section = text[text.find("Install Footprint Reduction"):]
        specific_deps = re.findall(
            r"(torch|scipy|geopandas|plotly|matplotlib|seaborn|markdown|joblib|properscoring)",
            dep_list_section[:500],
        )
        if specific_deps:
            counts.add(len(set(specific_deps)))

        assert len(counts) <= 2, (
            f"Investigation uses {len(counts)} different dependency counts: {sorted(counts)}. "
            "Should define 'heavyweight' and use one consistent number."
        )
