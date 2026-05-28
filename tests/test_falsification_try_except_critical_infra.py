"""
Falsification audit: "A try/except solution is not an issue in critical infrastructure"
Generated: 2026-05-28

These tests encode findings from the falsification audit. Each test currently
FAILS, documenting a real issue. Fix the underlying code, then the test passes.
"""

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# F-1 (HARD): Nested ImportError masking — shim catches wrong ImportError
# ---------------------------------------------------------------------------

class TestF1_NestedImportErrorMasking:
    """
    ADR-054's shim pattern catches bare `except ImportError`. If views_reporting
    is installed but has a broken transitive dependency (torch, scipy, geopandas),
    the shim catches the WRONG ImportError and tells the user "Install
    views-reporting" — a completely misleading diagnosis.

    Fix: use `except ImportError as e:` + inspect `e` or at minimum chain with
    `raise ... from e` so the original traceback is preserved.
    """

    def test_shim_preserves_original_cause_via_from_e(self):
        """
        The prescribed shim pattern must use 'from e' to preserve the original
        ImportError's traceback. Without it, a broken transitive dependency
        (e.g., torch._C missing) is misdiagnosed as 'views-reporting not installed'.
        """
        # Read ADR-054's shim pattern
        adr_path = Path("documentation/ADRs/054_visualization_and_reporting_extraction.md")
        content = adr_path.read_text()

        # Find the shim code block
        in_shim = False
        shim_lines = []
        for line in content.splitlines():
            if "try:" in line and not in_shim:
                # Look ahead for the ImportError pattern
                in_shim = True
                shim_lines = [line]
            elif in_shim:
                shim_lines.append(line)
                if ")" in line and "raise" in "".join(shim_lines):
                    break

        shim_text = "\n".join(shim_lines)

        # The shim MUST contain 'from e' to chain exceptions
        assert "from e" in shim_text, (
            f"ADR-054 shim pattern catches bare 'except ImportError' without "
            f"'from e'. A broken transitive dependency inside views_reporting "
            f"will be misdiagnosed as 'views-reporting not installed'. "
            f"Pattern found:\n{shim_text}"
        )

    def test_extraction_plan_shims_preserve_original_cause(self):
        """
        All 7 shim examples in extraction_pr_plans.md must use 'from e'.
        """
        plans_path = Path(
            "reports/views_reporting_extraction/extraction_pr_plans.md"
        )
        content = plans_path.read_text()

        # Count shim patterns: 'except ImportError:' (bare) vs 'except ImportError as e:'
        bare_count = content.count("except ImportError:")
        chained_count = content.count("except ImportError as e:")

        assert bare_count == 0, (
            f"extraction_pr_plans.md has {bare_count} bare 'except ImportError:' "
            f"patterns (should be 0). All shims must use 'except ImportError as e:' "
            f"with 'raise ... from e' to preserve tracebacks."
        )
        assert chained_count >= 7, (
            f"Expected at least 7 shim examples with 'except ImportError as e:', "
            f"found {chained_count}."
        )


# ---------------------------------------------------------------------------
# F-3 (SOFT): Missing 'from e' — inconsistent with dataloaders.py precedent
# ---------------------------------------------------------------------------

class TestF3_MissingFromE:
    """
    dataloaders.py:1171 already uses the correct pattern:
        except ImportError as e:
            raise ImportError(...) from e

    The ADR-054 shim and all extraction plan shims omit 'from e', silently
    discarding the original traceback. This is inconsistent with the codebase's
    own precedent and makes debugging harder in critical infrastructure.
    """

    def test_investigation_report_shim_uses_from_e(self):
        """
        The investigation report's Section 8 shim example must use 'from e'.
        """
        inv_path = Path(
            "reports/views_reporting_extraction/"
            "architectural_misplacement_investigation.md"
        )
        content = inv_path.read_text()

        # Find the shim section
        in_section_8 = False
        in_code_block = False
        shim_code = []
        for line in content.splitlines():
            if "Re-Export Shims" in line:
                in_section_8 = True
            elif in_section_8 and line.strip().startswith("```python"):
                in_code_block = True
            elif in_section_8 and in_code_block:
                if line.strip().startswith("```"):
                    break
                shim_code.append(line)

        shim_text = "\n".join(shim_code)
        assert "from e" in shim_text, (
            f"Investigation report shim example omits 'from e'. "
            f"Pattern found:\n{shim_text}"
        )


# ---------------------------------------------------------------------------
# F-5 (SOFT): report.py silent degradation violates ADR-008
# ---------------------------------------------------------------------------

class TestF5_ReportSilentDegradation:
    """
    report.py:213 catches ImportError for the markdown module and falls back
    to plain text without logging. ADR-008 requires 'Log at WARNING + document
    scope of degradation' for degraded operations. The module doesn't even
    import logging.
    """

    @pytest.mark.xfail(reason="C-133: report.py missing logging import")
    def test_report_module_imports_logging(self):
        """ReportModule must import logging to comply with ADR-008."""
        report_path = Path(
            "views_pipeline_core/modules/reports/report.py"
        )
        content = report_path.read_text()

        assert "import logging" in content or "from logging" in content, (
            "report.py does not import logging. ADR-008 requires all modules "
            "that handle degraded operations to log at WARNING level."
        )

    @pytest.mark.xfail(reason="C-133: report.py fallback missing logger.warning")
    def test_report_markdown_fallback_logs_warning(self):
        """
        The except ImportError block in add_markdown() must log a WARNING
        before falling back to plain text. ADR-008 §Degraded Operation:
        'Log at WARNING + document scope of degradation'.
        """
        report_path = Path(
            "views_pipeline_core/modules/reports/report.py"
        )
        lines = report_path.read_text().splitlines(True)

        # Find the except ImportError block
        for i, line in enumerate(lines):
            if "except ImportError:" in line and i > 190:
                # Check next 5 lines for logger.warning
                context = "".join(lines[i : i + 5])
                assert "logger.warning" in context or "logging.warning" in context, (
                    f"report.py:{i+1} catches ImportError for markdown module "
                    f"without logging a WARNING. ADR-008 requires degraded "
                    f"operations to 'Log at WARNING + document scope of degradation'. "
                    f"Block:\n{context}"
                )
                return

        pytest.fail("Could not find 'except ImportError:' block in report.py after line 190")
