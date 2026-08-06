"""
Falsification audit: "PR 1 is the simplest possible extraction"
Generated: 2026-05-28

These tests encode findings. Each currently FAILS, documenting a real issue.
Fix the underlying issue, then the test passes.
"""
import pytest

pytest.importorskip("views_reporting")

from pathlib import Path

# ---------------------------------------------------------------------------
# F-1 (HARD): Test file imports bypass the shim
# ---------------------------------------------------------------------------

class TestF1_TestImportBypassesShim:
    """
    test_transformations.py:5 imports via the direct file path:
        from views_pipeline_core.modules.transformations.transformations import ...

    The re-export shim in __init__.py only covers package-level imports:
        from views_pipeline_core.modules.transformations import ...

    After transformations.py is deleted, the direct file import fails with
    ModuleNotFoundError. All 38 tests in test_transformations.py break.

    Fix: update the test import to use the package path, or add the test
    file to the PR plan's "Files Changed" table with an import update.
    """

    @pytest.mark.skip(reason="RETIRED (#183, 2026-07-24): the audited artifact (test_transformations.py shim-bypass import) was deleted with the shim; the extraction era is closed (ADR-054 update block).")
    def test_test_file_uses_package_import_not_file_import(self):
        """
        test_transformations.py must import via the package path (covered
        by the shim) not the direct file path (bypasses the shim).
        """
        test_path = Path("tests/test_modules/test_transformations.py")
        content = test_path.read_text()

        direct_import = (
            "from views_pipeline_core.modules.transformations.transformations"
        )

        has_direct = direct_import in content

        assert not has_direct, (
            f"test_transformations.py imports via direct file path:\n"
            f"  {direct_import}\n"
            f"This bypasses the re-export shim in __init__.py. After "
            f"transformations.py is deleted, this import fails with "
            f"ModuleNotFoundError and all 38 tests break.\n"
            f"Fix: change to: from views_pipeline_core.modules."
            f"transformations import DatasetTransformationModule"
        )

    def test_pr_plan_lists_test_file_in_changed_files(self):
        """
        The PR 1 plan must acknowledge test_transformations.py needs
        an import update (or note that the shim covers it).
        """
        plan_path = Path(
            "reports/views_reporting_extraction/extraction_pr_plans.md"
        )
        content = plan_path.read_text()

        pr1_section = content.split("## PR 1:")[1].split("## PR 2:")[0]

        assert "test_transformations" in pr1_section, (
            "PR 1 plan does not mention test_transformations.py. "
            "The test file imports via the direct file path "
            "(views_pipeline_core.modules.transformations.transformations) "
            "which bypasses the shim. The plan must either list the test "
            "file in 'Files Changed' or note the import update needed."
        )


# ---------------------------------------------------------------------------
# F-2 (SOFT): Private attribute coupling across package boundary
# ---------------------------------------------------------------------------

class TestF2_PrivateAttributeCoupling:
    """
    transformations.py:119-120 accesses dataset._time_id and
    dataset._entity_id — private attributes on _ViewsDataset.
    After extraction, this is cross-package access to private attrs.

    Fix: either document this as an accepted coupling (in the PR plan
    or a CIC), or add public accessors to _ViewsDataset.
    """

    @pytest.mark.skip(reason="RETIRED (#183, 2026-07-24): transformations.py no longer exists anywhere (upstream deleted it, their #119; shim removed here) — the probe's subject is gone. The private-attribute coupling concern lives on as register C-135.")
    def test_pr_plan_acknowledges_private_attr_coupling(self):
        """
        PR 1 plan must acknowledge that transformations.py accesses
        private attributes (_time_id, _entity_id) across the package
        boundary, or the source file must use public accessors.
        """
        plan_path = Path(
            "reports/views_reporting_extraction/extraction_pr_plans.md"
        )
        content = plan_path.read_text()
        pr1_section = content.split("## PR 1:")[1].split("## PR 2:")[0]

        source_path = Path(
            "../views-reporting/views_reporting/transformations/transformations.py"
        )
        if not source_path.exists():
            source_path = Path(
                "views_pipeline_core/modules/transformations/transformations.py"
            )
        assert source_path.exists(), (
            f"transformations.py not found at views-reporting or "
            f"pipeline-core path: {source_path}"
        )
        source = source_path.read_text()

        uses_private = "_time_id" in source and "_entity_id" in source
        plan_mentions = (
            "_time_id" in pr1_section
            or "_entity_id" in pr1_section
            or "private attr" in pr1_section.lower()
        )

        assert not uses_private or plan_mentions, (
            "transformations.py accesses _time_id and _entity_id (private "
            "attributes on _ViewsDataset). After extraction to views-reporting, "
            "this becomes cross-package private attribute access. The PR plan "
            "should acknowledge this coupling."
        )


# ---------------------------------------------------------------------------
# F-3 (SOFT): README.md not in the plan
# ---------------------------------------------------------------------------

class TestF3_ReadmeOrphaned:
    """
    modules/transformations/README.md (10 KB) is not mentioned in PR 1's
    "Files Changed" table. After extraction, the README describes a module
    that has been deleted, with import paths pointing to the old location.
    """

    def test_pr_plan_mentions_readme(self):
        """PR 1 plan must account for the README.md file."""
        plan_path = Path(
            "reports/views_reporting_extraction/extraction_pr_plans.md"
        )
        content = plan_path.read_text()
        pr1_section = content.split("## PR 1:")[1].split("## PR 2:")[0]

        readme_exists = Path(
            "views_pipeline_core/modules/transformations/README.md"
        ).exists()

        assert not readme_exists or "README" in pr1_section, (
            "modules/transformations/README.md exists (10 KB) but is not "
            "mentioned in PR 1's plan. After extraction, the README "
            "describes a deleted module with stale import paths. "
            "The plan should either move it, delete it, or update it."
        )
