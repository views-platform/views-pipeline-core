"""
Failing test stubs from falsification audit of the claim:
  "views-reporting is an undeclared hard dependency — stage.py does
   a hard import not behind try/except, so it's not designed to be optional."

Verdict: CONTESTED (4 soft falsifications, 1 hard falsification).

Each stub encodes a falsifying observation from the audit.
"""
import pytest


class TestSoftF1_ReportingStageImportableWithoutViewsReporting:
    """
    P1: The claim says 'hard import (not behind try/except)' but the imports
    are method-level deferred. ReportingStage can be imported AND instantiated
    without views-reporting. The crash is method-call-time, not import-time.

    Severity: Soft falsification — the claim's framing of 'hard import'
    overstates the coupling. It IS a runtime dependency, but lazy-loaded.
    """

    @pytest.mark.skip(reason="Falsification stub — observation, not a defect to fix")
    def test_reporting_stage_importable_without_views_reporting(self):
        # If views-reporting were truly a hard dependency, this import would fail.
        # But it succeeds because the views_reporting imports are inside methods.
        from views_pipeline_core.managers.reporting.stage import ReportingStage
        assert ReportingStage is not None

    @pytest.mark.skip(reason="Falsification stub — observation, not a defect to fix")
    def test_reporting_stage_instantiable_without_views_reporting(self):
        from views_pipeline_core.managers.reporting.stage import ReportingStage
        from unittest.mock import MagicMock
        stage = ReportingStage(wandb_module=MagicMock(), wandb_notifications=False)
        assert stage is not None


class TestSoftF2_ReconciliationGuardedByCoreConfigSniffer:
    """
    P2: The reconciliation hard imports (ensemble.py:733, dataframe_ensemble.py:909)
    are architecturally unreachable when views-reporting is absent, because
    CoreConfigSniffer.sniff_all() fires first and raises ImportError via find_spec.

    Severity: Soft falsification — claim treats reconciliation imports as unguarded.
    """

    @pytest.mark.skip(reason="Falsification stub — observation, not a defect to fix")
    def test_sniffer_blocks_reconciliation_when_views_reporting_absent(self):
        # CoreConfigSniffer._check_reconciliation_config() at line 319 does:
        #   if importlib.util.find_spec("views_reporting") is None: raise ImportError
        # This fires BEFORE EnsembleManager._apply_reconciliation() → __reconcile_pg_with_c()
        # So the hard import at ensemble.py:733 is never reached.
        pass


class TestSoftF3_ReportingGatedByCliFlag:
    """
    P3: All reporting methods are gated by `args.report` CLI flag. Without --report,
    the deferred views_reporting imports in stage.py are never executed.

    Call sites:
      model.py:1098  — if self.args.report and self.args.forecast
      model.py:1100  — if self.args.report and self.args.evaluate
      ensemble.py:184-187  — same pattern
      prediction_frame_ensemble.py:291-294  — same pattern (ctx.args.report)

    Severity: Soft falsification — the dependency is opt-in, not automatic.
    """

    @pytest.mark.skip(reason="Falsification stub — observation, not a defect to fix")
    def test_reporting_not_called_without_report_flag(self):
        # Without --report, generate_forecast_report/generate_evaluation_report
        # are never called. The views_reporting imports never execute.
        pass


class TestSoftF4_MajorityDesignPostureIsOptional:
    """
    P4: 6 out of 10 views_reporting import sites use try/except ImportError
    (explicitly optional). The 4 'hard' sites are all gated by runtime guards.
    The claim 'not designed to be optional' contradicts the observed 6/10 ratio.

    Severity: Soft falsification — the package's dominant architecture IS optional.
    """

    @pytest.mark.skip(reason="Falsification stub — observation, not a defect to fix")
    def test_majority_import_sites_are_optional(self):
        # 6 re-export shims: mapping, reconciliation, reports, statistics,
        #   transformations, visualizations — all try/except ImportError
        # 4 deferred hard: stage.py:72, stage.py:145, ensemble.py:733,
        #   dataframe_ensemble.py:909 — all behind runtime guards
        pass


class TestHardF5_ADR054DeclaresIntentionalNonDependency:
    """
    P5: ADR-054 line 80 explicitly states:
      'Pipeline-core NEVER depends on views-reporting.'
    The CI workflow 'run_pytest_minimal.yml' actively tests without views-reporting.
    The omission from pyproject.toml is a documented architectural decision.
    Calling it 'a bug' mischaracterizes a conscious governance choice.

    C-144 tracks the tension; Issue #130 proposes declaring as optional extra.
    But the current state is deliberate, not an oversight.

    Severity: Hard falsification — documentation directly contradicts the 'bug' framing.
    """

    @pytest.mark.skip(reason="Falsification stub — ADR-054 governs this decision")
    def test_adr054_declares_no_dependency_on_views_reporting(self):
        # ADR-054 line 80: "Pipeline-core NEVER depends on views-reporting."
        # This is a deliberate architectural decision, not a bug.
        pass

    @pytest.mark.skip(reason="Falsification stub — CI proves intentional support")
    def test_ci_explicitly_tests_without_views_reporting(self):
        # .github/workflows/run_pytest_minimal.yml:
        #   "pip uninstall -y views-reporting ..."
        #   "Run core-only tests (without views-reporting)"
        pass
