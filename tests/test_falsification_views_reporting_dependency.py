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
    P2: ORIGINALLY — the reconciliation hard imports of `views_reporting.reconciliation`
    were guarded by `CoreConfigSniffer`'s `find_spec("views_reporting")` check.

    RESOLVED by #195: reconciliation no longer imports views-reporting at all (DIP
    injection — the ensemble managers call an injected `Reconciler` port + a
    views-frames-only adapter). The sniffer's find_spec guard was removed; pipeline-core
    has zero `views_reporting.reconciliation` references. This stub is retained as a
    historical record of the resolved coupling.
    """

    @pytest.mark.skip(reason="Resolved by #195 — reconciliation decoupled from views-reporting")
    def test_sniffer_blocks_reconciliation_when_views_reporting_absent(self):
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

    # ENFORCED, as of #375: the two stubs that used to sit here named this rule, cited
    # ADR-054, and asserted nothing — `@pytest.mark.skip` with a body of `pass`. They
    # reported the constraint as covered while permitting every way of breaking it.
    #
    # `tests/test_reporting_is_not_a_dependency.py` now checks it for real: no declared
    # pin (in dependencies, extras or groups), no module-scope `views_reporting` import
    # anywhere under the package (by AST), the ADR still stating the rule, and the
    # minimal CI job still uninstalling views-reporting. All four are mutation-tested.
    #
    # The stubs are deleted rather than converted: a skipped test that duplicates a real
    # one is worse than no test, because the summary line counts it as coverage.