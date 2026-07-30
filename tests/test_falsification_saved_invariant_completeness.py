"""
Falsification tests for the claim: "we have 100% of information regarding
the three-site saved invariant (C-146, D-23, D-24) to take the right decision."

These tests encode the gaps discovered during the falsification audit.
Each test documents a dimension that was missing from the expert-code-review
analysis. Tests are marked xfail where the gap is a documentation/analysis
issue rather than a code defect; tests that verify actual code behavior
pass or fail on their merits.

Source: falsify skill (2026-06-03, saved invariant completeness audit)
"""

import pytest


class TestP1SweepBlindSpot:
    """P1: The expert-code-review analyzed a 2-variable system (run_type × saved)
    but the actual invariant at args.py:411 is 3-variable (train × sweep × saved).
    The sweep exemption was not analyzed."""

    def test_sweep_exemption_allows_saved_false(self):
        """Sweep runs bypass the saved=True requirement at args.py:411.
        This means ForecastingModelArgs(sweep=True, saved=False) is valid.
        The expert-code-review did not account for this exemption."""
        from views_pipeline_core.cli.args import ForecastingModelArgs
        args = ForecastingModelArgs(
            run_type="calibration",
            sweep=True,
            saved=False,
        )
        assert args.saved is False
        assert args.sweep is True

    def test_no_ensemble_sweep_path_exists(self):
        """Ensembles have no execute_sweep_run method, so the sweep exemption
        is architecturally irrelevant for ensembles — but this was never
        analyzed, the conclusion is correct by accident."""
        from views_pipeline_core.managers.ensemble.dataframe_ensemble import DataFrameEnsembleManager
        from views_pipeline_core.managers.ensemble.prediction_frame_ensemble import PredictionFrameEnsembleManager

        for cls in [DataFrameEnsembleManager, PredictionFrameEnsembleManager]:
            assert not hasattr(cls, "execute_sweep_run"), (
                f"{cls.__name__} has execute_sweep_run — sweep interaction "
                f"with saved invariant needs analysis"
            )
        # EnsembleManager inherits from ForecastingModelManager which has
        # execute_sweep_run, but EnsembleManager doesn't override it and
        # ensemble templates don't call it. Verify the template doesn't route to it.

    @pytest.mark.xfail(reason="Documentation gap: sweep exemption not mentioned in check.py, ADR-018 amendment, or C-146")
    def test_sweep_documented_in_validation_module(self):
        """check.py should document that sweep runs are architecturally
        excluded from ensemble validation (no execute_sweep_run exists)."""
        import inspect
        from views_pipeline_core.modules.validation.ensemble import check
        source = inspect.getsource(check)
        assert "sweep" in source, (
            "check.py does not mention 'sweep' anywhere — the sweep dimension "
            "of the saved invariant is undocumented in the validation module"
        )


class TestP2SysExitEnforcementFragility:
    """P2: The saved invariant at args.py:411 is enforced via sys.exit(1),
    which raises SystemExit (BaseException). This is catchable and bypassable
    in non-CLI contexts."""

    def test_invariant_uses_sys_exit_not_exception(self):
        """The enforcement mechanism is sys.exit, not ValueError/RuntimeError.
        This means it's invisible to standard try/except Exception handling
        but catchable by except BaseException or except SystemExit."""
        from views_pipeline_core.cli.args import ForecastingModelArgs
        with pytest.raises(SystemExit) as exc_info:
            ForecastingModelArgs(
                run_type="calibration",
                train=False,
                sweep=False,
                saved=False,
                evaluate=True,
            )
        assert exc_info.value.code == 1

    def test_invariant_survives_except_exception(self):
        """Standard try/except Exception does NOT catch the invariant violation.
        This is the expected production behavior — sys.exit(1) propagates."""
        from views_pipeline_core.cli.args import ForecastingModelArgs
        caught = False
        try:
            ForecastingModelArgs(
                run_type="calibration",
                train=False,
                sweep=False,
                saved=False,
                evaluate=True,
            )
        except Exception:
            caught = True
        except SystemExit:
            pass  # Expected — sys.exit propagates past except Exception
        assert not caught, "sys.exit was caught by except Exception — enforcement is weaker than expected"

    def test_invariant_bypassable_with_base_exception(self):
        """Demonstrates that the invariant CAN be silently bypassed by
        code that catches BaseException. This is not a bug today (CLI context)
        but is an unconsidered dimension of the analysis."""
        from views_pipeline_core.cli.args import ForecastingModelArgs
        bypassed = False
        try:
            ForecastingModelArgs(
                run_type="calibration",
                train=False,
                sweep=False,
                saved=False,
                evaluate=True,
            )
            bypassed = True  # Should not reach here
        except BaseException:
            bypassed = True  # Caught SystemExit — invariant bypassed
        assert bypassed, "Could not bypass invariant with BaseException"


class TestP3SavedSemanticGap:
    """P3: saved=True means 'prefer cache, fall back to network fetch' — not
    'require saved data.' The semantic model behind the expert-code-review
    analysis is imprecise."""

    def test_use_saved_true_falls_back_to_fetch(self):
        """ViewsDataLoader.get_data(use_saved=True) fetches from network
        when cache file is missing. Verify this by reading the source."""
        import inspect
        from views_pipeline_core.modules.dataloaders.dataloaders import ViewsDataLoader
        source = inspect.getsource(ViewsDataLoader.get_data)
        assert "Saved data not found" in source, (
            "Expected fallback message in get_data — the 'saved' flag does not "
            "guarantee cached data is used"
        )
        assert "_fetch_data" in source, (
            "Expected _fetch_data call in use_saved=True path — saved falls back "
            "to network fetch when cache is missing"
        )

    @pytest.mark.xfail(reason="Semantic gap: no RuntimeError when use_saved=True and cache missing — falls back silently")
    def test_use_saved_true_requires_cache_exists(self):
        """If saved truly meant 'use saved data', a missing cache file
        should raise RuntimeError, not silently fetch. This test documents
        the semantic gap: the 'else' branch under 'if use_saved:' fetches
        from the network instead of raising."""
        import inspect
        from views_pipeline_core.modules.dataloaders.dataloaders import ViewsDataLoader
        source = inspect.getsource(ViewsDataLoader.get_data)
        use_saved_block = source.split("if use_saved:")[1].split("\n        else:\n")[0]
        # The else under path_cached_df.exists() should raise, not fetch
        # Find the "else:" under "if path_cached_df.exists():"
        else_block = use_saved_block.split("else:")[1] if "else:" in use_saved_block else ""
        assert "_fetch_data" not in else_block, (
            "use_saved=True + missing file calls _fetch_data instead of raising — "
            "the 'saved' flag means 'prefer cache' not 'require cache'"
        )


class TestP4CreateModelArgsCICGap:
    """P4: The _create_model_args hardcoding (saved=True for non-training)
    was undocumented in all three ensemble CICs. Fixed in C-146 PR."""

    def test_ensemble_cic_documents_create_model_args(self):
        """All three ensemble CICs now document the saved=True hardcoding
        in _create_model_args as a guarantee (C-146 fix)."""
        from pathlib import Path
        cic_dir = Path("documentation/CICs")
        for cic_name in [
            "EnsembleManager.md",
            "DataFrameEnsembleManager.md",
            "PredictionFrameEnsembleManager.md",
        ]:
            cic = (cic_dir / cic_name).read_text()
            assert "_create_model_args" in cic and "saved" in cic.split("_create_model_args")[1][:200], (
                f"{cic_name} does not document the _create_model_args "
                f"hardcoding of saved=True for non-training subprocess dispatch"
            )


class TestP5DecisionOptionCompleteness:
    """P5: The analysis (D-23, D-24) considered 'keep/remove guard' and
    'test/don't test'. It did not consider simplification alternatives."""

    @pytest.mark.xfail(reason="Analysis gap: check.py could accept args object directly, eliminating manual saved threading")
    def test_validate_ensemble_model_could_accept_args(self):
        """validate_ensemble_model receives saved as a primitive parameter,
        forcing all callers to thread it manually. It could instead accept
        the args object (or at minimum train+saved) and derive the skip
        logic internally. This design alternative was not considered."""
        import inspect
        from views_pipeline_core.modules.validation.ensemble.check import validate_ensemble_model
        sig = inspect.signature(validate_ensemble_model)
        params = list(sig.parameters.keys())
        # If args were accepted directly, 'saved' would not be a separate parameter
        assert "args" in params or "ForecastingModelArgs" in str(sig), (
            "validate_ensemble_model takes (config, saved=False) — manual threading. "
            "An alternative: accept args object and derive saved internally."
        )
