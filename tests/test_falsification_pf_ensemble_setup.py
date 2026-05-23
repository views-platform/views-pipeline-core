"""
Falsification audit: PF ensemble setup path (2026-05-23).

F-2: Ensemble template does not reference PredictionFrameEnsembleManager,
     making PF ensemble setup undiscoverable from scaffolding.
"""
import inspect


class TestF2TemplateDiscoverability:
    """The ensemble template should reference PredictionFrameEnsembleManager
    so users can discover PF ensemble setup from the generated scaffold."""

    def test_template_main_mentions_pf_ensemble_manager(self):
        """Template main.py must reference PredictionFrameEnsembleManager
        so users know it exists as an alternative to EnsembleManager."""
        from views_pipeline_core.templates.ensemble.template_main import generate

        source = inspect.getsource(generate)
        assert "PredictionFrameEnsembleManager" in source, (
            "Ensemble template main.py does not reference "
            "PredictionFrameEnsembleManager. Users following the template "
            "will always get EnsembleManager and cannot discover the PF "
            "ensemble path without reading source code."
        )

    def test_template_config_includes_prediction_format(self):
        """Template config_meta should include prediction_format key
        so users know it's a required decision point."""
        from views_pipeline_core.templates.ensemble.template_config_meta import generate

        source = inspect.getsource(generate)
        assert "prediction_format" in source, (
            "Ensemble template config_meta does not include prediction_format. "
            "Users cannot discover the PF/DF choice from the template."
        )
