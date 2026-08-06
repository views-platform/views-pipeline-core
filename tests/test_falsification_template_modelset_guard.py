"""
Regression guard for config_modelset template correctness.

Audit: Merge-safety falsification (2026-06-05)
Finding: P3 (SOFT FALSIFICATION) — template_config_meta.py lost "models" key,
         template_config_modelset.py has it. Cross-repo scaffolding in views-models
         depends on BOTH templates being correct. views-models issue #84 tracks wiring.

These tests ensure the templates in THIS repo stay correct so the companion
migration can rely on them.
"""

import tempfile
from pathlib import Path


def test_template_config_meta_does_not_generate_models_key():
    """models key belongs in config_modelset, not config_meta."""
    from views_pipeline_core.templates.ensemble.template_config_meta import generate

    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "config_meta.py"
        generate(script_path, "test_ensemble")
        source = script_path.read_text()
        assert '"models"' not in source, (
            "template_config_meta must not generate 'models' key — "
            "it was moved to template_config_modelset (PR #163)"
        )


def test_template_config_modelset_generates_models_key():
    """config_modelset template must provide the models key."""
    from views_pipeline_core.templates.ensemble.template_config_modelset import (
        generate,
    )

    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "config_modelset.py"
        generate(script_path, "test_ensemble")
        source = script_path.read_text()
        assert '"models"' in source, (
            "template_config_modelset must generate 'models' key"
        )


def test_template_config_modelset_function_name():
    """Scaffold and managers expect get_modelset_config as the entry point."""
    from views_pipeline_core.templates.ensemble.template_config_modelset import (
        generate,
    )

    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "config_modelset.py"
        generate(script_path, "test_ensemble")
        source = script_path.read_text()
        assert "def get_modelset_config():" in source, (
            "entry point must be get_modelset_config — managers use this name"
        )


def test_template_config_meta_model_generates_output_scale_key():
    """Model template must include output_scale for C-158 ensemble validation."""
    from views_pipeline_core.templates.model.template_config_meta import generate

    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "config_meta.py"
        generate(script_path, "test_model", "TestAlgo")
        source = script_path.read_text()
        assert "output_scale" in source, (
            "model template must include output_scale key (commented or uncommented) "
            "for ensemble output scale consistency validation (C-158)"
        )
