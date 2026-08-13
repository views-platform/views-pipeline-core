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


def test_the_model_template_names_every_key_the_sniffer_requires():
    """A scaffolded model must at least *mention* the keys it cannot run without.

    The template deliberately emits an incomplete config — `level` and `creator` are
    commented out for the author to fill in — so this does not check that a generated
    config passes `CoreConfigSniffer`. It checks something weaker and more useful: that a
    model author sees every mandatory key, rather than discovering two of them from a
    `KeyError` on their first run.

    `prediction_format` and `rolling_origin_stride` were absent entirely until #462. PR
    #328 noticed and added them, uncommented; they are now commented and marked REQUIRED,
    consistent with the other mandatory keys.

    **Derived from `MANDATORY_KEYS_MODEL`**, not listed here. Adding a mandatory key
    without mentioning it in the scaffold fails this test — which is the only thing that
    would say so, since the template is generated code and nothing else reads it.
    """
    import tempfile
    from pathlib import Path

    from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer
    from views_pipeline_core.templates.model import template_config_meta

    destination = Path(tempfile.mkdtemp()) / "config_meta.py"
    template_config_meta.generate(destination, "some_model", "SomeAlgorithm")
    generated = destination.read_text()

    required = list(CoreConfigSniffer.MANDATORY_KEYS_UNIVERSAL) + list(
        CoreConfigSniffer.MANDATORY_KEYS_MODEL
    )
    # `steps` and `time_steps` live in config_hyperparameters.py, not here.
    elsewhere = {"steps", "time_steps", "deployment_status"}

    missing = [k for k in required if k not in elsewhere and f'"{k}"' not in generated]
    assert not missing, (
        f"the model scaffold never mentions mandatory key(s) {missing}. An author "
        f"following the template discovers them from a KeyError on their first run. "
        f"Add them commented, marked REQUIRED, alongside `level` and `creator` — or, if "
        f"they genuinely belong in another config file, add them to the exclusion set "
        f"here with a reason."
    )
