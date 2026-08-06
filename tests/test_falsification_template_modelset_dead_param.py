"""
Regression tests for model_name parameter in template_config_modelset.

Audit: template_config_modelset.py dead parameter (2026-06-05, C-157)
Originally falsified: model_name accepted but never interpolated.
Fixed: code string converted to f-string, model_name embedded in docstring.
"""

import tempfile
from pathlib import Path


def test_model_name_appears_in_generated_output():
    """model_name must appear in generated code (C-157 regression guard)."""
    from views_pipeline_core.templates.ensemble.template_config_modelset import (
        generate,
    )

    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "config_modelset.py"
        generate(script_path, "test_ensemble_alpha")
        source = script_path.read_text()
        assert "test_ensemble_alpha" in source


def test_different_model_names_produce_different_output():
    """Different model_name values must produce different output (C-157 regression guard)."""
    from views_pipeline_core.templates.ensemble.template_config_modelset import (
        generate,
    )

    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "a.py"
        p2 = Path(tmp) / "b.py"
        generate(p1, "alpha_ensemble")
        generate(p2, "completely_different_name")
        assert p1.read_text() != p2.read_text()
