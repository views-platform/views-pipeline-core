"""
Tests for ensemble-level output_scale consistency validation (C-158).

Verifies that validate_output_scale_consistency() catches mixed output scales
across ensemble constituent models before predictions are aggregated.
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from views_pipeline_core.modules.validation.ensemble.check import (
    validate_output_scale_consistency,
)


def _write_config_meta(directory: Path, output_scale=None):
    """Write a minimal config_meta.py with optional output_scale."""
    configs_dir = directory / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "def get_meta_config():",
        "    return {",
        '        "name": "test_model",',
    ]
    if output_scale is not None:
        lines.append(f'        "output_scale": "{output_scale}",')
    lines.append("    }")
    (configs_dir / "config_meta.py").write_text("\n".join(lines))
    return configs_dir


def _mock_path_manager(model_configs):
    """Return a side_effect for ModelPathManager that maps names to temp dirs."""
    def _side_effect(name):
        mp = MagicMock()
        mp.configs = model_configs[name]
        return mp
    return _side_effect


class TestOutputScaleConsistency:

    def test_all_models_same_scale_passes(self, tmp_path):
        configs = {}
        for name in ["model_a", "model_b", "model_c"]:
            d = tmp_path / name
            configs[name] = _write_config_meta(d, output_scale="log")

        with patch(
            "views_pipeline_core.managers.model.path.ModelPathManager",
            side_effect=_mock_path_manager(configs),
        ):
            validate_output_scale_consistency(["model_a", "model_b", "model_c"])

    def test_models_different_scales_raises(self, tmp_path):
        configs = {}
        configs["model_a"] = _write_config_meta(tmp_path / "model_a", output_scale="log")
        configs["model_b"] = _write_config_meta(tmp_path / "model_b", output_scale="natural")

        with patch(
            "views_pipeline_core.managers.model.path.ModelPathManager",
            side_effect=_mock_path_manager(configs),
        ):
            with pytest.raises(ValueError, match="output_scale mismatch"):
                validate_output_scale_consistency(["model_a", "model_b"])

    def test_mixed_declared_undeclared_warns(self, tmp_path):
        configs = {}
        configs["model_a"] = _write_config_meta(tmp_path / "model_a", output_scale="log")
        configs["model_b"] = _write_config_meta(tmp_path / "model_b", output_scale=None)

        with patch(
            "views_pipeline_core.managers.model.path.ModelPathManager",
            side_effect=_mock_path_manager(configs),
        ), patch(
            "views_pipeline_core.modules.validation.ensemble.check.logger"
        ) as mock_logger:
            validate_output_scale_consistency(["model_a", "model_b"])
            mock_logger.warning.assert_called_once()
            assert "partially declared" in mock_logger.warning.call_args[0][0]

    def test_no_models_declare_passes_silently(self, tmp_path):
        configs = {}
        configs["model_a"] = _write_config_meta(tmp_path / "model_a", output_scale=None)
        configs["model_b"] = _write_config_meta(tmp_path / "model_b", output_scale=None)

        with patch(
            "views_pipeline_core.managers.model.path.ModelPathManager",
            side_effect=_mock_path_manager(configs),
        ), patch(
            "views_pipeline_core.modules.validation.ensemble.check.logger"
        ) as mock_logger:
            validate_output_scale_consistency(["model_a", "model_b"])
            mock_logger.warning.assert_not_called()