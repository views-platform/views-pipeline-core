"""
Falsification audit: "PR #167 is safe to merge"
Generated: 2026-06-05

F2 (RESOLVED): validate_output_scale_consistency() previously crashed if
get_meta_config() returned None. Fixed by adding isinstance(meta, dict)
guard before meta.get("output_scale").
"""

import types
from unittest.mock import patch, MagicMock


def test_f2_get_meta_config_returns_none_crashes():
    """
    F2: If a model's get_meta_config() returns None instead of a dict,
    validate_output_scale_consistency() must skip gracefully instead of
    crashing with AttributeError.
    """
    from views_pipeline_core.modules.validation.ensemble.check import (
        validate_output_scale_consistency,
    )

    def _make_mock_mp(name):
        mp = MagicMock()
        config_path = MagicMock()
        config_path.exists.return_value = True
        config_path.__str__ = lambda self: f"/fake/models/{name}/configs/config_meta.py"
        mp.configs.__truediv__ = lambda self, key: config_path
        return mp

    def _make_module_returning_none():
        mod = types.ModuleType("fake")
        mod.get_meta_config = lambda: None
        return mod

    with patch(
        "views_pipeline_core.managers.model.path.ModelPathManager"
    ) as MockMPM:
        MockMPM.side_effect = lambda name: _make_mock_mp(name)

        with patch("importlib.util.spec_from_file_location") as mock_spec:
            mock_loader = MagicMock()
            mock_spec_obj = MagicMock()
            mock_spec_obj.loader = mock_loader
            mock_spec.return_value = mock_spec_obj

            with patch("importlib.util.module_from_spec") as mock_mod:
                mod = _make_module_returning_none()
                mock_mod.return_value = mod

                # This should NOT raise — it should skip gracefully
                # Currently it raises AttributeError: 'NoneType' has no attribute 'get'
                validate_output_scale_consistency(["model_a", "model_b"])