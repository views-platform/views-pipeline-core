"""
Falsification tests for ADR-011 "Mandatory vs Optional Configurations" claims.

Audit: C-151/C-153/C-154/C-155 foundational fix audit
Probe: P2 — Contract (ADR-011 mandatory config claim)
Original verdict: HARD FALSIFICATION (C-156)
Resolution: ADR-011 corrected (2026-06-05)

These tests now verify the CORRECTED claims:
  - Base mandatory set is 4 files (deployment, hyperparameters, meta, partitions)
  - config_queryset.py and config_sweep.py are model-only
  - config_modelset.py is ensemble-only and optional
"""

import inspect


def test_base_mandatory_configs_are_four():
    """ADR-011 corrected: 4 base configs mandatory for all types."""
    from views_pipeline_core.data.model_path import ModelPathManager

    source = inspect.getsource(ModelPathManager._initialize_scripts)
    for cfg in [
        "config_deployment.py",
        "config_hyperparameters.py",
        "config_meta.py",
        "config_partitions.py",
    ]:
        assert cfg in source, f"{cfg} should be in base _initialize_scripts"


def test_config_queryset_is_model_only():
    """ADR-011 corrected: config_queryset.py is model-only, not base."""
    from views_pipeline_core.data.model_path import ModelPathManager

    base_source = inspect.getsource(ModelPathManager._initialize_scripts)
    model_source = inspect.getsource(
        ModelPathManager._initialize_model_specific_scripts
    )
    assert "config_queryset.py" not in base_source
    assert "config_queryset.py" in model_source


def test_config_sweep_is_model_only():
    """ADR-011 corrected: config_sweep.py is model-only, not base."""
    from views_pipeline_core.data.model_path import ModelPathManager

    base_source = inspect.getsource(ModelPathManager._initialize_scripts)
    model_source = inspect.getsource(
        ModelPathManager._initialize_model_specific_scripts
    )
    assert "config_sweep.py" not in base_source
    assert "config_sweep.py" in model_source


def test_config_inputdata_stale_name_absent():
    """ADR-011 corrected: config_inputdata.py no longer referenced."""
    from views_pipeline_core.data.model_path import ModelPathManager

    all_source = (
        inspect.getsource(ModelPathManager._initialize_scripts)
        + inspect.getsource(
            ModelPathManager._initialize_model_specific_scripts
        )
        + inspect.getsource(
            ModelPathManager._initialize_ensemble_specific_scripts
        )
    )
    assert "config_inputdata.py" not in all_source