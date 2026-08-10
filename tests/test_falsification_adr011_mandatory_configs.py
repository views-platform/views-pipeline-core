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
    """ADR-011 corrected: 4 base configs mandatory for all types.

    The maturity config is checked separately, below. Under ADR-057 it is mandatory under
    *either* of two filenames, so a source-text grep for one literal would now report a
    model that had finished renaming as non-compliant.
    """
    from views_pipeline_core.data.model_path import ModelPathManager

    source = inspect.getsource(ModelPathManager._initialize_scripts)
    for cfg in [
        "config_hyperparameters.py",
        "config_meta.py",
        "config_partitions.py",
    ]:
        assert cfg in source, f"{cfg} should be in base _initialize_scripts"

    assert "_resolve_maturity_config_path" in source, (
        "the base initializer no longer requires a maturity config at all. ADR-011 makes "
        "it one of the four mandatory base configs; ADR-057 only changed what it may be "
        "called, not whether it is required."
    )


def _resolver_for(tmp_path, validate=True):
    """A ModelPathManager stub with just enough state to exercise the resolver.

    Built with `object.__new__` deliberately: a real construction validates the whole
    model tree, which would make this a test of the fixture rather than of the rule.
    """
    from views_pipeline_core.data.model_path import ModelPathManager

    manager = object.__new__(ModelPathManager)
    manager.model_dir = tmp_path
    manager._validate = validate
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    return manager


def test_the_maturity_config_is_mandatory_under_either_name(tmp_path):
    """ADR-057: renaming the file must not make a compliant model non-compliant."""
    for filename in ("config_deployment.py", "config_maturity.py"):
        directory = tmp_path / filename.replace(".py", "")
        manager = _resolver_for(directory)
        (directory / "configs" / filename).write_text("")
        resolved = manager._resolve_maturity_config_path()
        assert resolved.name == filename, (
            f"a model carrying only {filename} did not resolve to it (got {resolved.name})"
        )


def test_the_new_name_wins_when_both_exist(tmp_path):
    """Must match the config loader's precedence, or the reported file is not the read one."""
    manager = _resolver_for(tmp_path)
    (tmp_path / "configs" / "config_deployment.py").write_text("")
    (tmp_path / "configs" / "config_maturity.py").write_text("")
    assert manager._resolve_maturity_config_path().name == "config_maturity.py"


def test_neither_name_present_fails_and_names_both(tmp_path):
    """The failure must not send a renamed model back to re-create the file it deleted."""
    import pytest

    manager = _resolver_for(tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        manager._resolve_maturity_config_path()

    message = str(excinfo.value)
    assert "config_maturity.py" in message and "config_deployment.py" in message, (
        f"the failure names only one acceptable filename: {message}"
    )


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
