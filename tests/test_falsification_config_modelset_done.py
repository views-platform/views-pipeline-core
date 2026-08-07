"""
Falsification regression tests for config_modelset extraction (PR #163).

Originally generated as xfail stubs by falsification audit 2026-06-05.
C-150 fix made config_modelset.py truly optional — these now pass as
regression tests ensuring the fix holds.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── P2: Regression test (was HARD FALSIFICATION, fixed by C-150) ──────────
# config_modelset.py is optional per CICs (§4). Construction must
# succeed when the file is absent and validate=True.


def test_ensemble_path_construction_without_config_modelset():
    """
    EnsemblePathManager("some_ensemble") must succeed even when
    configs/config_modelset.py does not exist on disk.
    """
    from views_pipeline_core.managers.ensemble import EnsemblePathManager

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "views_pipeline"
        project_root.mkdir()
        (project_root / "LICENSE.md").touch()
        ens_dir = project_root / "ensembles" / "test_ensemble"
        (ens_dir / "configs").mkdir(parents=True)
        (ens_dir / "artifacts").mkdir()
        (ens_dir / "data" / "generated").mkdir(parents=True)
        (ens_dir / "data" / "processed").mkdir(parents=True)
        (ens_dir / "data" / "raw").mkdir(parents=True)
        (ens_dir / "reports").mkdir()
        (ens_dir / "notebooks").mkdir()

        for f in [
            "config_deployment.py",
            "config_hyperparameters.py",
            "config_meta.py",
            "config_partitions.py",
        ]:
            (ens_dir / "configs" / f).touch()
        (ens_dir / "main.py").touch()
        (ens_dir / "README.md").touch()

        # config_modelset.py intentionally NOT created
        with patch.object(EnsemblePathManager, "_root", project_root), \
             patch.object(EnsemblePathManager, "get_models", return_value=project_root / "ensembles"), \
             patch("views_pipeline_core.managers.ensemble.EnsemblePathManager._get_model_dir", return_value=ens_dir):
            path_mgr = EnsemblePathManager("test_ensemble", validate=True)
            assert path_mgr.model_name == "test_ensemble"


# ── BONUS: Regression test (was HARD FALSIFICATION, fixed by C-150) ───────
# _initialize_ensemble_specific_scripts no longer calls
# _build_absolute_directory for config_modelset.py, so validate=True
# does not crash on missing optional files.


def test_validate_true_does_not_crash_on_missing_config_modelset():
    """
    _initialize_ensemble_specific_scripts must not call
    _build_absolute_directory for config_modelset.py — it checks
    existence directly and skips when absent.
    """
    from views_pipeline_core.data.model_path import ModelPathManager

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "ensembles" / "test_ensemble"
        (model_dir / "configs").mkdir(parents=True)
        # config_modelset.py NOT created

        mgr = MagicMock(spec=ModelPathManager)
        mgr._validate = True
        mgr.model_dir = model_dir
        mgr.scripts = []

        ModelPathManager._initialize_ensemble_specific_scripts(mgr)
        assert len(mgr.scripts) == 0


# ── P5: Moot after C-150 fix ─────────────────────────────────────────────
# config_modelset.py is truly optional now. No migration needed —
# existing ensembles work without it. CICs already document the
# optional behavior.


@pytest.mark.skip(
    reason="P5: Migration docs moot after C-150 fix — config_modelset.py "
    "is truly optional. Existing ensembles work without it."
)
def test_migration_documentation_exists():
    """Retained for audit trail. No longer actionable."""
    pass