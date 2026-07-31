"""Import-purity guards — architectural rules asserted in a subprocess, not promised.

Two rules live here:

1. **pandas is a LEGACY option, never a foundational import** (#320, C-225). Importing
   the base manager — the class every engine must extend — must not load pandas.
2. **The delivery path must not be able to provision** (#331/#332, C-233). Creating a
   bucket, database or collection is a deliberate act performed by a person; if
   ``views_pipeline_core.modules.appwrite.provisioning`` is reachable from the code that
   publishes a forecast, the least-privilege key the platform is moving to cannot be
   issued. The dependency runs one way — ``provisioning`` imports ``file``, never the
   reverse — and that is what these probes check.

The frame-native goal (epic #300) requires that importing the base manager —
the class every engine must extend — does not load pandas. These tests are the
permanent architectural tripwire: a future top-level import that re-couples the
frame path to the legacy DataFrame tier turns them red in every CI job.

All three probes run in a SUBPROCESS: the pytest process itself has pandas
loaded (fixtures, other suites), so in-process assertions would be meaningless.
They assert pandas is NOT loaded, so they need pandas installed only in the
sense that any environment qualifies — no env gate, no importorskip.
"""
import subprocess
import sys

PROBE = (
    "import sys; {imports}; "
    "loaded = sorted(m for m in sys.modules if m == 'pandas' or m.startswith('pandas.')); "
    "assert not loaded, f'pandas loaded by {{__name__}} chain: {{loaded[:3]}}'"
)


def _run_probe(imports: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", PROBE.format(imports=imports)],
        capture_output=True,
        text=True,
    )


def test_bare_package_import_is_pandas_free():
    """`import views_pipeline_core` alone must never load pandas (true pre-#320; pinned)."""
    result = _run_probe("import views_pipeline_core")
    assert result.returncode == 0, result.stderr


def test_manager_import_is_pandas_free():
    """Importing ForecastingModelManager must not load pandas (#320 acceptance).

    This is the C-225 guard: the base manager's transitive import closure is
    the floor under every engine process. If this turns red, some module on
    the chain regained a top-level pandas import (or `managers/__init__.py`
    reverted to eager fan-out) — fix the import, do not gate this test.
    """
    result = _run_probe(
        "from views_pipeline_core.managers.model import ForecastingModelManager"
    )
    assert result.returncode == 0, result.stderr


def test_manager_package_import_is_pandas_free():
    """`import views_pipeline_core.managers` (the lazy PEP 562 facade) stays pandas-free."""
    result = _run_probe("import views_pipeline_core.managers")
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Provisioning purity (#331/#332, C-233)
# ---------------------------------------------------------------------------

_PROVISIONING = "views_pipeline_core.modules.appwrite.provisioning"

FORBIDDEN_PROBE = (
    "import sys; {imports}; "
    "assert '{forbidden}' not in sys.modules, "
    "'{forbidden} was imported by the delivery path'"
)


def _run_forbidden_probe(imports: str, forbidden: str = _PROVISIONING):
    return subprocess.run(
        [sys.executable, "-c", FORBIDDEN_PROBE.format(imports=imports, forbidden=forbidden)],
        capture_output=True,
        text=True,
    )


def test_storage_module_does_not_import_provisioning():
    """`file.py` is the delivery path's storage surface; it must not reach provisioning."""
    result = _run_forbidden_probe(
        "import views_pipeline_core.modules.appwrite.file"
    )
    assert result.returncode == 0, result.stderr


def test_datastore_does_not_import_provisioning():
    """`DatastoreModule` is what the savers call — the closest caller to a real upload."""
    result = _run_forbidden_probe(
        "from views_pipeline_core.modules.datastore import DatastoreModule"
    )
    assert result.returncode == 0, result.stderr


def test_appwrite_package_does_not_import_provisioning():
    """The package `__init__` must not re-export it into the delivery path either."""
    result = _run_forbidden_probe("import views_pipeline_core.modules.appwrite")
    assert result.returncode == 0, result.stderr


def test_savers_do_not_import_provisioning():
    """The publish path end to end: AppwriteSaver -> DatastoreModule -> file.py."""
    result = _run_forbidden_probe(
        "from views_pipeline_core.managers.prediction.savers import AppwriteSaver"
    )
    assert result.returncode == 0, result.stderr


def test_provisioning_is_importable_on_its_own():
    """The rule is one-way, not a ban: the setup entrypoint must still work."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {_PROVISIONING} as p; assert p.AppwriteProvisioner"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
