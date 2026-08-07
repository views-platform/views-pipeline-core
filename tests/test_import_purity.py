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


# ---------------------------------------------------------------------------
# S5 (#345) — the vendor SDK must not be on the delivery path's import graph.
#
# C-253, measured: views-hydranet, views-baseline and views-evaluation contain ZERO
# references to Appwrite and all three install its SDK, because `appwrite` sits in
# `[tool.poetry.dependencies]` rather than in an extra. That is CRP violated, and SDP
# inverted — the platform's most-depended-upon package depends on a vendor SDK whose
# `databases.list*` surface deprecated at server 1.8.0.
#
# The DIP seam was already correct: `PredictionSaver` is a Protocol and `AppwriteSaver`
# implements it. Only the packaging was wrong — and the module DEFINING the Protocol
# imported the SDK at module scope, so making the dependency optional would have broken
# importing the Protocol itself (falsification finding F1, register C-253).
# ---------------------------------------------------------------------------

_APPWRITE = "appwrite"


def test_bare_package_import_does_not_load_the_appwrite_sdk():
    """Already true before #345 — pinned so the blast radius cannot grow."""
    result = _run_forbidden_probe("import views_pipeline_core", forbidden=_APPWRITE)
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_the_savers_module_does_not_load_the_appwrite_sdk():
    """F1's first half. `savers.py` defines the `PredictionSaver` Protocol AND the two
    local savers; a module-scope `from appwrite.exception import AppwriteException`
    meant an optional extra would break importing the Protocol."""
    result = _run_forbidden_probe(
        "import views_pipeline_core.managers.prediction.savers", forbidden=_APPWRITE
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_the_prediction_io_manager_does_not_load_the_appwrite_sdk():
    """F1's second half — `managers/prediction/io.py` carried the same eager import."""
    result = _run_forbidden_probe(
        "import views_pipeline_core.managers.prediction.io", forbidden=_APPWRITE
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_managers_facade_does_not_load_the_appwrite_sdk():
    result = _run_forbidden_probe(
        "import views_pipeline_core.managers.prediction", forbidden=_APPWRITE
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_local_savers_work_without_the_sdk_on_the_import_graph():
    """The point of the whole change: a repo that never touches Appwrite can save
    predictions. Constructing the local savers must not pull the vendor in."""
    result = _run_forbidden_probe(
        "from views_pipeline_core.managers.prediction.savers import "
        "NpzSaver, LocalParquetSaver, PredictionSaver; "
        "NpzSaver(); LocalParquetSaver()",
        forbidden=_APPWRITE,
    )
    assert result.returncode == 0, result.stdout + result.stderr


_WITHOUT_EXTRA = '''
import builtins, sys
_real = builtins.__import__
def _blocked(name, *a, **k):
    if name == "appwrite" or name.startswith("appwrite."):
        raise ImportError("No module named '%s'" % name)
    return _real(name, *a, **k)
builtins.__import__ = _blocked
for _m in [m for m in sys.modules if m.startswith("appwrite")]:
    del sys.modules[_m]
{body}
'''


def _run_without_the_extra(body: str) -> subprocess.CompletedProcess:
    """Run a probe in an interpreter where `import appwrite` always fails.

    The extra IS installed in this environment, so a test that merely checks the SDK is
    absent from `sys.modules` cannot tell "not imported" from "not installed". Blocking
    the import is what makes the no-extra path genuinely exercised rather than assumed —
    the same reason `test_appwrite_pagination.py` builds its double from the SDK's real
    query encoding instead of from belief (C-218).
    """
    return subprocess.run(
        [sys.executable, "-c", _WITHOUT_EXTRA.format(body=body)],
        capture_output=True,
        text=True,
    )


def test_the_package_is_usable_with_the_extra_uninstalled():
    """The whole point of #345: a repo that never touches Appwrite can still save."""
    result = _run_without_the_extra(
        "import views_pipeline_core\n"
        "from views_pipeline_core.managers.prediction.savers import ("
        "    NpzSaver, LocalParquetSaver, PredictionSaver)\n"
        "assert isinstance(NpzSaver(), PredictionSaver)\n"
        "assert isinstance(LocalParquetSaver(), PredictionSaver)\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_fault_resolver_degrades_to_stdlib_types_without_the_extra():
    result = _run_without_the_extra(
        "from views_pipeline_core.managers.prediction.vendor_faults import "
        "upload_transport_faults\n"
        "faults = upload_transport_faults()\n"
        "assert faults == (ConnectionError, TimeoutError, OSError), faults\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_asking_for_appwrite_without_the_extra_names_the_install_command():
    """A bare `ModuleNotFoundError` six frames inside file.py tells the operator
    nothing. Follows the `_require_dense_report_consumer` idiom."""
    result = _run_without_the_extra(
        "try:\n"
        "    import views_pipeline_core.modules.appwrite\n"
        "    raise SystemExit('imported despite the extra being absent')\n"
        "except ImportError as e:\n"
        "    assert \"pip install 'views-pipeline-core[appwrite]'\" in str(e), str(e)\n"
        "    assert 'ADR-047' in str(e), 'the message should say what NEEDS no extra'\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr