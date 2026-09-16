"""Import-purity guards — architectural rules asserted in a subprocess, not promised.

Five rules live here (the header once said two; three arrived without updating it):


1. **pandas is a LEGACY option, never a foundational import** (#320, C-225). Importing
   the base manager — the class every engine must extend — must not load pandas.
2. **The delivery path must not be able to provision** (#331/#332, C-233). Creating a
   bucket, database or collection is a deliberate act performed by a person; if
   ``views_pipeline_core.modules.appwrite.provisioning`` is reachable from the code that
   publishes a forecast, the least-privilege key the platform is moving to cannot be
   issued. The dependency runs one way — ``provisioning`` imports ``file``, never the
   reverse — and that is what these probes check.
3. **The Appwrite SDK is an extra, and the package works without it** (#345, C-253) —
   probed by BLOCKING the import, since the extra is installed here.
4. **No module in the package imports viewser** (#511 map, ADR-063) — every module,
   derived from the filesystem, imports under a blocked ``viewser``; and a model whose
   config imports a missing data-source client gets the install command, not
   "Could not find queryset" (C-321).
5. **The log-writing module stays off the heavy chain** (#496).

The frame-native goal (epic #300) requires that importing the base manager —
the class every engine must extend — does not load pandas. These tests are the
permanent architectural tripwire: a future top-level import that re-couples the
frame path to the legacy DataFrame tier turns them red in every CI job.

All three probes run in a SUBPROCESS: the pytest process itself has pandas
loaded (fixtures, other suites), so in-process assertions would be meaningless.
They assert pandas is NOT loaded, so they need pandas installed only in the
sense that any environment qualifies — no env gate, no importorskip.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

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

# The four probes below that import the appwrite package FOR REAL (not blocked) need the
# SDK on disk. The `test-without-viewser` CI job installs no extras at all, so there they
# skip rather than fail on a premise the job deliberately does not install. Derived from
# the environment, not from a job name.
_needs_the_appwrite_sdk = pytest.mark.skipif(
    importlib.util.find_spec("appwrite") is None,
    reason="imports the appwrite package for real; the SDK is the optional `appwrite` extra",
)

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


@_needs_the_appwrite_sdk
def test_storage_module_does_not_import_provisioning():
    """`file.py` is the delivery path's storage surface; it must not reach provisioning."""
    result = _run_forbidden_probe(
        "import views_pipeline_core.modules.appwrite.file"
    )
    assert result.returncode == 0, result.stderr


@_needs_the_appwrite_sdk
def test_datastore_does_not_import_provisioning():
    """`DatastoreModule` is what the savers call — the closest caller to a real upload."""
    result = _run_forbidden_probe(
        "from views_pipeline_core.modules.datastore import DatastoreModule"
    )
    assert result.returncode == 0, result.stderr


@_needs_the_appwrite_sdk
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


@_needs_the_appwrite_sdk
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


_WITH_AN_IMPORT_BLOCKED = '''
import builtins, sys
_real = builtins.__import__
def _blocked(name, *a, **k):
    if name == "{blocked}" or name.startswith("{blocked}."):
        # ModuleNotFoundError with `.name`, exactly what a genuinely absent package raises:
        # code that keys on `.name` (the queryset loader) must see the real shape, and
        # code that catches ImportError sees a subclass of it, as before.
        raise ModuleNotFoundError("No module named '%s'" % name, name=name)
    return _real(name, *a, **k)
builtins.__import__ = _blocked
for _m in [m for m in sys.modules if m.startswith("{blocked}")]:
    del sys.modules[_m]
{body}
'''


def _run_with_import_blocked(blocked: str, body: str) -> subprocess.CompletedProcess:
    """Run a probe in an interpreter where `import <blocked>` always fails.

    The dependency IS installed in this environment, so a test that merely checks it is
    absent from `sys.modules` cannot tell "not imported" from "not installed". Blocking
    the import is what makes the without-it path genuinely exercised rather than assumed —
    the same reason `test_appwrite_pagination.py` builds its double from the SDK's real
    query encoding instead of from belief (C-218).

    What this CANNOT see: a dependency that arrives only through the blocked package's own
    chain (pyarrow and tqdm did, through viewser, until 3.3.0 declared them) — those are
    installed regardless. The `test-without-viewser` CI job resolves the manifest without
    viewser for that class; this probe is the in-process half.

    Extracted on the second incident (appwrite, then viewser — WET before DRY): the two
    copies differed only in the blocked name and in one raising a bare ImportError, which
    was the less faithful of the two.
    """
    return subprocess.run(
        [sys.executable, "-c", _WITH_AN_IMPORT_BLOCKED.format(blocked=blocked, body=body)],
        capture_output=True,
        text=True,
    )


def _run_without_the_extra(body: str) -> subprocess.CompletedProcess:
    """The appwrite extra blocked — see `_run_with_import_blocked`."""
    return _run_with_import_blocked("appwrite", body)


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


# ----------------------------------------------------------------------------------
# Nothing in the package imports viewser (#511 map; ADR-063)
# ----------------------------------------------------------------------------------

def _run_without_viewser(body: str) -> subprocess.CompletedProcess:
    """viewser blocked — see `_run_with_import_blocked`."""
    return _run_with_import_blocked("viewser", body)


def test_the_viewser_block_can_actually_fail():
    """The control — a probe that cannot fail is not a guard."""
    result = _run_without_viewser("import viewser")
    assert result.returncode != 0


def test_every_module_in_the_package_imports_without_viewser():
    """Import every module of the package under a blocked `viewser`.

    The module list is DERIVED FROM THE FILESYSTEM, not from `pkgutil.walk_packages`:
    `views_pipeline_core/modules/` has no `__init__.py`, and `walk_packages` does not
    descend into namespace packages — the first version of this test walked 63 modules
    and never saw dataloaders, wandb, appwrite, validation or reconciliation at all
    (a guard wrong about its own scope, again). `__main__` modules are skipped because
    importing one runs it.

    The only import failure tolerated is a module refusing because a DECLARED extra is
    absent — decided by the ROOT CAUSE of the ImportError being a ModuleNotFoundError for
    one of that extra's own packages (read from the manifest), not by the extra's name
    appearing in the message: a module-scope `import viewser` inside the appwrite package's
    guarded block produced a message that named the extra and slipped through (guard
    audit, mutation P5). Anything else — a missing required dependency, a module-scope
    `import viewser` anywhere — fails the test and names the module.
    """
    result = _run_without_viewser(_IMPORT_EVERY_MODULE)
    assert result.returncode == 0, result.stdout + result.stderr
    # The exit code is not the only signal: a module calling sys.exit(0) at import ended the
    # walk early and green (guard audit, mutation P3b). The summary line proves it finished.
    assert "modules walked;" in result.stdout, result.stdout


# Shared with the `test-without-viewser` CI job, which runs THIS test from /tmp against the
# installed wheel; keeping the walker in one place is what keeps the two in step.
_IMPORT_EVERY_MODULE = """
import importlib, pathlib, tomllib
import views_pipeline_core as pkg
root = pathlib.Path(pkg.__path__[0])
manifest = pathlib.Path(%r)
extras = tomllib.loads(manifest.read_text())["tool"]["poetry"].get("extras", {})
extra_packages = {p.replace("-", "_") for pkgs in extras.values() for p in pkgs}
def _root_missing_module(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ModuleNotFoundError):
            return (exc.name or "").split(".")[0]
        exc = exc.__cause__ or exc.__context__
    return None
names = []
for f in sorted(root.rglob("*.py")):
    parts = list(f.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if parts and parts[-1] == "__main__":
        continue
    names.append(".".join([pkg.__name__, *parts]))
assert len(names) > 100, names
failed, refused_by_extra = {}, {}
for n in names:
    try:
        importlib.import_module(n)
    except ImportError as e:
        bucket = refused_by_extra if _root_missing_module(e) in extra_packages else failed
        bucket[n] = str(e)
assert not failed, failed
print(len(names), "modules walked;", len(refused_by_extra), "refused by a declared extra")
""" % str((Path(__file__).resolve().parents[1] / "pyproject.toml"))


def test_a_viewser_model_without_viewser_says_how_to_install_it():
    """End to end through the real loader: the message reaches the operator, not a
    `RuntimeError('Could not find queryset')` two calls later (register C-321)."""
    result = _run_without_viewser(
        "import tempfile, pathlib\n"
        "from unittest.mock import patch\n"
        "from views_pipeline_core.data.model_path import ModelPathManager\n"
        "root = pathlib.Path(tempfile.mkdtemp()); (root / '.gitignore').touch()\n"
        "m = root / 'models' / 'probe_model'\n"
        "for d in ('artifacts', 'configs', 'data/raw', 'data/processed', 'data/generated', "
        "'reports', 'notebooks', 'logs'):\n"
        "    (m / d).mkdir(parents=True)\n"
        "for f in ('config_deployment', 'config_hyperparameters', 'config_meta', "
        "'config_partitions', 'config_sweep'):\n"
        "    (m / 'configs' / (f + '.py')).touch()\n"
        "(m / 'main.py').touch(); (m / 'README.md').touch()\n"
        "(m / 'configs' / 'config_queryset.py').write_text("
        "'from viewser import Queryset\\ndef generate():\\n    return Queryset()\\n')\n"
        "with patch.object(ModelPathManager, 'find_project_root', return_value=root):\n"
        "    try:\n"
        "        ModelPathManager('probe_model', validate=True).get_queryset()\n"
        "        raise SystemExit('loaded a viewser queryset with viewser blocked')\n"
        "    except ImportError as e:\n"
        "        assert 'pip install viewser' in str(e), str(e)\n"
        "        assert 'views-pipeline-core[viewser]' in str(e), str(e)\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ----------------------------------------------------------------------------------
# The log-writing module stays off the heavy chain (#496)
# ----------------------------------------------------------------------------------

HEAVY_PROBE = (
    "import sys; {imports}; "
    "heavy = sorted(m for m in ('numpy', 'pandas', 'views_frames') if m in sys.modules); "
    "assert not heavy, f'heavy imports pulled in: {{heavy}}'"
)


def test_log_file_utils_does_not_pull_in_the_heavy_chain():
    """`files/utils.py` writes the run log and must stay importable for that alone.

    Its own header states the property: pandas is imported function-locally so the module
    can sit on the frame-native import chain without loading it. #496 added
    `config_maturity` to it, and a module-level import of the sniffer would have pulled
    `views_frames.SpatialLevel` — and numpy behind it — straight back onto that chain:
    measured at 0.131s to import the sniffer versus 0.012s for this module with the
    import inside the function — both with a bare interpreter, because timing through
    `conda run` folds conda's own startup into the number and was how this figure came
    to disagree with the one in `files/utils.py` (0.035 vs 0.013).

    The property was documented in a comment and pinned by nothing, which is why it was
    a comment away from being lost. If this turns red, move the offending import into the
    function that needs it — do not relax the test.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            HEAVY_PROBE.format(imports="import views_pipeline_core.files.utils"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_the_heavy_probe_can_actually_fail():
    """The control. A probe that cannot fail is not a guard.

    Without this, `test_log_file_utils_does_not_pull_in_the_heavy_chain` would pass
    identically if the probe string were misspelled or the module names were wrong.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            HEAVY_PROBE.format(
                imports="import views_pipeline_core.modules.validation.core_config_sniffer"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "the sniffer imports views_frames at module scope, so this probe MUST fail — "
        "if it passes, the probe is measuring nothing"
    )
