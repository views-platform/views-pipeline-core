"""Import-purity guards: pandas is a LEGACY option, never a foundational import (#320, C-225).

The frame-native goal (epic #300) requires that importing the base manager —
the class every engine must extend — does not load pandas. These tests are the
permanent architectural tripwire: a future top-level import that re-couples the
frame path to the legacy DataFrame tier turns them red in every CI job.

Both tests run the probe in a SUBPROCESS: the pytest process itself has pandas
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
