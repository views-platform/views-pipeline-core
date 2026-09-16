"""`--update_viewser` refuses, and the transformation library is gone. 2026-09-16.

Run: `conda run -n views_pipeline pytest tests/test_modules/test_update_viewser_is_retired.py -q`

## What was retired

`UpdateViewser` — ADR-037's emergency fallback for the February 2025 ingester outage, which
patched a cached VIEWSER frame with hand-supplied GED/ACLED months by replaying the
queryset's transformation chain through `views-transformation-library`. It was briefly live
(2025-10-09 to 2025-11-24), switched off with no recorded reason, never configured on any
machine, and its own ADR warned against letting it become permanent.

## Why the flag refuses instead of disappearing

Three READMEs still told operators to pass `-u`. Deleting the flag would turn a silent no-op
into `unrecognized arguments`, which tells nobody why — and is a public-surface removal that
needs a major. So for one window the flag parses and REFUSES, naming the retirement and the
ADR. That is C-317's shape from the 3.2.0 audit: accepted-but-unimplemented becomes refused,
not shrugged. The flag itself goes at 4.0.

## Why the import probe is here

The dependency was on the import path of every data fetch and of 14 test files, through a
single module-scope import in a file that existed to serve a dead feature. Removing the pin
without pinning its absence is how it comes back in a docstring's worth of time.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_the_retired_flag_refuses_before_any_fetch():
    """Passing `--update_viewser` raises, naming the retirement and ADR-037.

    Driven through the real `_execute_data_fetching` on a manager built without its
    constructor (which pulls wandb, logging and four ADR-045 stages that this check never
    reaches). The refusal must fire BEFORE any data is touched, so nothing else is set up —
    if the check were placed after the fetch, this test would fail on the missing loader
    rather than on the refusal.
    """
    from views_pipeline_core.exceptions import DataFetchException
    from views_pipeline_core.managers.model.model import ForecastingModelManager

    manager = object.__new__(ForecastingModelManager)
    manager._args = SimpleNamespace(update_viewser=True)
    manager._wandb_module = MagicMock()

    with pytest.raises(DataFetchException, match="retired on 2026-09-16") as info:
        manager._execute_data_fetching()

    message = str(info.value)
    assert "037_ingester_emergency_solution" in message, "must name the ADR that explains it"
    assert "Drop the flag" in message, "must tell the operator what to do"


def test_without_the_flag_the_refusal_does_not_fire():
    """The control. If the refusal fired regardless, every run would stop here."""
    from views_pipeline_core.exceptions import DataFetchException
    from views_pipeline_core.managers.model.model import ForecastingModelManager

    manager = object.__new__(ForecastingModelManager)
    manager._args = SimpleNamespace(update_viewser=False)
    manager._wandb_module = MagicMock()
    # No loader is attached, so the method will fail on the NEXT line — which is the
    # point: it must get PAST the refusal. Anything but DataFetchException proves it did.
    with pytest.raises(Exception) as info:
        manager._execute_data_fetching()
    assert not isinstance(info.value, DataFetchException) or "retired" not in str(info.value), (
        "the refusal fired with the flag off"
    )


def test_the_flag_still_parses_for_one_window():
    """Removing the flag is the 4.0 job; this window it must parse and warn in its help.

    An operator following `views-models/README.md:415-424` types `-u`. They must reach the
    refusal above, not argparse's `unrecognized arguments`.
    """
    from unittest.mock import patch

    from views_pipeline_core.cli.args import ForecastingModelArgs

    with patch.object(sys, "argv", ["script.py", "--run_type", "calibration", "--train", "-u"]):
        parsed = ForecastingModelArgs.parse_args()
    assert parsed.update_viewser is True


PROBE = (
    "import sys; {imports}; "
    "loaded = sorted(m for m in sys.modules if m.startswith('views_transformation_library')); "
    "assert not loaded, f'views_transformation_library loaded by {{__name__}}: {{loaded[:3]}}'"
)


@pytest.mark.parametrize(
    "imports",
    [
        "import views_pipeline_core",
        "from views_pipeline_core.modules.dataloaders import ViewsDataLoader",
        "import views_pipeline_core.modules.dataloaders.dataloaders",
        "from views_pipeline_core.managers.model import ForecastingModelManager",
    ],
)
def test_nothing_on_the_data_path_loads_the_transformation_library(imports):
    """The pin's absence, pinned.

    Subprocess, like every probe in `test_import_purity.py`, because the pytest process
    may already have unrelated modules loaded. Before the retirement, resolving
    `ViewsDataLoader` pulled the library and scikit-learn behind it on every fetch and at
    collection of 14 test files.
    """
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(imports=imports)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_the_probe_can_actually_fail():
    """The control — a probe that cannot fail is not a guard."""
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(
            imports="import types; sys.modules['views_transformation_library'] = types.ModuleType('x')"
        )],
        capture_output=True, text=True,
    )
    assert result.returncode != 0, "the probe reports clean when the module is present"
