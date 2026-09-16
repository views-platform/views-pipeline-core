"""`--update_viewser` and `UpdateViewser` refuse for one window; the library is gone. C-318.

Run: `conda run -n views_pipeline pytest tests/test_modules/test_update_viewser_is_retired.py -q`

The history — what was retired, when it was live, and why — is written ONCE, in the closing
note of `documentation/ADRs/037_ingester_emergency_solution.md`. This file does not restate
it; the first version did, in six places that already disagreed with each other (C-320).

## What this file pins, and why each pin exists

- The flag refuses AT THE BOUNDARY (`ForecastingModelArgs._validate`), so every entry
  point — model, sweep, ensemble parent, child subprocess — hits it. The first version
  refused inside one manager's data-fetch method, which the ensemble parent never calls: a
  cached ensemble run with `-u` finished green (C-319).
- The flag still PARSES, so an operator following an old README is told, not handed
  `unrecognized arguments`.
- The stub CLASS raises on construction, and the help text says RETIRED — both promised in
  a CHANGELOG, a facade comment and an ADR, and pinned by nothing until this version.
- The library's absence is probed against EVERY submodule of the facade, derived from
  `_LAZY_SUBMODULES`. The first version hand-listed four import targets, none of which
  resolved the lazy stub — so pasting the old file back tripped nothing here.
- The 4.0 removal has an executable trigger. "Removed at 4.0" appeared in eleven places
  with no mechanism; a test that asserts `major < 4` is one.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from tests.test_import_purity import _run_forbidden_probe
from views_pipeline_core.modules.dataloaders import _LAZY_SUBMODULES

_FLAG_ARGV = ["script.py", "--run_type", "calibration", "--train", "-u"]


# ----------------------------------------------------------------------------------
# The flag: parses, then refuses at the boundary
# ----------------------------------------------------------------------------------


def test_the_retired_flag_refuses_at_argument_validation(capsys):
    """`-u` reaches `_validate` and exits there, before any manager exists.

    Driven through the real `parse_args`, the way every entry point reaches it. The
    refusal must name the ADR and tell the operator what to do; it must NOT be the
    argparse `unrecognized arguments` exit, which tells them nothing.
    """
    from views_pipeline_core.cli.args import ForecastingModelArgs

    with patch.object(sys, "argv", _FLAG_ARGV), pytest.raises(SystemExit) as info:
        ForecastingModelArgs.parse_args()

    assert info.value.code == 1, "a validation refusal exits 1; argparse's unknown-flag exit is 2"
    out = capsys.readouterr()
    text = out.out + out.err
    assert "retired" in text and "037_ingester_emergency_solution" in text, text
    assert "unrecognized arguments" not in text, "the flag must still parse — it refuses, not vanishes"
    assert "To fix: drop the flag" in text


def test_without_the_flag_validation_passes():
    """The control. Same argv minus `-u` must construct normally."""
    from views_pipeline_core.cli.args import ForecastingModelArgs

    with patch.object(sys, "argv", _FLAG_ARGV[:-1]):
        parsed = ForecastingModelArgs.parse_args()
    assert parsed.update_viewser is False


def test_the_help_text_says_retired():
    """The one thing an operator reads before typing the flag."""
    from views_pipeline_core.cli.args import ForecastingModelArgs

    with patch.object(sys, "argv", ["script.py", "--help"]), pytest.raises(SystemExit):
        with patch("sys.stdout") as fake_out:
            ForecastingModelArgs.parse_args()
    rendered = "".join(str(c.args[0]) for c in fake_out.write.call_args_list if c.args)
    assert "RETIRED" in rendered, "the help text no longer says the flag is retired"


# ----------------------------------------------------------------------------------
# The class: resolves, then refuses on construction
# ----------------------------------------------------------------------------------


def test_the_stub_still_resolves_through_the_facade():
    from views_pipeline_core.modules.dataloaders import UpdateViewser  # noqa: F401


def test_the_stub_refuses_on_construction():
    """Promised by the facade comment, the CHANGELOG and ADR-037 — now asserted.

    Mutation the first version missed: replacing the `raise` with `return None` left every
    test green, because nothing ever constructed the stub.
    """
    from views_pipeline_core.modules.dataloaders import UpdateViewser

    with pytest.raises(RuntimeError, match="retired on 2026-09-16"):
        UpdateViewser(None, None, None, None)


def test_the_stub_keeps_the_recorded_signature():
    """Four REQUIRED positional parameters, as the surface snapshot recorded at 3.0.0.

    The first stub gave them defaults, which the surface guard does not flag (it watches
    only for narrowing) — so a widened contract was silently recorded as unchanged.
    """
    import inspect

    from views_pipeline_core.modules.dataloaders import UpdateViewser

    params = inspect.signature(UpdateViewser).parameters
    assert list(params) == ["queryset", "viewser_df", "data_path", "months_to_update"]
    assert all(p.default is inspect.Parameter.empty for p in params.values()), (
        "a default on a recorded-required parameter widens the contract the snapshot holds"
    )


# ----------------------------------------------------------------------------------
# The dependency: absent from every path the facade can resolve
# ----------------------------------------------------------------------------------


_TARGETS = sorted(
    [f"import views_pipeline_core.modules.dataloaders.{m}" for m in _LAZY_SUBMODULES]
    + ["from views_pipeline_core.managers.model import ForecastingModelManager"]
)


@pytest.mark.parametrize("imports", _TARGETS)
def test_nothing_the_facade_resolves_loads_the_transformation_library(imports):
    """Derived from `_LAZY_SUBMODULES`, so the stub module itself is a target.

    The first version listed four imports by hand and none resolved the lazy stub — so
    the one module that had ever imported the library was the one module never probed.
    Reuses `test_import_purity._run_forbidden_probe`, the sixth copy of which this file
    would otherwise have been.
    """
    result = _run_forbidden_probe(imports, forbidden="views_transformation_library")
    assert result.returncode == 0, result.stderr


def test_the_loader_no_longer_imports_viewser_at_module_scope():
    """A property the retirement produced by accident, pinned so it is not lost by one.

    Cutting the dead methods removed the loader's only module-level `viewser` import;
    `import views_pipeline_core.modules.dataloaders.dataloaders` went from 1.6s to 0.4s.
    The facade's own docstring had claimed the loader pulls viewser at module level —
    corrected. Nothing forbade a viewser import on this path until now (C-309's shape).
    """
    result = _run_forbidden_probe(
        "import views_pipeline_core.modules.dataloaders.dataloaders", forbidden="viewser"
    )
    assert result.returncode == 0, result.stderr


def test_the_probe_can_actually_fail():
    """The control — a probe that cannot fail is not a guard."""
    result = _run_forbidden_probe(
        "import types; sys.modules['views_transformation_library'] = types.ModuleType('x')",
        forbidden="views_transformation_library",
    )
    assert result.returncode != 0


# ----------------------------------------------------------------------------------
# The window closes at 4.0 — executably
# ----------------------------------------------------------------------------------


def test_the_one_window_shim_is_gone_by_the_next_major():
    """'Removed at 4.0' was written in eleven places with no mechanism. This is one.

    When `pyproject.toml` reads a major >= 4, this fails and names what to delete. The
    guards around it point the WRONG way otherwise: the parse test above defends the flag
    against its own removal, and refreshing the surface snapshot with the stub present
    would record it as a live 4.0 export.
    """
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    major = int(re.search(r'^version = "(\d+)\.', text, re.M).group(1))

    assert major < 4, (
        "major is >= 4: the one-window retirement shim must go now. Delete "
        "`--update_viewser` (cli/args.py: field, flag, _validate refusal, and its forwarding "
        "in the three ensemble managers), `modules/dataloaders/update_viewser.py`, the "
        "`UpdateViewser` entries in `modules/dataloaders/__init__.py`, the flag row in "
        "`documentation/CICs/ForecastingModelArgs.md` §11, and this test file — THEN refresh "
        "the surface snapshot, not before."
    )
