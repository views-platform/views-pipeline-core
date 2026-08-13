"""What views-impact depends on cannot be removed without this failing. #461, epic #458.

Run: `conda run -n views_pipeline pytest tests/test_views_impact_conformance.py -q`

## Why this check is here and not in views-impact

views-hydranet, views-baseline, views-r2darts2 and views-stepshifter are exempt from
`test_every_neighbour_has_a_conformance_check.py` as consumers: the expectation lives in
their repo and is tested there.

**views-impact has no tests.** Its `tests/` directory contains only `__init__.py`. So
"their side" does not exist — and it subclasses `ForecastingModelManager`, overrides ten
inherited methods, and imports nineteen symbols across eight of our modules. It is pinned
`<3.0.0` and cannot adopt 3.x unchanged (views-impact#5), and it lost its author.

Until it has an owner and a suite, the check belongs here. **This is a deliberate
exception with a trigger: when views-impact has its own conformance test, delete this
file** and return it to `KNOWN_CONSUMERS`.

## Derived, not listed

The surface is read out of views-impact's source every run. A hand-written list of
nineteen symbols stops being true the first time it imports a twentieth — and this whole
epic exists because nobody noticed a consumer for weeks.

The keyword check is the one that matters. Every call views-impact makes into this package
is checked against the real signature, so a parameter renamed here fails **here**, naming
their call site. That check would have caught PR #328's situation in July, before its
author left.

## Three outcomes

| state | result |
|---|---|
| views-impact not checked out beside this repo | **skip**, naming why |
| present, surface intact | **pass** |
| present, surface broken | **fail**, naming the symbol and their line |

Enforced on a developer's machine, not CI — no runner can see another repo. Same limit as
the wire-fixture drift check and the SessionAuth ratchet, and the same reason it is
acceptable: whoever has views-impact checked out is whoever can act on the answer.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IMPACT = REPO_ROOT.parent / "views-impact" / "views_impact"

#: Managers whose methods views-impact overrides. Attribute name in their code -> our class.
_OVERRIDE_PARENTS = ("ForecastingModelManager", "ModelManager")


def _impact_sources() -> list[Path]:
    return [
        p
        for p in sorted(IMPACT.rglob("*.py"))
        if not {"__pycache__", ".git"} & set(p.parts)
    ]


def _parse(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):  # pragma: no cover
        return None


def imported_symbols() -> set[tuple[str, str, str, int]]:
    """`(module, name, file, lineno)` for every symbol views-impact imports from us."""
    found = set()
    for path in _impact_sources():
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("views_pipeline_core")
            ):
                for alias in node.names:
                    found.add((node.module, alias.name, path.name, node.lineno))
    return found


def _local_name_to_symbol() -> dict[str, tuple[str, str]]:
    """Local binding -> `(module, original name)`, so calls can be traced back."""
    mapping = {}
    for module, name, _, _ in imported_symbols():
        mapping[name] = (module, name)
    return mapping


def keyworded_calls() -> list[tuple[str, str, set[str], str, int]]:
    """`(module, name, keywords, file, lineno)` for calls to imported symbols."""
    bindings = _local_name_to_symbol()
    calls = []
    for path in _impact_sources():
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id not in bindings:
                continue
            module, name = bindings[node.func.id]
            kws = {k.arg for k in node.keywords if k.arg}
            if kws:
                calls.append((module, name, kws, path.name, node.lineno))
    return calls


def method_calls_on_our_objects() -> list[tuple[str, str, set[str], str, int]]:
    """Calls of the form `self._model_path.foo(bar=...)`, mapped to our classes."""
    from views_pipeline_core.data.model_path import ModelPathManager
    from views_pipeline_core.modules.dataloaders import ViewsDataLoader
    from views_pipeline_core.modules.wandb.wandb import WandBModule

    owners = {
        "_model_path": ModelPathManager,
        "model_path": ModelPathManager,
        "_data_loader": ViewsDataLoader,
        "data_loader": ViewsDataLoader,
        "_wandb_module": WandBModule,
        "wandb_module": WandBModule,
    }
    calls = []
    for path in _impact_sources():
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            value = node.func.value
            holder = (
                value.attr
                if isinstance(value, ast.Attribute)
                else value.id if isinstance(value, ast.Name) else None
            )
            if holder not in owners:
                continue
            calls.append(
                (
                    owners[holder].__name__,
                    node.func.attr,
                    {k.arg for k in node.keywords if k.arg},
                    path.name,
                    node.lineno,
                )
            )
    return calls


def overridden_methods() -> list[tuple[str, str, str, int]]:
    """`(subclass, method, file, lineno)` where the method shadows one of ours."""
    from views_pipeline_core.managers.model.model import ForecastingModelManager

    found = []
    for path in _impact_sources():
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                for b in node.bases
            }
            if not bases & set(_OVERRIDE_PARENTS):
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and hasattr(
                    ForecastingModelManager, item.name
                ):
                    found.append((node.name, item.name, path.name, item.lineno))
    return found


#: Calls views-impact makes that we deliberately do **not** satisfy, and why.
#:
#: Same discipline as `EXEMPT` in the boundary meta-test: a reason you can disagree with,
#: and an issue to follow. Anything not listed here fails.
#:
#: This exists because "make the test pass" and "do the right thing" pointed different
#: ways exactly once, and the right thing was to leave a red line pointing at a one-line
#: migration rather than reintroduce the defect.
PENDING_MIGRATION: dict[tuple[str, str], str] = {
    ("ModelPathManager", "_get_processed_data_file_paths"): (
        "views-impact calls the PRIVATE name PR #328 proposed. #459 deliberately shipped "
        "it public as `get_processed_data_file_paths`, because reaching a private method "
        "across a repo boundary is the defect #433 removed — and adding a private alias "
        "to keep this green would reintroduce it to make a test pass. One-line change on "
        "their side; tracked in views-impact#5 with the rest of their migration."
    ),
}


# ----------------------------------------------------------------------------------
# Gate + controls
# ----------------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not IMPACT.is_dir(),
    reason="views-impact is not checked out beside this repo; no CI runner can see it",
)


def test_the_scan_reads_something():
    """Control. An empty scan would make every assertion below vacuous — the failure this
    repo has shipped before (#415, C-59) and again in this epic's own meta-test."""
    sources = _impact_sources()
    assert len(sources) >= 5, f"only {len(sources)} views-impact modules found"
    assert len(imported_symbols()) >= 10, "views-impact imports almost nothing from us — "\
        "either the scan broke or the consumer relationship has changed fundamentally"


def test_the_scan_finds_the_manager_subclass():
    """If it stopped subclassing our manager, this file's whole premise is stale."""
    overrides = overridden_methods()
    assert overrides, (
        "views-impact no longer overrides any ForecastingModelManager method. Either it "
        "stopped subclassing us — in which case this file should shrink — or the scan is "
        "broken."
    )


# ----------------------------------------------------------------------------------
# The conformance assertions
# ----------------------------------------------------------------------------------


def test_every_symbol_views_impact_imports_still_exists():
    """A removed or moved symbol breaks their import at module load, not at use."""
    broken = []
    for module, name, where, line in sorted(imported_symbols()):
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - a module we deleted
            broken.append(f"{where}:{line} imports {name} from {module} — module: {exc}")
            continue
        if not hasattr(mod, name):
            broken.append(f"{where}:{line} imports {name} from {module} — SYMBOL GONE")
    assert not broken, (
        "views-impact imports symbols this package no longer provides:\n  "
        + "\n  ".join(broken)
        + "\nIt has no tests of its own, so this is the only thing that will say so."
    )


def test_no_keyword_views_impact_passes_is_unknown_to_us():
    """The check that would have caught #328's situation in July.

    Every keyword argument views-impact passes into this package, against the real
    signature. A parameter renamed here fails **here**, naming their call site — rather
    than surfacing months later as a `TypeError` in someone's run.
    """
    problems = []
    for module, name, kws, where, line in keyworded_calls():
        try:
            obj = getattr(importlib.import_module(module), name)
            params = set(inspect.signature(obj).parameters)
        except (TypeError, ValueError, AttributeError, ImportError):
            continue  # not introspectable; the import test above covers existence
        if any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(obj).parameters.values()
        ):
            continue  # **kwargs accepts anything
        unknown = kws - params
        if unknown:
            problems.append(
                f"{where}:{line} calls {name}({', '.join(sorted(unknown))}=...) — "
                f"accepts {sorted(params)}"
            )
    assert not problems, "views-impact passes keywords we do not accept:\n  " + "\n  ".join(
        problems
    )


def test_no_method_keyword_views_impact_passes_is_unknown_to_us():
    """The same, for methods called on our objects rather than imported functions.

    This is where the real gap was: `get_latest_model_artifact_path(targets_suffix=...)`
    and `get_processed_data_file_paths(...)` were both reached this way, and neither the
    import check nor a plain grep would have found them.
    """
    import views_pipeline_core.data.model_path as _mp
    import views_pipeline_core.modules.dataloaders as _dl
    import views_pipeline_core.modules.wandb.wandb as _wb

    classes = {
        "ModelPathManager": _mp.ModelPathManager,
        "ViewsDataLoader": _dl.ViewsDataLoader,
        "WandBModule": _wb.WandBModule,
    }
    problems = []
    for cls_name, method, kws, where, line in method_calls_on_our_objects():
        cls = classes[cls_name]
        if (cls_name, method) in PENDING_MIGRATION:
            continue
        if not hasattr(cls, method):
            problems.append(f"{where}:{line} calls {cls_name}.{method}() — METHOD GONE")
            continue
        try:
            params = inspect.signature(getattr(cls, method)).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue
        unknown = kws - set(params)
        if unknown:
            problems.append(
                f"{where}:{line} calls {cls_name}.{method}({', '.join(sorted(unknown))}=...) "
                f"— accepts {sorted(set(params) - {'self'})}"
            )
    assert not problems, (
        "views-impact calls methods with keywords we do not accept:\n  "
        + "\n  ".join(problems)
    )


def test_every_method_views_impact_overrides_still_exists_on_our_class():
    """An override of a method we deleted is dead code that looks alive.

    Worse than dead: `ImpactModelManager` overrides ten of ours, and if the parent stops
    calling one, their implementation silently never runs. Nothing in either repo would
    say so.
    """
    from views_pipeline_core.managers.model.model import ForecastingModelManager

    gone = [
        f"{where}:{line} {subclass}.{method}()"
        for subclass, method, where, line in overridden_methods()
        if not hasattr(ForecastingModelManager, method)
    ]
    assert not gone, (
        "views-impact overrides methods `ForecastingModelManager` no longer has:\n  "
        + "\n  ".join(gone)
    )


def test_the_four_additions_made_for_views_impact_are_still_here():
    """#459 added these because views-impact needs them. Named, because their absence is
    the one thing the derived scan above cannot report — a removed feature simply stops
    appearing in their source once they work around it again."""
    from views_pipeline_core.data.constants import model_artifact_filename
    from views_pipeline_core.data.model_path import ModelPathManager
    from views_pipeline_core.files.utils import generate_model_file_name
    from views_pipeline_core.modules.wandb.wandb import WandBModule

    assert "targets_suffix" in inspect.signature(generate_model_file_name).parameters
    assert (
        "targets_suffix"
        in inspect.signature(ModelPathManager.get_latest_model_artifact_path).parameters
    )
    assert hasattr(ModelPathManager, "get_processed_data_file_paths")
    assert hasattr(WandBModule, "log_yearly_evaluation")
    assert model_artifact_filename("calibration", "20241105_143022", ".pt") == (
        "calibration_model_20241105_143022.pt"
    )


# ----------------------------------------------------------------------------------
# The allowlist keeps itself honest
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(PENDING_MIGRATION))
def test_every_pending_migration_is_justified(key):
    """A reason long enough to argue with, and somewhere to follow it."""
    reason = PENDING_MIGRATION[key]
    assert len(reason) >= 60, f"PENDING_MIGRATION{key} reason is too thin: {reason!r}"
    assert "#" in reason, f"PENDING_MIGRATION{key} cites no issue"


@pytest.mark.parametrize("key", sorted(PENDING_MIGRATION))
def test_no_pending_migration_outlives_its_reason(key):
    """Fails when views-impact migrates, or when we quietly satisfy the call after all.

    Both are good outcomes and both must remove the entry. An allowlist that keeps
    entries after they stop being true is how it stops describing anything — the same
    rule `test_no_exemption_outlives_its_neighbour` enforces next door.
    """
    import views_pipeline_core.data.model_path as _mp
    import views_pipeline_core.modules.dataloaders as _dl
    import views_pipeline_core.modules.wandb.wandb as _wb

    cls_name, method = key
    cls = {
        "ModelPathManager": _mp.ModelPathManager,
        "ViewsDataLoader": _dl.ViewsDataLoader,
        "WandBModule": _wb.WandBModule,
    }[cls_name]

    assert not hasattr(cls, method), (
        f"{cls_name}.{method} now exists, so the entry in PENDING_MIGRATION is stale. "
        f"Remove it — and check the reason first: it says we deliberately did NOT provide "
        f"this, so someone adding it should have overturned that decision on purpose."
    )

    still_called = any(
        (c, m) == key for c, m, _, _, _ in method_calls_on_our_objects()
    )
    assert still_called, (
        f"views-impact no longer calls {cls_name}.{method} — it has migrated. Remove the "
        f"PENDING_MIGRATION entry."
    )
