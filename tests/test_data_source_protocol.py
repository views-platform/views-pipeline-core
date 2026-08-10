"""`IDataSource` must describe `ViewsDataLoader`, not an intention. Issue #144.

## Why a structural test and not `isinstance`

`IDataSource` is `runtime_checkable`, which sounds like it makes this test unnecessary. It
does not. `runtime_checkable` verifies that the named attributes *exist* — it never looks
at their signatures. A Protocol whose `get_data` had drifted to different parameters would
still certify, and an engine programming against the declared contract would break at the
call site while every check in this repo stayed green.

That is the whole failure mode the Protocol was introduced to prevent, so it is the thing
worth testing. (`issubclass` is unavailable here anyway: the Protocol has property members,
and Python refuses `issubclass` on Protocols with non-method members.)

## What is compared, and what is deliberately not

**Compared:** parameter names, order, kind, and defaults. That is the call site — the part
an engine writes and the part that breaks when it drifts.

**Not compared:** return annotations. `types.py` is held pandas-free at import
(`tests/test_import_purity.py`), so the Protocol annotates `get_data` as `Any` rather than
`tuple[pd.DataFrame, list]`. Comparing returns would force a choice between a Protocol that
lies and an import guard that fails. The difference is recorded in the Protocol's own
docstring rather than hidden here.

`types.py` also uses `from __future__ import annotations` while `dataloaders.py` does not,
so annotations arrive as strings on one side and objects on the other. Parameter *types*
are therefore compared as text, and only where both sides declare one.
"""

from __future__ import annotations

import inspect

import pytest

from views_pipeline_core.modules.dataloaders import ViewsDataLoader
from views_pipeline_core.types import IDataSource

METHODS = ("get_feature_frame", "get_data")
PROPERTIES = ("cached_frame_path", "cached_data_path")


def _members(protocol) -> set[str]:
    """Every member the Protocol declares, derived rather than listed.

    A hand-listed member set is how a Protocol grows a method that nothing checks.
    """
    return {
        name
        for name in dir(protocol)
        if not name.startswith("_") and not hasattr(object, name)
    }


def test_the_test_covers_every_member_the_protocol_declares():
    """If `IDataSource` gains a member, this file must gain a case for it."""
    declared = _members(IDataSource)
    covered = set(METHODS) | set(PROPERTIES)
    assert declared == covered, (
        f"IDataSource declares {sorted(declared)} but this file checks {sorted(covered)}. "
        f"Unchecked members: {sorted(declared - covered)}; checked but gone: "
        f"{sorted(covered - declared)}. A Protocol member nothing verifies is a promise "
        f"to engine repos that nothing keeps."
    )


@pytest.mark.parametrize("name", METHODS)
def test_the_loader_declares_the_method(name):
    assert hasattr(ViewsDataLoader, name), (
        f"IDataSource promises '{name}' and ViewsDataLoader does not provide it. An "
        f"engine programming against the Protocol would fail at the call site."
    )


@pytest.mark.parametrize("name", METHODS)
def test_parameters_are_identical(name):
    """The call site is the contract. Names, order, kind and defaults must all match."""
    expected = list(inspect.signature(getattr(IDataSource, name)).parameters.values())
    actual = list(inspect.signature(getattr(ViewsDataLoader, name)).parameters.values())

    assert [p.name for p in expected] == [p.name for p in actual], (
        f"{name}: parameter names/order differ.\n"
        f"  IDataSource:     {[p.name for p in expected]}\n"
        f"  ViewsDataLoader: {[p.name for p in actual]}"
    )
    assert [p.kind for p in expected] == [p.kind for p in actual], (
        f"{name}: parameter kinds differ — a positional argument became keyword-only or "
        f"the reverse, which breaks callers without changing any name."
    )
    assert [p.default for p in expected] == [p.default for p in actual], (
        f"{name}: defaults differ.\n"
        f"  IDataSource:     {[p.default for p in expected]}\n"
        f"  ViewsDataLoader: {[p.default for p in actual]}\n"
        f"A parameter that is optional in the Protocol and required in the "
        f"implementation is a promise the implementation does not keep."
    )


def _annotation_text(annotation: object) -> str:
    """A comparable spelling of an annotation, whichever form it arrived in.

    `types.py` has `from __future__ import annotations`, so its annotations are strings
    (`"Optional[int]"`). `dataloaders.py` does not, so its are objects
    (`typing.Optional[int]`, `<class 'str'>`). Comparing them raw reports every parameter
    as drifted, which is a broken test rather than a finding — the first version of this
    check did exactly that.
    """
    if isinstance(annotation, str):
        return annotation.strip("'\"")
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


@pytest.mark.parametrize("name", METHODS)
def test_parameter_type_annotations_agree_where_both_declare_one(name):
    """Compared as normalised text: the two modules spell annotations differently."""
    expected = inspect.signature(getattr(IDataSource, name)).parameters
    actual = inspect.signature(getattr(ViewsDataLoader, name)).parameters

    mismatched = []
    for parameter_name, declared in expected.items():
        if declared.annotation is inspect.Parameter.empty:
            continue
        implemented = actual[parameter_name].annotation
        if implemented is inspect.Parameter.empty:
            continue
        if _annotation_text(declared.annotation) != _annotation_text(implemented):
            mismatched.append(
                f"{parameter_name}: protocol={_annotation_text(declared.annotation)!r} "
                f"implementation={_annotation_text(implemented)!r}"
            )
    assert not mismatched, f"{name}: annotation drift — {mismatched}"


def test_the_annotation_normaliser_does_not_flatten_real_differences():
    """The comparison above is only meaningful if it can still say no.

    Its first version reported every parameter as drifted; the obvious repair is to
    normalise until nothing ever differs, which would be worse — a check that cannot fail.
    """
    assert _annotation_text("Optional[int]") == _annotation_text(__import__("typing").Optional[int])
    assert _annotation_text(str) == _annotation_text("str")
    assert _annotation_text(str) != _annotation_text(int)
    assert _annotation_text("Optional[int]") != _annotation_text("Optional[str]")


@pytest.mark.parametrize("name", PROPERTIES)
def test_properties_are_properties_on_both_sides(name):
    """A property that became a method still satisfies `runtime_checkable`.

    `loader.cached_frame_path` and `loader.cached_frame_path()` are different call sites,
    and only one of them works. Nothing else in this repo would notice the swap.
    """
    assert isinstance(getattr(IDataSource, name), property), (
        f"IDataSource.{name} is no longer a property; this test's premise is stale."
    )
    assert isinstance(getattr(ViewsDataLoader, name), property), (
        f"ViewsDataLoader.{name} is not a property, but IDataSource declares it as one. "
        f"An engine reading it as an attribute would receive a bound method."
    )


def test_the_protocol_does_not_drag_pandas_into_types():
    """`types.py` is held pandas-free at import; the Protocol must not have broken that.

    Duplicated from `test_import_purity.py` on purpose — that file tests the package's
    import graph as a whole, and this one states why *this* Protocol annotates `Any`. A
    future author tempted to "fix" the annotation will hit a test that explains itself.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import views_pipeline_core.types, sys; "
            "sys.exit(1 if 'pandas' in sys.modules else 0)",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, (
        "importing views_pipeline_core.types now pulls in pandas. IDataSource annotates "
        "get_data's return as Any for exactly this reason — see its docstring."
    )
