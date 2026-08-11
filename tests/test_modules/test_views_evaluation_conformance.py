"""What this repo assumes of views-evaluation, asked of the installed package. #430.

Run: `conda run -n views_pipeline pytest tests/test_modules/test_views_evaluation_conformance.py -q`

## Probe, not conformance test — the difference this closes

`managers/reporting/stage.py:100`'s `_require_evaluation_source_consumer` is a good
runtime probe: it turns a confusing `AttributeError` deep in a stage into an actionable
message. It also fires only when that code path executes, in production, on the operator's
time — **it has never run in CI**. A conformance test runs before merge, against the
installed neighbour, and fails while someone can still do something about it.

C-193 is the standing reminder of what the gap costs: a constructor shape changed under
this repo and nothing noticed until engines broke.

## Three outcomes, never two

| state | result |
|---|---|
| `views_evaluation` not installed | **skip**, naming why |
| installed, contract located | **verify** |
| installed, contract *not* locatable | **fail** |

The third is the one that gets written as a skip, and a check that skips on both stops
checking the day the neighbour renames an attribute — while still reporting green. That
collapse from three states to two is the shape of Cluster J and it is the whole reason
these tests are written this way.

## No copy of their vocabulary lands here

Every expectation below is read off the installed package. A local copy of a neighbour's
enum or field list is the defect with an extra step: it goes stale silently and nothing
fails until production. See `test_datafactory_format_contract.py` and register C-62.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import sys

import numpy as np
import pytest

#: Package prefixes other tests replace with `MagicMock` and never restore.
_POISONED_PREFIX = "views_evaluation"


@contextlib.contextmanager
def _real_package():
    """Yield the genuinely-installed `views_evaluation`, whatever `sys.modules` holds.

    **Load-bearing, not defensive habit.** `tests/test_explicit_tasks.py:11` and
    `tests/test_falsification_foundational_fix.py:12` assign
    `sys.modules["views_evaluation"] = MagicMock()` at module import time and never
    restore it. Anything collected afterwards sees a mock.

    A conformance test that accepted that mock would pass every assertion in this file
    while verifying nothing — a `MagicMock` satisfies `hasattr`, and `inspect.signature`
    happily describes it. It would also have been invisible, because the file passes when
    run alone.

    Loading from disk with `PathFinder` was tried first and is *worse*: the real
    `views_evaluation/__init__.py` imports its own submodules, and those inner imports
    still resolve through the poisoned `sys.modules`, so the result depends on collection
    order. An order-dependent conformance test is worse than a wrong one.

    So: evict the mocks, import for real, put the mocks back exactly as they were. Other
    tests keep the doubles they rely on, and this one gets the truth.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == _POISONED_PREFIX or name.startswith(_POISONED_PREFIX + ".")
    }
    for name in saved:
        del sys.modules[name]
    try:
        yield importlib.import_module(_POISONED_PREFIX)
    except ImportError:
        yield None
    finally:
        for name in [
            n
            for n in sys.modules
            if n == _POISONED_PREFIX or n.startswith(_POISONED_PREFIX + ".")
        ]:
            del sys.modules[name]
        sys.modules.update(saved)


def _installed() -> bool:
    with _real_package() as package:
        return package is not None


if not _installed():
    pytest.skip(
        "views-evaluation is not installed; the `test-core-only` CI job runs without it. "
        "When absent there is no contract to verify — which is a skip, not a pass.",
        allow_module_level=True,
    )


def _evaluation_frame():
    """The class this repo imports, from the dotted path `adapter.py:5` uses.

    Located by the same import this repo actually writes — a test that found the class by
    another route would keep passing after that import broke.
    """
    with _real_package():
        module = importlib.import_module("views_evaluation.evaluation.evaluation_frame")
        return getattr(module, "EvaluationFrame", None)


def test_the_real_package_is_reachable_even_when_sys_modules_is_mocked():
    """The reason `_real_package` exists, pinned so nobody simplifies it away.

    If a future cleanup makes those two files restore `sys.modules`, this starts skipping
    and the indirection can go. Until then it records a live hazard: any conformance test
    written the obvious way, against any mocked package, silently verifies a mock.
    """
    in_sys_modules = sys.modules.get(_POISONED_PREFIX)
    if in_sys_modules is None or (
        inspect.ismodule(in_sys_modules) and not hasattr(in_sys_modules, "_mock_name")
    ):
        pytest.skip("sys.modules is not poisoned in this collection order")

    with _real_package() as package:
        assert package is not in_sys_modules
        assert inspect.ismodule(package) and not hasattr(package, "_mock_name")
    # and the mock must be back afterwards, or this test breaks the ones that need it
    assert sys.modules.get(_POISONED_PREFIX) is in_sys_modules


def test_the_probe_can_find_anything_at_all():
    """Control. If the module were empty, every assertion below would be vacuous."""
    with _real_package() as package:
        assert inspect.ismodule(package)
        assert dir(package), "views_evaluation exposes no names at all"


def test_native_evaluator_is_importable_from_the_package_root():
    """`managers/evaluation/stage.py:90` does `from views_evaluation import NativeEvaluator`.

    A move to a submodule would break that import at evaluation time — after training has
    already run.
    """
    with _real_package() as package:
        assert hasattr(package, "NativeEvaluator"), (
            f"views-evaluation no longer exports `NativeEvaluator` from its package root. "
            f"`managers/evaluation/stage.py` imports it from there. Available: "
            f"{sorted(n for n in dir(package) if not n.startswith('_'))}"
        )


def test_native_evaluator_takes_a_single_config_mapping():
    """`stage.py:122` calls `NativeEvaluator(context.configs)` — one positional argument.

    Verified against the real signature rather than by constructing one, because
    construction may touch config this repo has no business fabricating. A new
    *required* parameter is the break that matters; new optional ones are fine.
    """
    with _real_package() as package:
        signature = inspect.signature(package.NativeEvaluator)
    required = [
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.default is inspect.Parameter.empty
        and param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    assert len(required) == 1, (
        f"`NativeEvaluator` now requires {required}; `managers/evaluation/stage.py:122` "
        f"passes exactly one argument (`context.configs`). Signature: {signature}"
    )


def test_evaluation_frame_is_importable_where_this_repo_imports_it():
    """`modules/validation/adapter.py:5` imports from
    `views_evaluation.evaluation.evaluation_frame`.

    Fails rather than skips if the module has moved — that is the third outcome. The
    package is installed, so an unlocatable contract means it changed, not that it is
    unavailable.
    """
    frame_cls = _evaluation_frame()
    assert frame_cls is not None and inspect.isclass(frame_cls), (
        "views-evaluation is installed but "
        "`views_evaluation.evaluation.evaluation_frame.EvaluationFrame` cannot be "
        "located. That is the path `modules/validation/adapter.py:5` uses. "
        "**Not a skip** — the package is here and the contract moved. A check that "
        "skipped on both absence and relocation would stop checking the day they "
        "renamed it, while still reporting green."
    )


def test_evaluation_frame_accepts_the_keywords_this_repo_passes():
    """`EvaluationAdapter` builds frames with exactly these four keywords.

    Both construction sites (`adapter.py:205`, `adapter.py:295`) pass `y_true`, `y_pred`,
    `identifiers` and `metadata`. A renamed or newly-required parameter breaks evaluation
    for every model, and until now nothing here would have said so before a run.
    """
    signature = inspect.signature(_evaluation_frame())
    accepted = set(signature.parameters)
    passed = {"y_true", "y_pred", "identifiers", "metadata"}

    assert passed <= accepted, (
        f"`EvaluationFrame` no longer accepts {sorted(passed - accepted)}. "
        f"`modules/validation/adapter.py` passes all of {sorted(passed)} by keyword at "
        f"lines 205 and 295. Accepted now: {sorted(accepted)}"
    )

    newly_required = [
        name
        for name, param in signature.parameters.items()
        if name not in passed
        and name != "self"
        and param.default is inspect.Parameter.empty
        and param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    assert not newly_required, (
        f"`EvaluationFrame` now requires {newly_required}, which "
        f"`modules/validation/adapter.py` does not pass. Every evaluation run would raise."
    )


#: The two `y_pred` shapes `EvaluationAdapter` produces. Point forecasts arrive 1D and
#: are reshaped to `(N, 1)` at `modules/validation/adapter.py:139-140` before being
#: concatenated; sampled forecasts are already `(N, S)`. Both must be acceptable, and the
#: reshape is exactly the kind of normalisation that gets dropped in a refactor because
#: nothing downstream of this repo appears to depend on it.
_SAMPLE_COUNTS = {"point_forecast_reshaped_to_2d": 1, "sampled_forecast": 8}


@pytest.mark.parametrize("case,samples", sorted(_SAMPLE_COUNTS.items()))
def test_an_evaluation_frame_can_be_built_the_way_this_repo_builds_one(case, samples):
    """End-to-end on the real class, with the shapes and identifier keys this repo passes.

    The signature checks above prove the parameter *names* are accepted. This proves the
    neighbour does not reject the **contents** — the four identifier keys `time`, `unit`,
    `origin`, `step` from `adapter.py:205`, and the 2D `y_pred`. A neighbour that started
    validating either would pass every check above and fail every real run.

    Writing this test found a defect in the test, not the contract: the first version
    passed a 1D `y_pred` and views-evaluation refused it — `y_pred must be 2D (N, S)`.
    That is `adapter.py:139-140`'s reshape being load-bearing, which is worth pinning
    from this side too.
    """
    frame_cls = _evaluation_frame()
    rows = 4

    frame = frame_cls(
        y_true=np.arange(rows, dtype=float),
        y_pred=np.random.default_rng(seed=3).random((rows, samples)),
        identifiers={
            "time": np.arange(rows, dtype=int),
            "unit": np.full(rows, 1000, dtype=int),
            "origin": np.zeros(rows, dtype=int),
            "step": np.arange(1, rows + 1, dtype=int),
        },
        metadata={"target": "lr_sb_best"},
    )

    assert frame is not None, case


# ----------------------------------------------------------------------------------
# The metric vocabulary. Added after views-models#372 caught this repo recommending a
# config that clears its own gate and fails the neighbour's.
# ----------------------------------------------------------------------------------


def test_the_metric_key_names_this_repo_knows_are_the_ones_the_evaluator_knows():
    """`CoreConfigSniffer`'s metric-key constants must be views-evaluation's, exactly.

    `REGRESSION_METRIC_KEYS` and `CLASSIFICATION_METRIC_KEYS` are a local copy of the keys
    `NativeEvaluator._METRIC_LIST_KEYS` maps to `(task, kind)` cells. A copy of a
    neighbour's vocabulary goes stale silently — C-62, and the reason every other
    expectation in this file is read off the installed package rather than written down.

    Not deleted in favour of importing theirs: the sniffer runs at config load and this
    repo does not otherwise require views-evaluation at that point. A checked copy is the
    compromise, and this is the check.
    """
    from views_pipeline_core.modules.validation.core_config_sniffer import (
        CLASSIFICATION_METRIC_KEYS,
        REGRESSION_METRIC_KEYS,
    )

    with _real_package() as package:
        from importlib import import_module

        evaluator = import_module("views_evaluation.evaluation.native_evaluator")
        theirs = set(evaluator.NativeEvaluator._METRIC_LIST_KEYS)
        assert package is not None

    ours = set(REGRESSION_METRIC_KEYS) | set(CLASSIFICATION_METRIC_KEYS)
    assert ours == theirs, (
        f"this repo's metric-key constants have drifted from views-evaluation's. "
        f"Only here: {sorted(ours - theirs)}. Only there: {sorted(theirs - ours)}. "
        f"`CoreConfigSniffer` would accept a key the evaluator ignores, or reject one it "
        f"requires."
    )


def test_every_metric_cell_the_evaluator_declares_has_metrics_in_it():
    """A control: an empty cell would make the membership check below vacuous."""
    with _real_package():
        from importlib import import_module

        catalog = import_module("views_evaluation.evaluation.metric_catalog")
        membership = catalog.METRIC_MEMBERSHIP

    assert membership, "METRIC_MEMBERSHIP is empty"
    for cell, metrics in membership.items():
        assert metrics, f"cell {cell} declares no valid metrics"


def test_a_metric_is_only_valid_for_the_cell_the_evaluator_puts_it_in():
    """The specific thing this repo got wrong, pinned so the lesson is executable.

    `AP` is a classification **point** metric. This repo recommended it as a
    classification **sample** metric in views-models#372 — a config that clears
    `CoreConfigSniffer` (which checks only that a metric key is present) and is then
    refused by `NativeEvaluator._validate_config`, moving the failure from config-load to
    evaluation time rather than removing it. views-models caught it.

    Read off the installed package, so it tracks their vocabulary rather than restating
    it. If views-evaluation ever does make `AP` valid for samples, this fails and the
    recommendation can change — which is the correct direction for the dependency.
    """
    with _real_package():
        from importlib import import_module

        catalog = import_module("views_evaluation.evaluation.metric_catalog")
        membership = catalog.METRIC_MEMBERSHIP

    assert "AP" in membership[("classification", "point")], (
        "`AP` is no longer a classification point metric; the guidance in "
        "views-models#372 and the control in "
        "tests/test_falsification_gate_pooling_splash_zone.py depend on where it lives."
    )
    assert "AP" not in membership[("classification", "sample")], (
        "`AP` is now valid as a classification SAMPLE metric. The correction this repo "
        "sent to views-models#372 — use `Brier_cls_sample`, not `AP`, in "
        "`classification_sample_metrics` — is no longer the whole story. Re-check before "
        "repeating it."
    )
    assert "Brier_cls_sample" in membership[("classification", "sample")], (
        "`Brier_cls_sample` is what views-models' eight constituent models declare and "
        "what this repo's corrected control uses. If it moved cells, that control is now "
        "certifying the wrong thing."
    )
