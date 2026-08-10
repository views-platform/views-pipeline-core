"""Characterization of `_build_context` on both ensemble managers. Issue #432.

Run: `conda run -n views_pipeline pytest tests/test_managers/test_ensemble_context_characterization.py -q`

## Why this file exists before the refactor, not after

`_build_context` is written twice — once in `DataFrameEnsembleManager`, once in
`PredictionFrameEnsembleManager` — and a normalised AST diff puts the difference at **2 of
18 constructor arguments**. #432 collapses them into one implementation.

That refactor is only safe if "no behaviour change" is a measured claim rather than an
asserted one. So this file pins the *entire* `EnsembleContext` both managers produce, field
by field, from one shared config. It must run green before the unification and unchanged
after. If it does not, the unification changed something, and the diff will name which field.

This ordering is the point. #422 had to patch the same line in both copies because nobody
had pinned what either copy produced; unifying without characterizing first risks cementing
one copy's bug into the single remaining site.

## The completeness guard is the load-bearing part

`test_every_field_is_characterized` derives the field list from
`dataclasses.fields(EnsembleContext)` and asserts that the expectation tables below cover
all of them. A hand-written table silently stops covering a field the moment someone adds
one — and an uncompared field is exactly where a refactor hides a change. The repo has been
bitten by hand-listed worklists enough times that deriving is the standing rule.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.managers.ensemble.context import EnsembleContext
from views_pipeline_core.managers.ensemble.dataframe_ensemble import (
    DataFrameEnsembleManager,
)
from views_pipeline_core.managers.ensemble.prediction_frame_ensemble import (
    PredictionFrameEnsembleManager,
)

# ----------------------------------------------------------------------------------
# One config, fed to both managers. Every optional key is set to a NON-default value so
# a lost `c.get(...)` shows up as a changed field rather than coinciding with its default.
# ----------------------------------------------------------------------------------

SHARED_CONFIG = {
    "name": "test_ensemble",
    "models": ["purple_alien", "blue_cat"],
    "aggregation": "mean",
    "regression_targets": ["lr_sb_best", "lr_ns_best"],
    "classification_targets": ["by_sb_best"],
    "level": "pgm",
    "reconciliation": "pgm_cm_point",
    "reconcile_with": "some_cm_model",
    "use_weights": True,
    "weights": {"purple_alien": 0.7, "blue_cat": 0.3},
    "timestamp": "20260810_120000",
    "deployment_status": "deployed",
    "prediction_format": "dataframe",
    "expected_samples_per_model": 128,
    "steps": list(range(1, 37)),
    "run_type": "calibration",
}

PARTITION_DICT = {
    "calibration": {"train": (121, 444), "test": (445, 492)},
}

ARGS = ForecastingModelArgs(run_type="calibration", train=True, eval_type="standard")


def _model_path() -> MagicMock:
    mock = MagicMock()
    mock.model_name = "test_ensemble"
    mock.target = "ensemble"
    mock.root = Path("/test/root")
    return mock


def _build(manager_cls, config: dict):
    """Drive the real `_build_context` with the private state it actually reads.

    Deliberately bypasses `__init__` — the constructors pull in wandb, logging, the
    configuration manager and four ADR-045 stages, none of which `_build_context` touches.
    Constructing them would test the fixtures, not the method.
    """
    manager = object.__new__(manager_cls)
    manager._config_manager = MagicMock()
    manager._config_manager.get_combined_config.return_value = config
    manager._ensemble_path = _model_path()
    manager._partition_dict = PARTITION_DICT
    return manager._build_context(ARGS)


#: Fields both managers must produce identically, with their expected values.
#: Written out rather than computed, so a change in the derivation is visible here.
SHARED_EXPECTATIONS = {
    "configs": SHARED_CONFIG,
    "run_type": "calibration",
    "project": "test_ensemble_calibration",
    "eval_type": "standard",
    "args": ARGS,
    "models": ["purple_alien", "blue_cat"],
    "aggregation": "mean",
    # regression first, then classification — `combined_targets`, #380/#422
    "targets": ["lr_sb_best", "lr_ns_best", "by_sb_best"],
    "reconciliation": "pgm_cm_point",
    "reconcile_with": "some_cm_model",
    "use_weights": True,
    "weights": {"purple_alien": 0.7, "blue_cat": 0.3},
    "timestamp": "20260810_120000",
    "deployment_status": "deployed",
    "partition_dict": PARTITION_DICT,
}

#: The two fields that genuinely differ, and are expected to keep differing. #432 must
#: parameterise on exactly these — if the list grows, the unification changed behaviour.
DIVERGENT_EXPECTATIONS = {
    PredictionFrameEnsembleManager: {
        # a literal, not read from config: this manager only ever emits PredictionFrames
        "prediction_format": "prediction_frame",
        "expected_samples_per_model": 128,
    },
    DataFrameEnsembleManager: {
        "prediction_format": "dataframe",
        # never passed by this manager, so the dataclass default stands
        "expected_samples_per_model": None,
    },
}

#: `model_path` is compared by identity, not value — it is the injected mock.
_IDENTITY_FIELDS = {"model_path"}


MANAGERS = [
    pytest.param(PredictionFrameEnsembleManager, id="prediction_frame"),
    pytest.param(DataFrameEnsembleManager, id="dataframe"),
]


def test_every_field_is_characterized():
    """No `EnsembleContext` field may go uncompared by the tests below.

    This is the guard that keeps the characterization honest. Add a field to
    `EnsembleContext` without adding it here and this fails — which is the only reason
    the tables below can be trusted as *complete* rather than merely *present*.
    """
    declared = {f.name for f in dataclasses.fields(EnsembleContext)}
    covered = set(SHARED_EXPECTATIONS) | _IDENTITY_FIELDS
    for divergent in DIVERGENT_EXPECTATIONS.values():
        covered |= set(divergent)

    assert declared == covered, (
        f"uncompared EnsembleContext field(s): {sorted(declared - covered)}; "
        f"stale entries: {sorted(covered - declared)}. An uncompared field is where a "
        f"refactor hides a behaviour change — add it to SHARED_EXPECTATIONS or to "
        f"DIVERGENT_EXPECTATIONS."
    )


@pytest.mark.parametrize("manager_cls", MANAGERS)
def test_shared_fields_are_identical_across_both_managers(manager_cls):
    """The 16 arguments #432 says are byte-identical. Pinned by value, not by diff."""
    ctx = _build(manager_cls, dict(SHARED_CONFIG))

    actual = {name: getattr(ctx, name) for name in SHARED_EXPECTATIONS}
    assert actual == SHARED_EXPECTATIONS


@pytest.mark.parametrize("manager_cls", MANAGERS)
def test_divergent_fields_keep_their_per_manager_values(manager_cls):
    """The 2 arguments that genuinely differ. #432 parameterises on exactly these."""
    ctx = _build(manager_cls, dict(SHARED_CONFIG))

    expected = DIVERGENT_EXPECTATIONS[manager_cls]
    actual = {name: getattr(ctx, name) for name in expected}
    assert actual == expected


@pytest.mark.parametrize("manager_cls", MANAGERS)
def test_model_path_is_passed_through_unchanged(manager_cls):
    """Compared by identity: `_build_context` must pass the path object, not rebuild it."""
    manager = object.__new__(manager_cls)
    manager._config_manager = MagicMock()
    manager._config_manager.get_combined_config.return_value = dict(SHARED_CONFIG)
    path = _model_path()
    manager._ensemble_path = path
    manager._partition_dict = PARTITION_DICT

    assert manager._build_context(ARGS).model_path is path


@pytest.mark.parametrize("manager_cls", MANAGERS)
def test_optional_keys_fall_back_to_their_documented_defaults(manager_cls):
    """The other half of the contract: what happens when the config omits everything.

    `SHARED_CONFIG` sets every optional key to a non-default value, which pins the read
    path but says nothing about the fallbacks. Both are behaviour, and a unification that
    dropped a `.get()` default would pass the tests above.

    `deployment_status`'s silent `'shadow'` is pinned here as *current behaviour*, not as
    endorsement — #432 notes it should end up with exactly one site, and whether it should
    default at all is ADR-017 follow-up.
    """
    minimal = {
        "name": "test_ensemble",
        "models": ["purple_alien"],
        "aggregation": "mean",
        "regression_targets": ["lr_sb_best"],
    }
    manager = object.__new__(manager_cls)
    manager._config_manager = MagicMock()
    manager._config_manager.get_combined_config.return_value = minimal
    manager._ensemble_path = _model_path()
    manager._partition_dict = None  # the `or {}` branch
    ctx = manager._build_context(ARGS)

    assert ctx.reconciliation is None
    assert ctx.reconcile_with is None
    assert ctx.use_weights is False
    assert ctx.weights == {}
    assert ctx.timestamp == ""
    assert ctx.deployment_status == "shadow"
    assert ctx.partition_dict == {}
    assert ctx.targets == ["lr_sb_best"]


def test_the_two_managers_agree_on_everything_except_the_declared_divergences():
    """Stated as its own test because it is #432's actual premise.

    If this fails, the unification is being attempted on a false description of the
    difference, and the parameterisation will be wrong.
    """
    pf = _build(PredictionFrameEnsembleManager, dict(SHARED_CONFIG))
    df = _build(DataFrameEnsembleManager, dict(SHARED_CONFIG))

    differing = {
        f.name
        for f in dataclasses.fields(EnsembleContext)
        if f.name not in _IDENTITY_FIELDS
        and getattr(pf, f.name) != getattr(df, f.name)
    }

    assert differing == {"prediction_format", "expected_samples_per_model"}, (
        f"the two managers differ on {sorted(differing)}, not on the two fields #432 "
        f"is parameterised for."
    )
