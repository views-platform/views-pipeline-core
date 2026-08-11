"""A real views-models ensemble config must survive this repo's own gates. #430.

Run: `conda run -n views_pipeline pytest tests/test_modules/test_views_models_conformance.py -q`

## The boundary the incident went through

An agent working in views-hydranet pushed a fix to this repository (#422, self-disclosed
in #427) because a *targets* change needed three repos at once and no check on any side
would have caught the mismatch. views-models was probe-only: this repo reads 35-odd string
keys off an untyped config dict that another repo writes, and nothing verified that a real
config from that repo still satisfies them.

This closes that from this side. views-models#371 is the reciprocal check.

## Why a vendored fixture, and why not rusty_bucket

views-models is not a Python package — it is a repository of config scripts. There is
nothing to `pip install` and interrogate, so the only honest artifact is a **verbatim copy
of a real config**, pinned to the commit it came from:
`tests/fixtures/views_models/white_mustang_configs.py`.

`white_mustang` deliberately, not `rusty_bucket`. rusty_bucket is the ensemble the incident
concerns, and it is mid-rebase on views-models#367 — where it currently declares
`classification_targets` with **no** classification metric key, which `CoreConfigSniffer`
refused outright. **Resolved 2026-08-11 by views-models#383**, which is why the
gated shape is now vendored below.
Freezing it now would capture either a broken shape or a moving branch. `white_mustang` is
merged and stable.

The trade is stated rather than hidden: this fixture exercises the **regression-only**
config shape. The gated shape that caused the incident is covered by the falsification
test until #367 lands, at which point rusty_bucket should be vendored here too.

## What "conformance" means here

Not "the fixture equals a snapshot" — that only detects that someone changed the fixture.
It means **a real config from the neighbour passes the gates this repo will actually run
it through**: `CoreConfigSniffer`, `combined_targets`, and `EnsembleContext.from_config`.
If this repo tightens a rule that real configs do not satisfy, this fails here rather than
in someone's production run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.managers.configuration.configuration import combined_targets
from views_pipeline_core.managers.ensemble.context import EnsembleContext
from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "views_models" / "white_mustang_configs.py"


def _fixture_module():
    spec = importlib.util.spec_from_file_location("_vendored_views_models", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vendored():
    if not FIXTURE.exists():  # pragma: no cover - the fixture is committed
        pytest.fail(
            f"the vendored views-models config is missing at {FIXTURE}. This test cannot "
            f"degrade to a skip: without the neighbour's real artifact there is nothing "
            f"being conformed to, and a green run would mean nothing."
        )
    return _fixture_module()


@pytest.fixture(scope="module")
def combined(vendored):
    """The merged config, assembled the way `EnsembleManager` assembles one.

    `config_modelset` merges into `config_meta` with modelset taking precedence
    (`managers/ensemble/ensemble.py`), and `config_deployment` contributes
    `deployment_status`.
    """
    config = dict(vendored.get_meta_config())
    config.update(vendored.get_deployment_config())
    config.update(vendored.get_modelset_config())
    return config


# ----------------------------------------------------------------------------------
# The fixture must be a real artifact before anything asserted about it means something
# ----------------------------------------------------------------------------------


def test_the_fixture_records_where_it_came_from(vendored):
    """A vendored copy with no provenance is indistinguishable from one somebody invented.

    Three outcomes, not two: no fixture is a **failure** (above), a fixture without
    provenance is a **failure** (here), and a fixture with provenance is verified.
    """
    assert vendored.SOURCE_REPO == "views-models"
    assert vendored.SOURCE_ENSEMBLE == "white_mustang"
    assert len(vendored.SOURCE_COMMIT) >= 7, (
        f"SOURCE_COMMIT is {vendored.SOURCE_COMMIT!r} — a fixture that cannot be traced "
        f"back to a commit cannot be refreshed or audited."
    )


def test_the_fixture_exposes_the_three_config_callables(vendored):
    """The names this repo loads by string through `load_config_from_script`.

    `managers/ensemble/ensemble.py` asks for `get_modelset_config`,
    `managers/model/model.py` for `get_meta_config` and `get_deployment_config`. A rename
    on the neighbour's side breaks config loading at construction.
    """
    for method in ("get_meta_config", "get_deployment_config", "get_modelset_config"):
        assert callable(getattr(vendored, method, None)), (
            f"the vendored config does not define `{method}()`. This repo loads it by "
            f"that exact name."
        )


# ----------------------------------------------------------------------------------
# The conformance assertions: this repo's gates, run on the neighbour's real config
# ----------------------------------------------------------------------------------


def test_a_real_ensemble_config_passes_the_config_sniffer(combined):
    """`CoreConfigSniffer.sniff_all` runs before any side effect in both ensemble managers.

    If this repo tightens a rule that real views-models configs do not satisfy, the
    failure belongs here — at merge, naming the rule — not in an operator's run.
    """
    sniffer = object.__new__(CoreConfigSniffer)
    sniffer._c = dict(combined)
    sniffer._check_targets_and_metrics()


def test_the_targets_this_repo_derives_match_what_the_config_declares(combined):
    """`combined_targets` is what #422 routed `_build_context` through.

    It also **raises** on the retired `targets` key, so this doubles as a check that no
    live views-models config has resurrected it.
    """
    assert combined_targets(combined) == combined["regression_targets"]


def test_a_real_config_builds_an_ensemble_context(combined):
    """The end of the line: the neighbour's config, through this repo's real constructor.

    Exercises every `configs[...]` read `EnsembleContext.from_config` performs — the
    required keys (`name`, `models`, `aggregation`) and the optional ones. A key this repo
    requires and views-models stopped writing shows up as a `KeyError` naming it.
    """
    ctx = EnsembleContext.from_config(
        combined,
        model_path=None,
        args=ForecastingModelArgs(run_type="calibration", train=True),
        partition_dict=None,
        prediction_format="dataframe",
    )

    assert ctx.models == combined["models"]
    assert ctx.aggregation == combined["aggregation"]
    assert ctx.deployment_status == combined["deployment_status"]
    assert ctx.reconciliation == combined["reconciliation"]


def test_the_required_keys_are_present_rather_than_defaulted(combined):
    """The keys this repo would otherwise silently default.

    `from_config` supplies `"shadow"` for a missing `deployment_status` and `{}` for
    missing weights. A real config that stopped declaring them would still build a
    context — and the run would proceed under a default nobody chose. That is the
    silent-substitution shape this epic exists to remove, so the presence is asserted
    separately from the value.
    """
    for key in ("name", "models", "aggregation", "level", "deployment_status"):
        assert key in combined, (
            f"a real views-models ensemble config no longer declares `{key}`. This repo "
            f"reads it, and for `deployment_status` would silently substitute 'shadow'."
        )


# ----------------------------------------------------------------------------------
# The gated shape — the one the incident was actually about
# ----------------------------------------------------------------------------------

GATED_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "views_models" / "rusty_bucket_configs.py"
)


@pytest.fixture(scope="module")
def gated():
    """`rusty_bucket` as views-models#383 shipped it — the ensemble that declares a gate.

    `white_mustang` is regression-only and cannot exercise this path. Until 2026-08-11 the
    gated shape did not exist to vendor: rusty_bucket ran on eight `temporary_*` stand-ins
    that declared no `classification_targets`, and views-models#367's attempt declared them
    with no classification metric key, which `CoreConfigSniffer` refuses.

    A previous version of this file carried a test asserting the gated shape was NOT
    covered here, so that vendoring one would fail and force this docstring to be
    corrected rather than left lying. It did. This is that correction.
    """
    if not GATED_FIXTURE.exists():  # pragma: no cover - the fixture is committed
        pytest.fail(f"the vendored gated config is missing at {GATED_FIXTURE}")
    spec = importlib.util.spec_from_file_location("_vendored_rusty_bucket", GATED_FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = dict(module.get_meta_config())
    config.update(module.get_deployment_config())
    config.update(module.get_modelset_config())
    return config


def test_the_gated_config_declares_the_channel_the_incident_was_about(gated):
    """The `by_*` occurrence channel, declared — which is what makes the fix do anything.

    #422 stopped the pool dropping a declared gate. That fix is inert until an ensemble
    declares one. This asserts the real config now does.
    """
    assert gated["classification_targets"] == ["by_sb_best", "by_ns_best", "by_os_best"]


def test_the_gated_config_passes_this_repos_gate(gated):
    """`CoreConfigSniffer` — the check views-models#367's version failed."""
    sniffer = object.__new__(CoreConfigSniffer)
    sniffer._c = dict(gated)
    sniffer._check_targets_and_metrics()


def test_the_gated_config_pools_the_gate_channel(gated):
    """End of the line: the declared gate reaches the pooled target list.

    Regression first, then classification. Before #422 this returned only the three
    `lr_*` magnitudes and the ensemble's AP was understated with no error anywhere.
    """
    ctx = EnsembleContext.from_config(
        gated,
        model_path=None,
        args=ForecastingModelArgs(run_type="calibration", train=True),
        partition_dict=None,
        prediction_format="prediction_frame",
        expected_samples_per_model=16,
    )
    assert ctx.targets == [
        "lr_sb_best", "lr_ns_best", "lr_os_best",
        "by_sb_best", "by_ns_best", "by_os_best",
    ]


def test_the_gated_config_also_satisfies_views_evaluation(gated):
    """Two gates, not one — the lesson of C-287.

    `CoreConfigSniffer` checks a metric key is *present*; views-evaluation additionally
    requires each metric to be valid for the cell its key names. This repo recommended
    `classification_sample_metrics: ["AP"]`, which clears the first and fails the second
    (`AP` is a classification *point* metric). views-models#372 caught it. Asserting both
    here is what stops that advice being given again.
    """
    pytest.importorskip("views_evaluation", reason="the second gate needs the neighbour")
    from views_evaluation.evaluation.native_evaluator import NativeEvaluator

    config = {**gated, "steps": list(range(1, 37))}
    NativeEvaluator._validate_config(config)
