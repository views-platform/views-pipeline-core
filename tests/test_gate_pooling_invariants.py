"""Invariants surviving the #422/#427 gate-pooling incident, after its audit closed.

Run: `conda run -n views_pipeline pytest tests/test_gate_pooling_invariants.py -q`

## The audit is over

This was `test_falsification_gate_pooling_splash_zone.py`, the executable record of a
falsification audit of the claim *"we understand the underlying cause, the issues, and the
splash zone 100%."* It did not survive; three findings came out of it.

All three are now closed:

- **F1** — views-models#367 declared `classification_targets` with no classification
  metric key, which `CoreConfigSniffer` refuses at load. It was carried here as a
  `strict=True` xfail. **Resolved 2026-08-11 by views-models#383**, which shipped
  `rusty_bucket` with the roster and all three declaration lines. The xfail is deleted
  rather than relaxed, per the epic's own definition of done — and it could never have
  gone green, because #367's broken shape never merged. Its replacement is a conformance
  test against the config that *did*: `test_views_models_conformance.py`, vendoring
  `rusty_bucket` at views-models@085a6230.
- **F2** — the pooling fix itself. Landed in #422, pinned in both managers' tests.
- **F3** — the retired `targets` key. Pinned as intended behaviour in both managers.

## Why the file survives the audit that created it

Three invariants here guard live code and belong to no single manager, so moving them
into one manager's test file would hide half of what they assert:

1. `CoreConfigSniffer` accepts metric names views-evaluation refuses (**C-287, open**).
2. Declaring a target no constituent produces fails **loud** on the PredictionFrame path.
3. The same, on the DataFrame path.

(2) and (3) matter because they are C-286 inverted: the original defect dropped a declared
channel silently. If a later change made a missing channel silently tolerated instead of
raising, the pool would quietly contain fewer channels than the config declares — and the
tolerance would look like robustness.
"""

from __future__ import annotations

import pytest

from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer


def _native_evaluator():
    """views-evaluation's own config gate, or a skip if it is not installed."""
    pytest.importorskip(
        "views_evaluation",
        reason="the second gate lives in views-evaluation; without it there is only one "
        "gate to check and this control cannot do its job.",
    )
    from views_evaluation.evaluation.native_evaluator import NativeEvaluator

    return NativeEvaluator


def _sniffer(config: dict) -> CoreConfigSniffer:
    instance = object.__new__(CoreConfigSniffer)
    instance._c = config
    return instance


RUSTY_BUCKET_AFTER_367 = {
    "name": "rusty_bucket",
    "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample"],
    # NOTE: no classification metric key — this is what #367 actually writes.
    #
    # `steps` and `evaluation_profile` are not part of what #367 changes. They are here
    # because views-evaluation's `_validate_config` requires them, and the controls below
    # run the config through that gate as well as this repo's — which is the correction
    # views-models#372 asked for. Without them the second gate fails for an unrelated
    # reason and the control proves nothing about the metric key.
    "steps": list(range(1, 37)),
    "evaluation_profile": "hydranet_ucdp",
}

#: The metric keys views-models decided on, recorded in their
#: `tests/test_roster_conformance.py` and merged as views-models#376.
#:
#: **Both**, not one. `Brier_cls_sample` is what the eight constituent models already
#: declare, so it restores existing behaviour rather than adding any. `AP` is additional,
#: and goes under `classification_point_metrics` because `AP` lives in
#: `METRIC_MEMBERSHIP[("classification", "point")]` — putting it under the sample key is
#: the mistake this repo made and views-models#372 caught.
#:
#: Mirrored here so the control below tests the config that will actually ship, not a
#: plausible one. A control that certifies a shape nobody deploys certifies nothing.
DECIDED_METRIC_KEYS = {
    "classification_point_metrics": ["AP"],
    "classification_sample_metrics": ["Brier_cls_sample"],
}


# ----------------------------------------------------------------------------------
# The gap views-models#372 exposed: two gates, and the earlier one is weaker
# ----------------------------------------------------------------------------------


def test_the_sniffer_accepts_metric_names_the_evaluator_rejects():
    """`CoreConfigSniffer` checks a metric key is PRESENT, never that its contents are VALID.

    `_check_targets_and_metrics` asks only whether one of `CLASSIFICATION_METRIC_KEYS` is
    non-empty. views-evaluation's `NativeEvaluator._validate_config` additionally requires
    every named metric to be valid for the `(task, kind)` cell its key declares. So a
    config can clear this repo's gate and be refused by the next one — and the operator
    discovers it at evaluation time, after training has run.

    Found by views-models#372 while reviewing a fix *this repo recommended*: the suggested
    `classification_sample_metrics: ["AP"]` passes here and fails there, because `AP` is a
    classification **point** metric.

    Pinned as current behaviour, not fixed here — teaching the sniffer to validate names
    is a new failure mode and belongs in its own change, filed separately. This test fails
    the day that lands, which is the signal to delete it.
    """
    evaluator = _native_evaluator()
    config = {**RUSTY_BUCKET_AFTER_367, "classification_sample_metrics": ["AP"]}

    # gate one: passes
    _sniffer(dict(config))._check_targets_and_metrics()

    # gate two: refuses
    with pytest.raises(ValueError, match="not valid for"):
        evaluator._validate_config(config)


# ----------------------------------------------------------------------------------
# The inverse direction, raised by views-models#372: declaring a channel the members
# do not produce
# ----------------------------------------------------------------------------------


def test_declaring_a_target_no_member_produces_fails_loud_not_silent():
    """views-models' third blocker, answered from this side.

    None of `rusty_bucket`'s eight `temporary_*` stand-ins declares
    `classification_targets`, so declaring the `by_*` gate on the ensemble before the
    roster is rewired would claim a channel its constituents do not produce. views-models
    described the consequence as "a config that passes both gates and pools nothing".

    It is worse-sounding and better-behaved than that: **both managers refuse.** The
    PredictionFrame path raises naming the model and the target
    (`prediction_frame_ensemble.py:636`); the DataFrame path raises from
    `AggregationModule` naming the missing column. Verified by driving both, not by
    reading them.

    That matters for their sequencing decision. Shipping the declaration early is an
    error, not a wrong number — the opposite failure mode from C-286, which is the whole
    point of this epic. It is still a blocker; it is not a silent one.

    Pinned because nothing asserted it, and "pools nothing" is exactly the behaviour a
    future refactor might introduce while believing it was being tolerant.
    """
    import numpy as np
    import pandas as pd

    from views_pipeline_core.modules.aggregation.aggregator import AggregationModule

    # The DataFrame path: a declared target with no matching column.
    index = pd.MultiIndex.from_product(
        [[1, 2], [100, 101]], names=["month_id", "priogrid_id"]
    )
    frame = pd.DataFrame({"pred_lr_sb_best": np.zeros(4)}, index=index)
    aggregator = AggregationModule(
        index_cols=["month_id", "priogrid_id"],
        target_cols=["pred_lr_sb_best", "pred_by_sb_best"],
    )
    with pytest.raises(ValueError, match="Missing target columns"):
        aggregator.add_model(data=frame, weight=None, name="temporary_stand_in")
        aggregator.aggregate(method="mean", use_weights=False)


def test_the_prediction_frame_path_also_refuses_rather_than_dropping():
    """The same invariant on the path `rusty_bucket` actually uses.

    Asserted against the source rather than by constructing a full ensemble run: the
    guard is a single `if pf is None: raise` inside `_forecast_ensemble`, and standing up
    the surrounding machinery would test the fixtures. Its absence is what would matter,
    and its absence is what this detects.
    """
    import inspect

    from views_pipeline_core.managers.ensemble import prediction_frame_ensemble

    source = inspect.getsource(prediction_frame_ensemble.PredictionFrameEnsembleManager._forecast_ensemble)
    assert "did not produce a forecast" in source, (
        "`_forecast_ensemble` no longer raises when a constituent is missing a declared "
        "target. A silently skipped target is exactly C-286 in the other direction: the "
        "pool would quietly contain fewer channels than the config declares."
    )
