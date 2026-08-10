"""What is left open from the falsification audit of the #422/#367 splash-zone claim.

Run: `conda run -n views_pipeline pytest tests/test_falsification_gate_pooling_splash_zone.py -q`

The claim under test was *"we understand the underlying cause, the issues, and the splash
zone 100%."* It did not survive. Three findings came out of the audit; this file keeps the
**one that is still open**, and records where the other two went, because a finding that is
fixed belongs with the code it guards, not in an audit scratchpad.

## F1 — open. The fix is in another repo.

views-models#367 adds `classification_targets` to `rusty_bucket`'s ensemble config **without
a classification metric key**. `CoreConfigSniffer._check_targets_and_metrics` raises on
exactly that combination, and `sniff_all` runs it before any side effect in both ensemble
managers. So #367 as written fails at config load — independently of whether #422 merged.

Neither the authoring agent nor the reviewer spotted this; both asserted a *silent* failure
mode, and it is loud. The strict xfail below goes green when views-models#367 adds
`"classification_sample_metrics": ["AP"]` alongside the targets it introduces. Deleting this
file is the last step of that work, not a way to quiet it.

## F2 — closed by #422, and the coverage moved

The reviewer's claim that #422 "cannot fix the symptom it says it fixes" was wrong in
substance: views-hydranet's EXP-03 measured the effect directly, and pooling the gate channel
moves AP at h1 from 0.316 → 0.456 (sb), 0.177 → 0.355 (ns), 0.135 → 0.225 (os). Both managers
now derive the pooled target list via `combined_targets`.

Pinned in `tests/test_managers/test_prediction_frame_ensemble_manager.py`
(`TestBuildContextPoolsGateChannel`, `TestPoolEmitsGateChannelEndToEnd`) and in
`tests/test_managers/test_dataframe_ensemble_manager.py` (`TestBuildContextPoolsGateChannel`).

The DataFrame half had **no** coverage until this branch: #422 applied the identical one-line
fix to two duplicated `_build_context` bodies and tested one. That is the failure mode #432
exists to remove.

The stub that used to sit here was also broken. It asserted a hardcoded copy of the *old*
expression against `combined_targets`, which disagree by construction whenever a
classification target exists — so it could not have gone green for any fix, and gated nothing.

## F3 — not a defect. It is the intended design.

The workaround that *proved* the fix added a legacy `targets` key, and `combined_targets`
refuses that key. That refusal is #380's deliberate choice: a stale key must not outrank the
split ones, which is how the gate went missing in the first place. Pinned as behaviour in
both manager test files (`test_stale_targets_key_fails_loud`), not carried here as an xfail
waiting for a fix that must never come.

The original stub also claimed views-hydranet "holds a working hack that is about to become
an error". Checked: hydranet has **no** production code carrying the key. The only occurrence
is a toy fixture at `tests/test_eval_integration_toy.py:46`, unconnected to
`combined_targets`. The hack lived in the EXP-03 experiment configuration, not in committed
code, so nothing in that repo is on a path to erroring out. The claim was consistent with the
evidence and not established by it — C-273, once more, inside the audit written to catch it.
"""

from __future__ import annotations

import pytest

from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer


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
}


@pytest.mark.xfail(
    reason="F1: views-models#367 as written fails this check. The fix belongs in that PR — "
    "add `classification_sample_metrics` alongside the `classification_targets` it "
    "introduces. Delete this file when it lands.",
    strict=True,
)
def test_f1_the_config_views_models_367_writes_is_accepted():
    """#367's ensemble config must load. Today it does not.

    Declaring `classification_targets` obliges a classification metric key. #367 declares
    the targets and no metric, so the sniffer refuses the config before the ensemble runs.

    This fails whether or not #422 merged — the two PRs are not merely order-sensitive,
    the consumer half is broken on its own.
    """
    _sniffer(dict(RUSTY_BUCKET_AFTER_367))._check_targets_and_metrics()


def test_f1_control_the_same_config_with_a_metric_is_accepted():
    """The control. If this also failed, F1 would be about something else — a config the
    sniffer rejects for an unrelated reason would produce the same xfail above."""
    config = {**RUSTY_BUCKET_AFTER_367, "classification_sample_metrics": ["AP"]}
    _sniffer(config)._check_targets_and_metrics()


def test_f1_control_todays_config_is_accepted():
    """rusty_bucket as it stands loads fine — the defect is introduced by #367, not
    pre-existing. Without this, F1 could be blaming the wrong change."""
    config = {k: v for k, v in RUSTY_BUCKET_AFTER_367.items() if k != "classification_targets"}
    _sniffer(config)._check_targets_and_metrics()
