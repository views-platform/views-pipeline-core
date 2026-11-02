"""Falsification stubs for: "nothing more is needed in pipeline-core to run chrome_box".

Everything that could be probed about CORRECTNESS survived (config validation for both
ensembles and all sixteen members, `concat` at 8x128, the `pgm_cm` path, the reconciler
injection seam, and the full `load_cm_frame` -> `reconcile_frames` handoff for all three
targets against the installed 3.3.5). The surviving attack is CAPACITY, not correctness.

These are RED on purpose. They encode #563 — which is open, structurally confirmed, and
sits on chrome_box's exact shape (8 members x 3 targets at pgm x 128 draws). Delete them
only by fixing #563, never by relaxing them.
"""
import inspect

import pytest

from views_pipeline_core.managers.ensemble import prediction_frame_ensemble as pfe


@pytest.mark.xfail(
    reason="#563: member frames are held from the member loop through pool, "
    "reconcile and publish. Soft falsification — headroom, not incorrectness.",
    strict=True,
)
def test_forecast_ensemble_frees_member_frames_before_pooling():
    """`_forecast_ensemble` must not hold every member x target frame through pooling.

    chrome_box is the shape this bites: 8 members, 3 targets, pgm, 128 draws each.
    Pooling consumes `model_results` target-by-target, so a transpose to
    target-outer/member-inner would let each member's frames die after their target is
    pooled. Nothing frees them today — asserted structurally because the alternative is
    a memory measurement, and #563 says explicitly that the measurement comes first.
    """
    source = inspect.getsource(pfe.PredictionFrameEnsembleManager._forecast_ensemble)
    frees = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("del ") or "gc.collect" in line
    ]
    assert frees, (
        "no `del` or `gc.collect` between filling model_results and consuming it: "
        "every member x target PredictionFrame is live simultaneously through pool, "
        "reconcile and publish (#563)."
    )


@pytest.mark.xfail(
    reason="#563: same shape in the evaluation path, one dimension worse "
    "(members x targets x origins).",
    strict=True,
)
def test_evaluate_ensemble_frees_member_frames_before_pooling():
    """Same defect, larger footprint: the eval path's value type is a *list* per target."""
    source = inspect.getsource(pfe.PredictionFrameEnsembleManager._evaluate_ensemble)
    frees = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("del ") or "gc.collect" in line
    ]
    assert frees, (
        "_evaluate_ensemble accumulates members x targets x origins with nothing "
        "freed between the member loop and pooling (#563)."
    )
