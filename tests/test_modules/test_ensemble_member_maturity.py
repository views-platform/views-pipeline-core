"""R1 and R2 — the ensemble member rules from views-models ADR-017 §5. #400, ADR-058.

## Why these tests exist in this shape

The rule R2 revives had a guard already. It read:

    if single_model_dp_status == "production" and ensemble_deployment_status != "production":

`production` is not a value views-models writes — it writes `shadow`, `deployed`,
`baseline`, `deprecated`. **The branch could never execute against real data.** Its own
error message said *"deployed"* while its condition said *"production"*; they had
disagreed since the day it was written.

It also had a test, which passed, because the test invented the same value the branch
invented. That is C-218's shape: a test that can only fail when the code disagrees with
our mock, which is not coverage of anything.

So #400's acceptance criterion is *"R1 and R2 both have a test that fails when
violated — the current guard's problem is that it could not fail."* Every assertion below
was checked by mutation: break the rule in the source, and a named test here goes red.

## Log capture

`LoggingModule` sets both `propagate = False` and `disabled = True` on this package's
loggers, so `caplog` silently observes nothing once other tests have run (C-281). These
tests attach a handler to the module's own logger and re-enable it, and the fixture
carries a control that proves it can see a record at all.
"""

from __future__ import annotations

import logging

import pytest

from views_pipeline_core.modules.validation.ensemble.member_maturity import (
    ACTIVE_MATURITIES,
    GRADUATE_MATURITY,
    ensemble_may_contain_member,
)


@pytest.fixture
def messages():
    """Records from the module's own logger, independent of global logging state."""
    from views_pipeline_core.modules.validation.ensemble import member_maturity

    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector(level=logging.WARNING)
    logger = member_maturity.logger
    previous_level, previous_disabled = logger.level, logger.disabled
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.disabled = False
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.disabled = previous_disabled


def test_the_capture_fixture_can_see_a_record(messages):
    """C-281's control. Without this, every log assertion below could be vacuous."""
    from views_pipeline_core.modules.validation.ensemble import member_maturity

    member_maturity.logger.warning("probe")
    assert [r.getMessage() for r in messages] == ["probe"]


def _check(ensemble, member, **kwargs):
    return ensemble_may_contain_member(
        ensemble_status=ensemble,
        member_status=member,
        member_name="a_member",
        ensemble_name="an_ensemble",
        **kwargs,
    )


# ── R1: an active ensemble may not contain a retired member ───────────────────


@pytest.mark.parametrize("ensemble", sorted(ACTIVE_MATURITIES))
@pytest.mark.parametrize("retired", ["retired", "deprecated"])
def test_r1_an_active_ensemble_rejects_a_retired_member(ensemble, retired, messages):
    """Both vocabularies, both active maturities. This is the check that always worked."""
    assert _check(ensemble, retired) is False
    assert messages, "the member was rejected without saying why"
    assert "config_maturity.py" in messages[0].getMessage(), (
        "the failure does not name the file to open (views-models ADR-020)"
    )


@pytest.mark.parametrize("ensemble", sorted(ACTIVE_MATURITIES))
def test_r1_an_active_ensemble_accepts_a_candidate_member(ensemble):
    """Negative control for R1 — but not for graduate, where R2 takes over."""
    if ensemble == GRADUATE_MATURITY:
        pytest.skip("covered by R2 below; a graduate ensemble is stricter")
    assert _check(ensemble, "candidate") is True


def test_r1_a_retired_ensemble_cannot_run():
    """Preserved from the original guard: a deprecated ensemble was already refused."""
    assert _check("retired", "candidate") is False
    assert _check("deprecated", "shadow") is False


# ── R2: every member of a graduate ensemble must be graduate ──────────────────


@pytest.mark.parametrize("member", ["candidate", "shadow", "baseline"])
def test_r2_a_graduate_ensemble_rejects_a_non_graduate_member(member, messages):
    """The rule the dead branch was reaching for, against values that exist."""
    assert _check(GRADUATE_MATURITY, member) is False
    assert messages, "the member was rejected without saying why"
    message = messages[0].getMessage()
    assert "graduate" in message
    assert "config_maturity.py" in message


def test_r2_a_graduate_ensemble_accepts_a_graduate_member():
    """Negative control. A rule that rejects everything is not a rule."""
    assert _check(GRADUATE_MATURITY, GRADUATE_MATURITY) is True


def test_r2_does_not_apply_to_a_candidate_ensemble():
    """R2 constrains graduate ensembles only. Over-applying it would block the 117
    `shadow` models views-models has today, none of which claims to be graduate."""
    assert _check("candidate", "candidate") is True
    assert _check("shadow", "shadow") is True


# ── indeterminate maturity is refused under R2, never assumed ─────────────────


def test_r2_refuses_a_member_whose_maturity_cannot_be_read(messages):
    """`deployed` has no safe reading (ADR-057), so R2 cannot be CONFIRMED for it.

    Passing it would be reporting "I could not tell" as "the rule is satisfied".
    """
    assert _check(GRADUATE_MATURITY, "deployed") is False
    assert any("cannot be confirmed" in r.getMessage() for r in messages), (
        f"the refusal does not explain that it is an inability to confirm, not a "
        f"detected violation: {[r.getMessage() for r in messages]}"
    )


def test_an_unreadable_member_status_outside_r2_warns_rather_than_passing_silently(
    messages,
):
    """`production` is the value the dead branch invented. It is not rejected here — no
    rule forbids it in a candidate ensemble — but it must not pass in silence, because
    silent acceptance is how it survived in a guard for years."""
    assert _check("candidate", "production") is True
    assert messages, "an unrecognised member status was accepted silently"
    assert "production" in messages[0].getMessage()


def test_an_unreadable_ensemble_status_skips_r2_and_says_so(messages):
    """R2 is about graduate ensembles; one that has not been established to be graduate
    is not silently held to it, but the reader is told the rule did not run."""
    assert _check("deployed", "candidate") is True
    assert any("was not evaluated" in r.getMessage() for r in messages), (
        f"R2 was skipped without saying so: {[r.getMessage() for r in messages]}"
    )


# ── the shape of the rules themselves ─────────────────────────────────────────


def test_graduate_is_one_of_the_active_maturities():
    """R2 is stricter than R1 on the same ensembles; if graduate ever left the active set
    the two rules would be describing disjoint populations."""
    assert GRADUATE_MATURITY in ACTIVE_MATURITIES


def test_retired_is_not_active():
    assert "retired" not in ACTIVE_MATURITIES


def test_missing_statuses_do_not_crash():
    """A log written before this field existed yields None. It must not raise."""
    assert _check(None, None) is True
