"""Both the maturity and the legacy deployment vocabulary are accepted, for one window.

Issue #399, epic #398, ADR-057. Origin: views-models ADR-017.

## What changed upstream

views-models declared `deployment_status` on every model, one of
`{shadow, deployed, baseline, deprecated}`. ADR-017 found it answering three unrelated
questions at once — operational mode, lifecycle, and role — and found it **inert**:
nothing in the platform branched on it. Measured across their 128 config files on
2026-08-08: 117 `shadow`, 6 `baseline`, 4 `deprecated`, 1 `deployed`.

It becomes `maturity`, one of `{candidate, graduate, retired}`, in a file renamed
`config_deployment.py` -> `config_maturity.py`.

## Why both are accepted

Not as a deprecation period. This is being executed now, not carried. The window exists so
the two repositories do not have to land in the same minute — a flag-day rename would
break every consuming repo at once, and there are 128 configs on the other side of it.

The window closes when views-models reports no configs on the legacy vocabulary. That is a
number they can measure, rather than a date we would both forget.

## `deployed` is the one that cannot be translated

ADR-017 makes `deployed -> graduate` conditional on its own rule R2: every member of a
graduate ensemble must itself be graduate. Measured on 2026-08-08, the sole `deployed`
source is the ensemble `white_mustang`, and its three members — `average_cmbaseline`,
`zero_cmbaseline`, `locf_cmbaseline` — are all `shadow`.

So an automatic `deployed -> graduate` mapping would manufacture a violation of ADR-017's
own rule on the first day it ran. The sniffer refuses to guess and asks for the value to
be set deliberately. That refusal is a rule, so it is tested.
"""

from __future__ import annotations

import logging

import pytest

from views_pipeline_core.modules.validation.core_config_sniffer import (
    LEGACY_STATUS_TO_MATURITY,
    LEGACY_STATUSES_WITHOUT_A_SAFE_MAPPING,
    RETIRED_MATURITY,
    SUPPORTED_DEPLOYMENT_STATUSES,
    SUPPORTED_MATURITIES,
    CoreConfigSniffer,
)


@pytest.fixture
def warnings_emitted():
    """Collect WARNING records from the sniffer's own logger.

    Deliberately not `caplog`. `caplog` observes records that propagate to the root
    logger, and `LoggingModule` sets `propagate = False` on this package's loggers — so
    whether these tests can see anything depends on which other test ran first. Written
    with `caplog` they passed in isolation and captured NOTHING in the full suite: a test
    reporting success because it could not look, which is the failure mode this whole
    branch keeps finding elsewhere.

    Attaching a handler to the module's own logger makes capture independent of the global
    logging configuration.
    """
    from views_pipeline_core.modules.validation import core_config_sniffer

    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector(level=logging.WARNING)
    logger = core_config_sniffer.logger
    previous_level = logger.level
    previous_disabled = logger.disabled
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    # `LoggingModule` DISABLES this package's loggers when it configures them, and a
    # disabled logger drops records before any handler sees them — so attaching a handler
    # is not sufficient on its own. The control test below is what revealed this; without
    # it the whole file would have gone quietly green-when-alone and blind-in-the-suite.
    logger.disabled = False
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.disabled = previous_disabled


def _sniffer(**config):
    """A sniffer carrying only the keys `_check_deployment_status` reads."""
    instance = object.__new__(CoreConfigSniffer)
    instance._c = {"name": "rusty_bucket", **config}
    return instance


def test_the_capture_fixture_can_actually_see_a_warning(warnings_emitted):
    """The control. Every assertion below is worthless if this cannot observe anything."""
    from views_pipeline_core.modules.validation import core_config_sniffer

    core_config_sniffer.logger.warning("probe")
    assert [r.getMessage() for r in warnings_emitted] == ["probe"]


# ── the new vocabulary ────────────────────────────────────────────────────────


@pytest.mark.parametrize("maturity", ["candidate", "graduate"])
def test_new_values_are_accepted_silently(maturity, warnings_emitted):
    _sniffer(maturity=maturity)._check_deployment_status()
    assert not warnings_emitted, (
        f"maturity='{maturity}' is the target vocabulary and warned anyway: "
        f"{[r.getMessage() for r in warnings_emitted]}"
    )


def test_retired_refuses_to_run():
    """The `deprecated` block, restated. A retired model must not silently execute."""
    with pytest.raises(ValueError, match="cannot be run"):
        _sniffer(maturity=RETIRED_MATURITY)._check_deployment_status()


def test_an_unknown_maturity_fails_loud_and_lists_the_valid_set():
    with pytest.raises(ValueError) as excinfo:
        _sniffer(maturity="production")._check_deployment_status()
    message = str(excinfo.value)
    assert "production" in message
    for valid in SUPPORTED_MATURITIES:
        assert valid in message, f"the failure does not list '{valid}' as valid"
    assert "config_maturity.py" in message, "the failure does not name the file to edit"


# ── the legacy vocabulary ─────────────────────────────────────────────────────


@pytest.mark.parametrize("status", sorted(LEGACY_STATUS_TO_MATURITY))
def test_every_mapped_legacy_value_is_accepted_and_warns(status, warnings_emitted):
    """One case per legacy value — derived from the mapping constant, not listed."""
    expected = LEGACY_STATUS_TO_MATURITY[status]

    if expected == RETIRED_MATURITY:
        with pytest.raises(ValueError, match="cannot be run"):
            _sniffer(deployment_status=status)._check_deployment_status()
        return

    _sniffer(deployment_status=status)._check_deployment_status()

    assert warnings_emitted, f"legacy '{status}' was accepted silently"
    message = warnings_emitted[0].getMessage()
    assert expected in message, f"the warning does not say '{status}' reads as '{expected}'"
    assert "config_maturity.py" in message, "the warning does not name the file to edit"


def test_deployed_is_accepted_but_refuses_to_be_read_as_graduate(warnings_emitted):
    """The refusal that stops us manufacturing a views-models R2 violation on day one."""
    _sniffer(deployment_status="deployed")._check_deployment_status()

    assert warnings_emitted, "legacy 'deployed' was accepted silently"
    message = warnings_emitted[0].getMessage()
    assert "graduate" in message, "the warning does not explain what it is refusing to do"
    assert "NOT" in message or "not being read" in message.lower(), (
        "the warning does not make clear that no translation happened"
    )


def test_an_unknown_legacy_value_fails_loud():
    with pytest.raises(ValueError) as excinfo:
        _sniffer(deployment_status="production")._check_deployment_status()
    assert "production" in str(excinfo.value)


def test_a_missing_field_fails_loud_naming_the_new_key():
    with pytest.raises(KeyError) as excinfo:
        _sniffer()._check_deployment_status()
    assert "maturity" in str(excinfo.value)


# ── both at once ──────────────────────────────────────────────────────────────


def test_both_keys_present_prefers_maturity_and_warns(warnings_emitted):
    """Silently ignoring one of two contradictory keys is how the wrong file gets edited."""
    _sniffer(
        maturity="candidate", deployment_status="deployed"
    )._check_deployment_status()

    assert warnings_emitted, "a config declaring both keys was accepted silently"
    message = warnings_emitted[0].getMessage()
    assert "deployment_status" in message and "maturity" in message


def test_maturity_wins_even_when_the_legacy_value_would_have_blocked():
    """Precedence must be unconditional, or it is not precedence."""
    _sniffer(
        maturity="candidate", deployment_status="deprecated"
    )._check_deployment_status()


def test_a_retired_maturity_still_blocks_when_the_legacy_value_is_benign():
    with pytest.raises(ValueError, match="cannot be run"):
        _sniffer(
            maturity=RETIRED_MATURITY, deployment_status="shadow"
        )._check_deployment_status()


# ── the translation table itself ──────────────────────────────────────────────


def test_the_legacy_vocabulary_is_fully_accounted_for():
    """Every old value is either mapped or explicitly flagged as needing a decision.

    Enforced at import by an assert in the sniffer; restated here so the failure is a
    named test rather than a collection error nobody reads.
    """
    accounted = set(LEGACY_STATUS_TO_MATURITY) | LEGACY_STATUSES_WITHOUT_A_SAFE_MAPPING
    assert accounted == SUPPORTED_DEPLOYMENT_STATUSES, (
        f"unaccounted legacy values: {sorted(SUPPORTED_DEPLOYMENT_STATUSES - accounted)}"
    )


def test_no_legacy_value_maps_to_a_maturity_that_does_not_exist():
    unknown = set(LEGACY_STATUS_TO_MATURITY.values()) - SUPPORTED_MATURITIES
    assert not unknown, f"the mapping targets non-maturities: {sorted(unknown)}"


def test_deployed_is_not_silently_mapped():
    """The specific guard against the R2 violation, pinned as a rule rather than a comment."""
    assert "deployed" not in LEGACY_STATUS_TO_MATURITY, (
        "'deployed' acquired an automatic mapping. views-models ADR-017 makes "
        "deployed->graduate conditional on every member of a graduate ensemble also being "
        "graduate, and the one deployed source in views-models has three shadow members."
    )
    assert "deployed" in LEGACY_STATUSES_WITHOUT_A_SAFE_MAPPING
