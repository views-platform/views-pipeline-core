"""A report-enabled `dataframe` run must say so out loud. Issue #211, register C-191/D-36.

## What is deprecated, and what is not

`prediction_format` chooses more than a data shape. `stage.py` forks on it: the
`prediction_frame` branch is dense and bounded; the `dataframe` branch hands list-in-cell
values to views-reporting to densify, which is the #181 out-of-memory failure. So a
config field that reads like a format preference silently selects a memory-safety
posture — the thing C-191 objects to.

Only the **report-enabled** case is deprecated. A `dataframe` model that never generates a
report is not on the OOM path and is not being asked to move. That distinction is why the
check lives in the report stage and not in `CoreConfigSniffer`: the sniffer reads a config
dict and cannot see the `-re` flag, so it cannot tell the two apart. Reaching
`generate_forecast_report` proves report-enablement.

## Warn, not reject — deliberately

D-36 settled the direction: reject `dataframe` reports loudly rather than silently take
the OOM path. #211 makes the reject conditional on report-bearing models having somewhere
to go, and which of views-models' `dataframe` configs are report-enabled has never been
audited. Rejecting first would break runs with no alternative.

So these tests pin the **warn** state, and one of them pins the fact that it is *not* yet
a rejection — so that flipping the reject on is a visible, deliberate test change rather
than something that quietly starts happening.
"""

from __future__ import annotations

import logging

import pytest

from views_pipeline_core.managers.reporting.stage import (
    _warn_if_report_format_is_deprecated,
)
from views_pipeline_core.modules.validation.core_config_sniffer import (
    DEPRECATED_REPORT_PREDICTION_FORMATS,
    SUPPORTED_PREDICTION_FORMATS,
)

_DEPRECATION_MARKER = "DEPRECATED"


def test_a_report_enabled_dataframe_run_warns(caplog):
    """#211's first acceptance criterion."""
    with caplog.at_level(logging.WARNING):
        _warn_if_report_format_is_deprecated("dataframe", "rusty_bucket")

    assert caplog.records, "a deprecated report format produced no warning at all"
    message = caplog.records[-1].getMessage()
    assert _DEPRECATION_MARKER in message
    assert "rusty_bucket" in message, (
        "the warning does not name the model. In a run that reports for several models, "
        "an unattributed deprecation notice cannot be acted on."
    )
    assert "prediction_frame" in message, (
        "the warning does not name the remediation. A deprecation that does not say what "
        "to do instead is an announcement, not a migration instruction."
    )


def test_the_supported_dense_format_is_silent(caplog):
    """The negative control. A warning that always fires teaches people to ignore it."""
    with caplog.at_level(logging.WARNING):
        _warn_if_report_format_is_deprecated("prediction_frame", "rusty_bucket")

    assert not [
        record
        for record in caplog.records
        if _DEPRECATION_MARKER in record.getMessage()
    ], "the supported dense report path was reported as deprecated"


def test_it_warns_rather_than_raising():
    """Pins the warn/reject boundary so the flip cannot happen by accident.

    When the views-models audit lands and D-36's reject half is turned on, this test must
    be changed deliberately — which is the point. #211 requires the floor decision to be
    recorded before the hard error ships, and a test that silently starts passing would
    let that requirement be skipped.
    """
    for prediction_format in sorted(DEPRECATED_REPORT_PREDICTION_FORMATS):
        _warn_if_report_format_is_deprecated(prediction_format, "rusty_bucket")


def test_every_deprecated_format_is_a_real_format():
    """A deprecation set that names a non-existent format matches nothing and warns never.

    Enforced at import by an assert in `core_config_sniffer`; asserted here too so the
    failure is a named test rather than a collection error nobody reads.
    """
    unknown = DEPRECATED_REPORT_PREDICTION_FORMATS - SUPPORTED_PREDICTION_FORMATS
    assert not unknown, (
        f"{sorted(unknown)} is deprecated for reports but is not a supported "
        f"prediction format, so the deprecation can never fire."
    )


def test_the_deprecation_does_not_swallow_the_whole_vocabulary():
    """If every format were deprecated there would be nothing to migrate to."""
    remaining = SUPPORTED_PREDICTION_FORMATS - DEPRECATED_REPORT_PREDICTION_FORMATS
    assert remaining, (
        "every supported prediction format is deprecated for reports, so the warning's "
        "remediation points nowhere."
    )


@pytest.mark.parametrize("unknown_format", ["", "DataFrame", "prediction-frame", "None"])
def test_an_unrecognised_format_is_not_reported_as_deprecated(unknown_format, caplog):
    """Deprecation and invalidity are different findings with different remedies.

    An unsupported format is `CoreConfigSniffer`'s to reject; saying "deprecated" about it
    would send the reader to a migration guide for a value that was never valid.
    """
    with caplog.at_level(logging.WARNING):
        _warn_if_report_format_is_deprecated(unknown_format, "rusty_bucket")

    assert not [
        record
        for record in caplog.records
        if _DEPRECATION_MARKER in record.getMessage()
    ], f"{unknown_format!r} is not a supported format, so it cannot be a deprecated one"
