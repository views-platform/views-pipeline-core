"""The audit CLI's exit codes mean things. Nothing tested them. C-292.

Run: `conda run -n views_pipeline pytest tests/test_modules/test_appwrite_audit_cli.py -q`

## Why this file exists

An independent mutation audit found `views_pipeline_core/modules/appwrite/audit/__main__.py`
entirely unguarded — no test in the repo imported it. Two mutations survived the **full**
2683-test suite:

- the configuration-fault wrapper returning **1** instead of 2. 1 is this CLI's code for a
  substantive finding: a broken file/document pairing, or a container open to anyone. A
  half-loaded environment would have reported itself as an exposure.
- the `--permissions` branch returning **0** unconditionally, while still printing
  `VERDICT: OPEN`. A scheduled run, or anything reading the status rather than the text,
  would record an open partner container as clean.

The second is the one that matters. `tools/wipe_fao_shelf.py` has 24 exit-code assertions
for the same class of reason (C-271, C-244); this CLI had none, and it is the instrument an
operator runs to answer a security question.

## What these tests do NOT cover

The real substrate. They patch `build_file_manager` and `read_permissions`, so they pin the
**dispatch and the exit contract**, not the probe — that is
`test_appwrite_permissions_probe.py`'s job. Stated so nobody reads green here as evidence
that the probe is right.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import views_pipeline_core.modules.appwrite.audit.__main__ as cli
from views_pipeline_core.modules.appwrite.audit.permissions import (
    ContainerPermissions,
    PermissionsReport,
)


def _report(*, containers=(), indeterminate=()):
    return PermissionsReport(
        containers=list(containers), indeterminate=list(indeterminate)
    )


OPEN = ContainerPermissions(
    kind="collection", container_id="crafd", grants=['read("any")'], security_flag=False
)


def test_a_configuration_fault_exits_two_not_one(capsys):
    """1 means a finding. A missing environment variable is not a finding."""
    with patch.object(cli, "build_file_manager", side_effect=RuntimeError("no env")):
        assert cli.main(["--permissions"]) == 2
    assert "COULD NOT START" in capsys.readouterr().err


def test_an_open_container_exits_one():
    """The mutation that survived the whole suite: `return 0` here, while the printed
    report says VERDICT: OPEN."""
    with patch.object(cli, "build_file_manager", return_value=MagicMock()), patch.object(
        cli, "read_permissions", return_value=_report(containers=[OPEN])
    ):
        assert cli.main(["--permissions"]) == 1


def test_a_clean_shelf_exits_zero():
    """The control. If everything exited 1 the test above would pass for free."""
    clean = ContainerPermissions(
        kind="collection", container_id="crafd", grants=[], security_flag=False
    )
    with patch.object(cli, "build_file_manager", return_value=MagicMock()), patch.object(
        cli, "read_permissions", return_value=_report(containers=[clean])
    ):
        assert cli.main(["--permissions"]) == 0


def test_an_undetermined_read_exits_two_even_with_nothing_open():
    with patch.object(cli, "build_file_manager", return_value=MagicMock()), patch.object(
        cli, "read_permissions", return_value=_report(indeterminate=["could not read"])
    ):
        assert cli.main(["--permissions"]) == 2


def test_the_report_is_printed_not_only_returned(capsys):
    """The operator reads stdout. A verdict computed and never shown is not a verdict."""
    with patch.object(cli, "build_file_manager", return_value=MagicMock()), patch.object(
        cli, "read_permissions", return_value=_report(containers=[OPEN])
    ):
        cli.main(["--permissions"])
    assert "VERDICT: OPEN" in capsys.readouterr().out


def test_without_the_flag_the_pairing_audit_runs_instead():
    """Two modes, one CLI. `--permissions` must not leak into the default path, and the
    default path must not silently become the permissions one."""
    with patch.object(cli, "build_file_manager", return_value=MagicMock()), patch.object(
        cli, "read_permissions"
    ) as probe, patch.object(cli, "audit") as pairing, patch.object(
        cli, "exit_code", return_value=0
    ):
        pairing.return_value.render.return_value = "pairing report"
        assert cli.main([]) == 0
    probe.assert_not_called()
    pairing.assert_called_once()


@pytest.mark.parametrize("argv", [["--collection", "c"], ["--bucket", "b"]])
def test_half_a_coordinate_pair_is_a_configuration_fault_not_a_finding(argv, capsys):
    """`build_file_manager` refuses a half pair. Uncaught that exited 1, which this
    CLI's own comment defines as a container being open (C-271)."""
    assert cli.main(["--permissions", *argv]) == 2
    assert "COULD NOT START" in capsys.readouterr().err
