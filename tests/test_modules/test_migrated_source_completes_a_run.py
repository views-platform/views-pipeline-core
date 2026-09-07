"""A source that finished the ADR-057 migration can complete a run. #496, C-305.

Run: `conda run -n views_pipeline pytest tests/test_modules/test_migrated_source_completes_a_run.py -q`

## What this stops

PR #495 removed `deployment_status` from `MANDATORY_KEYS_UNIVERSAL`, so a migrated
config stopped failing the sniffer. It still crashed — later, and further from its cause:

    files/utils.py:134            create_log_file()        KeyError: 'deployment_status'
    ensemble/check.py:316         validate_ensemble_model()

Both subscripted the legacy key directly. A source that had completed the migration to
`config_maturity.py` has no such key.

## Why these tests go through the entry points and not the functions

`test_maturity_vocabulary_transition.py` and `test_ensemble_check.py` call the checks
directly, which is why 17 passing tests could not see C-305: **a composition defect is
invisible to a test that never composes.** The sniffer was correct and the caller was
wrong, and every test called the sniffer.

So these drive `handle_single_log_creation` and `handle_ensemble_log_creation` — the
functions a real run calls — and let them write a real log to a real directory. The
assertion is on the file that lands on disk, because that file is the interface: it is
read back by `validate_ensemble_model_deployment_status` on the next ensemble run.

## The vocabulary the log records

Whatever the config declares. A migrated source writes `candidate` where it used to write
`shadow`, and that is safe for one reason established before this change: the only reader
of the value normalises it (`member_maturity.py:85-86` calls `normalise_maturity` on both
sides). Traced 2026-09-07: **nothing outside pipeline-core reads the `Deployment Status`
line** — the readers are `check.py:150` and the member-copy loop in `files/utils.py`, and
every other occurrence platform-wide is a README table or a config docstring.
"""

from __future__ import annotations

import pytest

from views_pipeline_core.files.utils import (
    handle_single_log_creation,
    read_log_file,
)
from views_pipeline_core.modules.validation.core_config_sniffer import config_maturity


class _PathManager:
    """Enough of `ModelPathManager` for the log-writing path, which uses two attributes."""

    def __init__(self, root):
        self.data_generated = root / "generated"
        self.data_raw = root / "raw"
        self.data_generated.mkdir(parents=True, exist_ok=True)
        self.data_raw.mkdir(parents=True, exist_ok=True)


def _config(name="migrated_model", **keys):
    base = {"run_type": "calibration", "name": name, "timestamp": "20260907_120000"}
    base.update(keys)
    return base


# ----------------------------------------------------------------------------------
# The entry point a real run calls
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "declared,expected",
    [
        ({"maturity": "candidate"}, "candidate"),
        ({"maturity": "graduate"}, "graduate"),
        ({"maturity": "retired"}, "retired"),
        ({"deployment_status": "shadow"}, "shadow"),
        ({"deployment_status": "deployed"}, "deployed"),
        # Both present is a normal intermediate state during the window, and the new key
        # wins — the same precedence `_check_deployment_status` and
        # `ModelManager.__load_maturity_config` already apply.
        ({"maturity": "candidate", "deployment_status": "shadow"}, "candidate"),
    ],
)
def test_a_run_writes_a_log_whatever_vocabulary_the_config_declares(
    tmp_path, declared, expected
):
    """The regression. Before #496 the first three rows raised KeyError here."""
    path_manager = _PathManager(tmp_path)

    handle_single_log_creation(path_manager, _config(**declared), train=True)

    log = read_log_file(path_manager.data_generated / "calibration_log.txt")
    assert log["Deployment Status"] == expected, (
        f"a config declaring {declared} should log {expected!r}; the log says "
        f"{log.get('Deployment Status')!r}"
    )


def test_a_config_declaring_neither_key_fails_loudly_at_the_entry_point(tmp_path):
    """The control, and the reason this is an accessor rather than a `.get() or .get()`.

    `config.get("maturity") or config.get("deployment_status")` yields `None` here, and
    the run would write `Deployment Status: None` into the log — a value
    `normalise_maturity` documents as *indeterminate, not benign*, persisted to disk and
    read back by the next ensemble run. Today it raises, and it must keep raising.
    """
    path_manager = _PathManager(tmp_path)

    with pytest.raises(KeyError, match="neither 'maturity' nor the legacy"):
        handle_single_log_creation(path_manager, _config(), train=True)

    assert not (path_manager.data_generated / "calibration_log.txt").exists(), (
        "nothing should have been written"
    )


def test_the_log_is_readable_back_by_the_function_that_consumes_it(tmp_path):
    """The round trip, because the log is an interface and not just an artefact.

    `validate_ensemble_model_deployment_status` reads this key off disk on the next
    ensemble run. A log a migrated source can write but nothing can read would move the
    break rather than fix it.
    """
    path_manager = _PathManager(tmp_path)
    handle_single_log_creation(
        path_manager, _config(maturity="candidate"), train=True
    )

    log = read_log_file(path_manager.data_generated / "calibration_log.txt")

    for key in ("Single Model Name", "Deployment Status", "Data Generation Timestamp"):
        assert key in log, f"{key} missing — the ensemble reader subscripts it"


# ----------------------------------------------------------------------------------
# The accessor's own contract
# ----------------------------------------------------------------------------------


def test_the_accessor_does_not_translate_the_value():
    """It returns what was declared. Translation is `normalise_maturity`'s job and is
    applied downstream to both sides of a comparison; doing it here as well would
    translate twice and lose the distinction between what a config says and what it
    means."""
    assert config_maturity({"name": "m", "deployment_status": "shadow"}) == "shadow"
    assert config_maturity({"name": "m", "maturity": "candidate"}) == "candidate"


def test_the_accessor_names_the_file_to_edit():
    """A remediation pointing at the wrong file is worse than none, because it is
    followed — the reason `_check_deployment_status`'s messages were rewritten."""
    with pytest.raises(KeyError, match="config_maturity.py"):
        config_maturity({"name": "unmigrated"})


# ----------------------------------------------------------------------------------
# The ensemble entry point — the second site that subscripted the legacy key
# ----------------------------------------------------------------------------------


def _drive_validate_ensemble_model(monkeypatch, ensemble_config, member_status):
    """Run the real `validate_ensemble_model` body with its collaborators stubbed.

    Only the heavy dependencies are replaced — the path manager, the two managers, and
    the two sibling checks. The function's own body still runs, which is the point: the
    line under test is `config_maturity(config)` inside it, and a test that stubbed the
    function itself would prove nothing. Same reason the tests above go through
    `handle_single_log_creation` rather than `create_log_file`.
    """
    from views_pipeline_core.modules.validation.ensemble import check as check_module

    # Import before patching. `check.py` imports these inside the function, and
    # `managers.ensemble.ensemble` subclasses `ModelPathManager` at module scope — so a
    # patch applied before that module loads makes it try to subclass a lambda.
    import views_pipeline_core.data.model_path  # noqa: F401
    import views_pipeline_core.managers.ensemble  # noqa: F401
    import views_pipeline_core.managers.model  # noqa: F401

    monkeypatch.setattr(check_module, "validate_output_scale_consistency", lambda _: None)
    monkeypatch.setattr(check_module, "validate_model_conditions", lambda *a, **k: True)
    monkeypatch.setattr(check_module, "validate_partition_config", lambda *a, **k: True)

    seen = {}

    def _capture(path_generated, run_type, ensemble_status):
        seen["ensemble_status"] = ensemble_status
        return ensemble_status is not None and member_status is not None

    monkeypatch.setattr(
        check_module, "validate_ensemble_model_deployment_status", _capture
    )
    monkeypatch.setattr(
        "views_pipeline_core.data.model_path.ModelPathManager",
        lambda name: type("_P", (), {"data_generated": "/tmp/nowhere"})(),
    )
    monkeypatch.setattr(
        "views_pipeline_core.managers.model.ModelManager", lambda p: object()
    )
    monkeypatch.setattr(
        "views_pipeline_core.managers.ensemble.EnsembleManager", lambda p: object()
    )
    monkeypatch.setattr(
        "views_pipeline_core.managers.ensemble.EnsemblePathManager", lambda n: object()
    )

    check_module.validate_ensemble_model(ensemble_config)
    return seen


@pytest.mark.parametrize(
    "declared,expected",
    [
        ({"maturity": "graduate"}, "graduate"),
        ({"deployment_status": "deployed"}, "deployed"),
        ({"maturity": "graduate", "deployment_status": "deployed"}, "graduate"),
    ],
)
def test_a_migrated_ensemble_reaches_the_member_check(monkeypatch, declared, expected):
    """Before #496 the first row raised KeyError at `config["deployment_status"]`,
    before any member was examined."""
    config = {
        "name": "migrated_ensemble",
        "models": ["member_a"],
        "run_type": "calibration",
        **declared,
    }

    seen = _drive_validate_ensemble_model(monkeypatch, config, member_status="candidate")

    assert seen["ensemble_status"] == expected, (
        "the ensemble's own declared maturity must reach the member comparison "
        "unchanged — `normalise_maturity` is applied inside it, on both sides"
    )
