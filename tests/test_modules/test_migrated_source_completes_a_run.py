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

So these drive `handle_single_log_creation` — the function a real run calls — and let it
write a real log to a real directory, then read that file back through
`validate_ensemble_model_deployment_status`, the function the next ensemble run uses.

(`handle_ensemble_log_creation` is deliberately NOT claimed here. It has zero execution
coverage anywhere in this repo — an earlier draft of this docstring said these tests drove
it, which was false, and is the C-304 shape appearing inside a file written to stop it.) The
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

import logging

import pytest

from views_pipeline_core.files.utils import (
    handle_single_log_creation,
    read_log_file,
)
from views_pipeline_core.modules.validation.ensemble.check import (
    validate_ensemble_model_deployment_status,
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
        path_manager, _config(maturity="graduate"), train=True
    )

    log = read_log_file(path_manager.data_generated / "calibration_log.txt")

    for key in ("Single Model Name", "Deployment Status", "Data Generation Timestamp"):
        assert key in log, f"{key} missing — the ensemble reader subscripts it"

    # And then actually call the consumer, because the three assertions above are about
    # a dict and the test's name is about a function. Caught by mutation: making
    # `validate_ensemble_model_deployment_status` reject new-vocabulary values left this
    # file entirely green — the round trip was asserted, never performed.
    assert validate_ensemble_model_deployment_status(
        path_manager.data_generated, "calibration", "graduate"
    ) is True, (
        "a graduate ensemble reading a graduate member's freshly-written log must "
        "validate — this is the whole point of writing the declared vocabulary rather "
        "than translating on the way out"
    )


def test_the_round_trip_can_actually_fail(tmp_path):
    """The control for the assertion above.

    A round trip that returns True for everything proves nothing. ADR-058's R2 says a
    graduate ensemble's members must all be graduate, so the same path with a `candidate`
    member must come back False — and it must do so having actually read the file, which
    is what distinguishes this from asserting the rule against a hand-built dict.
    """
    path_manager = _PathManager(tmp_path)
    handle_single_log_creation(path_manager, _config(maturity="candidate"), train=True)

    assert validate_ensemble_model_deployment_status(
        path_manager.data_generated, "calibration", "graduate"
    ) is False, "a candidate member must not validate into a graduate ensemble (R2)"


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


def test_the_ensemble_site_raises_rather_than_passing_None_to_the_member_check(monkeypatch):
    """The neither-key case at the SECOND site, which the parametrisation above misses.

    Found by mutation: replacing `config_maturity(config)` in `check.py` with
    `config.get("maturity", config.get("deployment_status"))` — the exact `.get()`-or form
    the accessor's docstring argues against — passed the whole suite. Both failures the
    change's mutation table credits to that argument came from the `create_log_file` path,
    so the rule was pinned at one of its two sites and asserted at the other.

    What the `or` form would do here: hand `None` to `ensemble_may_contain_member`, where
    `normalise_maturity(None)` returns `None` meaning *indeterminate* — and an
    indeterminate ensemble maturity compared against a real member maturity is a rule
    evaluated on a value nobody supplied.
    """
    config = {"name": "no_vocabulary", "models": ["member_a"], "run_type": "calibration"}

    with pytest.raises(KeyError, match="config_maturity.py"):
        _drive_validate_ensemble_model(monkeypatch, config, member_status="candidate")


# ----------------------------------------------------------------------------------
# The third site — not named in #496, found tracing every reader of the field
# ----------------------------------------------------------------------------------


def _build_ensemble_context(config):
    """Drive the real `_build_context`, the way the characterization tests do.

    Reuses `test_ensemble_context_characterization`'s helper rather than calling
    `from_config` directly: that helper bypasses `__init__` (which pulls in wandb,
    logging and four ADR-045 stages) while still running the real body, which is where
    the line under test lives.
    """
    from tests.test_managers.test_ensemble_context_characterization import _build
    from views_pipeline_core.managers.ensemble.dataframe_ensemble import (
        DataFrameEnsembleManager,
    )

    return _build(DataFrameEnsembleManager, config)


def _ensemble_config(**keys):
    base = {
        "name": "test_ensemble",
        "models": ["purple_alien"],
        "aggregation": "mean",
        "regression_targets": ["lr_sb_best"],
    }
    base.update(keys)
    return base


def test_a_migrated_ensembles_context_carries_its_declared_maturity():
    """`EnsembleContext.from_config` read the legacy key alone, with a silent default.

    Unlike the two sites #496 names, this one did not crash on a migrated config — it
    returned `"shadow"`, the default, for an ensemble declaring `maturity: graduate`. A
    wrong value rather than a missing one, and `shadow` normalises to `candidate`, so an
    ensemble that had declared itself graduate would have been treated as a candidate.

    Nothing reads this field today, which is why it was a trap rather than a live defect:
    the first consumer added would have inherited the wrong value with no way to tell.
    """
    ctx = _build_ensemble_context(_ensemble_config(maturity="graduate"))

    assert ctx.deployment_status == "graduate", (
        "the ensemble declared graduate; the context must not silently say shadow"
    )


def test_an_ensemble_declaring_neither_key_still_gets_the_default():
    """The behaviour that predates #496 and is deliberately preserved.

    This site has always defaulted, and removing that would be a behaviour change riding
    on a bugfix — the thing `member_maturity`'s docstring warns against. The default is
    now passed explicitly, so a caller's tolerance for a missing field is visible at the
    call rather than implied by which accessor it reached for.
    """
    ctx = _build_ensemble_context(_ensemble_config())

    # The literal, not `DEFAULT_DEPLOYMENT_STATUS` imported from the module under test.
    # Comparing the output to the constant that produced it is true for every value the
    # constant could hold — swept through six and it passed on all of them, which is a
    # test that cannot fail.
    assert ctx.deployment_status == "shadow"


def test_the_default_is_opt_in_not_the_accessors_behaviour():
    """The distinction the `default` argument exists to keep.

    If `config_maturity` defaulted by itself, the two loud sites would have gone quiet
    when this one was fixed — one accessor silently changing three call sites' failure
    modes at once.
    """
    with pytest.raises(KeyError):
        config_maturity({"name": "m"})

    assert config_maturity({"name": "m"}, default="shadow") == "shadow"


# ----------------------------------------------------------------------------------
# The accessor's own contract — the parts a call site can get wrong
# ----------------------------------------------------------------------------------


def test_a_declared_but_empty_maturity_is_refused_not_passed_through():
    """The one value that cannot survive the round trip this accessor feeds.

    Reproduced end to end before this guard existed: `config_maturity({"maturity": ""})`
    returned `""`, `create_log_file` wrote the line `Deployment Status: `, and
    `read_log_file` on that file raised `ValueError: not enough values to unpack` —
    taking down the WHOLE log, not that field. `validate_ensemble_model_deployment_status`
    catches the ValueError and returns `False`, so the next ensemble run reports the
    *member* as failing validation. A blank maturity surfaces as a wrong accusation about
    a different model.

    Note what is NOT refused: `maturity: "shadwo"`. Vocabulary validity is the sniffer's
    job, and restating the value rules here is exactly the drift this accessor exists to
    prevent. What is refused is the value that corrupts the artifact.
    """
    with pytest.raises(ValueError, match="empty value"):
        config_maturity({"name": "m", "maturity": "", "deployment_status": "shadow"})


def test_the_empty_value_is_refused_before_it_reaches_the_log(tmp_path):
    """The chain above, at the entry point rather than at the accessor.

    A unit test of the accessor cannot see whether the caller reaches it — which is the
    C-305 lesson this whole file exists to apply.
    """
    with pytest.raises(ValueError, match="empty value"):
        handle_single_log_creation(
            _PathManager(tmp_path), _config(maturity="  "), train=True
        )


def test_presence_is_tested_the_same_way_the_sniffer_tests_it():
    """`is not None`, not truthiness — so a declared key is never skipped silently.

    Under truthiness, `maturity: ""` would fall through to the legacy key and this
    accessor would answer `"shadow"`: a plausible legacy value, produced for a config the
    file's own guard rejects. It now raises instead, which is the point — a declared key
    is *answered for*, never stepped over.
    """
    with pytest.raises(ValueError):
        config_maturity({"name": "m", "maturity": "", "deployment_status": "shadow"})

    assert config_maturity({"deployment_status": "shadow"}) == "shadow", (
        "an absent maturity, as opposed to an empty one, still falls through"
    )


def test_the_default_cannot_be_passed_positionally():
    """`default` is keyword-only, and that is the enforcement of "visible at the call".

    The docstring argues that a caller's tolerance for a missing field should be readable
    at the call site. `config_maturity(cfg, "shadow")` reads as neither, so the signature
    refuses it. Without the `*` the argument is a comment rather than a rule — and the
    other three tests all pass `default=` by keyword, so none of them can see it go.
    """
    with pytest.raises(TypeError):
        config_maturity({"name": "m"}, "shadow")  # type: ignore[misc]


def test_the_sentinel_is_a_type_not_a_value():
    """"No default given" and "a default of None" must not be the same thing.

    If the sentinel were `None`, `default=None` would silently mean "raise" instead of
    "return None" — and `None` is exactly the value this accessor exists never to hand
    back, because `create_log_file` would write `Deployment Status: None` to the run log.
    Typing the sentinel makes that distinction real rather than asserted in a comment.
    """
    from views_pipeline_core.modules.validation.core_config_sniffer import (
        _RAISE,
        _RaiseIfAbsent,
    )

    assert isinstance(_RAISE, _RaiseIfAbsent)
    assert _RAISE is not None, "a None sentinel would conflate unset with a None default"


# ----------------------------------------------------------------------------------
# Sites four and five — upstream of everything above, and they made it unreachable
# ----------------------------------------------------------------------------------


def _scripts(tmp_path, filename, function, value):
    """Write a real config script and return the `get_scripts()`-shaped map for it."""
    script = tmp_path / filename
    script.write_text(f"def {function}():\n    return {{'maturity': '{value}'}}\n"
                      if function == "get_maturity_config"
                      else f"def {function}():\n    return {{'deployment_status': '{value}'}}\n")
    return {filename: script}


def test_a_migrated_ensemble_can_load_its_maturity_config(tmp_path):
    """The defect that made the two sites #496 names unreachable.

    Both ensemble managers asked for `("config_deployment.py", "get_deployment_config")`
    by hand, while `ModelManager` resolved either name. So an ensemble that had completed
    the rename — precisely what ADR-057 asks for — loaded `None`, its combined config
    declared neither vocabulary, and `CoreConfigSniffer.sniff_all` refused the run on the
    first statement of `execute_single_run`. The log-writing and member-check sites were
    never reached, so fixing them alone would not have made a migrated ensemble runnable.
    """
    from views_pipeline_core.managers.configuration.script_config import (
        load_maturity_config,
    )

    scripts = _scripts(tmp_path, "config_maturity.py", "get_maturity_config", "graduate")

    assert load_maturity_config(scripts, "test_ensemble") == {"maturity": "graduate"}


def test_a_legacy_ensemble_still_loads_through_the_old_pair(tmp_path):
    """The other half of the window: the rename must not be required yet."""
    from views_pipeline_core.managers.configuration.script_config import (
        load_maturity_config,
    )

    scripts = _scripts(tmp_path, "config_deployment.py", "get_deployment_config", "shadow")

    assert load_maturity_config(scripts, "test_ensemble") == {"deployment_status": "shadow"}


@pytest.fixture
def resolver_warnings():
    """Collect WARNING records from the resolver's own logger.

    Deliberately not `caplog`, and this file learned it the hard way: written with
    `caplog` this test passed alone and captured NOTHING in the full suite, because
    `LoggingModule` sets `propagate = False` and `disabled = True` on this package's
    loggers, so what `caplog` can see depends on which test ran first. The same fixture
    and the same reason already exist in `test_maturity_vocabulary_transition.py`.
    """
    from views_pipeline_core.managers.configuration import script_config

    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector(level=logging.WARNING)
    logger = script_config.logger
    previous_level, previous_disabled = logger.level, logger.disabled
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.disabled = False  # a disabled logger drops records before any handler sees them
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.disabled = previous_disabled


def test_the_new_filename_wins_and_says_so(tmp_path, resolver_warnings):
    """Both files present is a normal half-finished rename — preferred, not refused.

    It warns, because a file that is being silently ignored is how the wrong config gets
    edited for a week.
    """
    from views_pipeline_core.managers.configuration.script_config import (
        load_maturity_config,
    )

    scripts = {
        **_scripts(tmp_path, "config_maturity.py", "get_maturity_config", "graduate"),
        **_scripts(tmp_path, "config_deployment.py", "get_deployment_config", "shadow"),
    }

    assert load_maturity_config(scripts, "test_ensemble") == {"maturity": "graduate"}

    text = " ".join(r.getMessage() for r in resolver_warnings)
    assert "config_deployment.py" in text and "ADR-057" in text, (
        f"the ignored file must be named, and the rule cited. Got: {text!r}"
    )


def test_one_file_alone_warns_about_nothing(tmp_path, resolver_warnings):
    """The control: proves the fixture above can actually observe silence.

    Without it, a warning assertion that never fires and a capture that never works look
    identical — which is precisely how the `caplog` version of the test above went green
    while seeing nothing.
    """
    from views_pipeline_core.managers.configuration.script_config import (
        load_maturity_config,
    )

    load_maturity_config(
        _scripts(tmp_path, "config_maturity.py", "get_maturity_config", "graduate"),
        "test_ensemble",
    )

    assert resolver_warnings == []


def test_no_caller_resolves_the_maturity_config_filename_by_hand():
    """Derived, not hand-listed — the rule this repo has been wrong about six times.

    The defect was not that `config_deployment.py` appeared somewhere; it was that a
    *call site* passed the legacy filename and the legacy entry point as arguments,
    instead of asking for "the maturity config". So this walks the AST for those two
    strings as VALUES — prose mentioning them, including the comment left at each site
    saying what it used to do, is not the defect and must not be flagged. A guard that
    fires on its own explanation gets allowlisted into uselessness.

    One place in the package may name them, and the window then closes in one edit
    rather than in however many sites a grep happens to find.

    Templates are excluded: they generate a model's own files and are a separate
    (registered) problem — pipeline-core still scaffolds new sources onto the legacy
    vocabulary, which is why ADR-057's close condition is not reachable by migration
    alone.
    """
    import ast
    import pathlib

    package = pathlib.Path(__file__).resolve().parents[2] / "views_pipeline_core"
    # The entry point, not the filename. `config_deployment.py` legitimately appears as
    # a value twice more — `LEGACY_MATURITY_CONFIG_FILENAME` in the sniffer, and
    # `ModelPathManager`'s resolution of the legacy path — and both are the transition
    # window working as designed. What no call site may do is name the legacy *loader*,
    # because that is the half that cannot be resolved from a filename.
    legacy = {"get_deployment_config"}

    offenders = sorted(
        str(f.relative_to(package))
        for f in package.rglob("*.py")
        if "templates" not in f.parts
        and any(
            isinstance(node, ast.Constant) and node.value in legacy
            for node in ast.walk(ast.parse(f.read_text()))
        )
    )

    assert offenders == ["managers/configuration/script_config.py"], (
        "the legacy entry point must appear as a value in exactly one "
        f"place — the resolver. Found: {offenders}"
    )
