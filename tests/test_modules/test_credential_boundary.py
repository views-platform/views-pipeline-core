"""S6 (#346) — the credential boundary: no ambient .env, a preflight everywhere, a frozen config.

Three defects that survived þing-01, and the first of them is the mechanism the whole
platform þing formed around.

**C-177** — `modules/datastore/datastore.py` called `dotenv.load_dotenv(dotenv.find_dotenv())`
at MODULE SCOPE. Importing anything that transitively reached it walked the filesystem
looking for a `.env` and mutated `os.environ` as an import side effect. PLATFORM-001 §3
clause 3 forbids this outright: *"a library reading whatever `.env` the working directory
holds is the disease this contract exists to cure."* A service entry point reading its own
process environment is legitimate; a library reading the working directory's is not.

**C-11** — the fail-loud preflight covered exactly one path.

**C-240** — `AppwriteConfig` validates at construction but is a plain `@dataclass`, so
post-construction mutation is unchecked — while its sibling `PredictionStoreConfig` is
`frozen=True`. Two configs for one seam, two different guarantees.
"""

import dataclasses
import subprocess
import sys

import pytest

from views_pipeline_core.exceptions.exceptions import ConfigurationException
from views_pipeline_core.modules.appwrite.file import AppwriteConfig, AuthMethod


# ---------------------------------------------------------------------------
# C-177 — importing a library must not read the filesystem or mutate the process
# ---------------------------------------------------------------------------

_ENV_SIDE_EFFECT_PROBE = """
import os, sys
before = dict(os.environ)
import views_pipeline_core.modules.datastore.datastore  # noqa: F401
after = dict(os.environ)
added = {k: v for k, v in after.items() if k not in before}
changed = {k for k in before if before[k] != after.get(k)}
assert not added and not changed, (
    "importing datastore mutated os.environ: added=%s changed=%s" % (sorted(added), sorted(changed))
)
"""


def _probe(source: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)


def test_importing_datastore_does_not_mutate_the_environment(tmp_path):
    """The C-177 regression, run from a directory that CONTAINS a .env.

    Running the probe in a clean directory would pass even with `load_dotenv` present —
    there would be nothing to find. The `.env` below is what makes the test able to fail
    for the reason it exists, rather than for the reason the environment happens to
    allow (C-218's structural question, applied to a filesystem probe).
    """
    (tmp_path / ".env").write_text("VIEWS_PIPELINE_CORE_S6_CANARY=leaked\n")

    result = subprocess.run(
        [sys.executable, "-c", _ENV_SIDE_EFFECT_PROBE],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_the_canary_probe_would_actually_catch_a_dotenv_load(tmp_path):
    """Proves the probe above is not vacuous.

    A test that passes because nothing could have gone wrong is worth nothing. This
    deliberately loads the `.env` and asserts the probe's own detection logic fires.
    """
    (tmp_path / ".env").write_text("VIEWS_PIPELINE_CORE_S6_CANARY=leaked\n")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, dotenv\n"
            "before = dict(os.environ)\n"
            "dotenv.load_dotenv(dotenv.find_dotenv())\n"
            "added = {k for k in os.environ if k not in before}\n"
            "assert 'VIEWS_PIPELINE_CORE_S6_CANARY' in added, added\n",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, (
        "the canary itself did not leak, so the test above proves nothing: "
        + result.stdout
        + result.stderr
    )


def test_no_importable_module_searches_for_a_dotenv_file():
    """Bans the BEHAVIOUR — a search rooted at the working directory — not one spelling.

    The first version of this check looked only for `find_dotenv`, which is the form
    `datastore.py` happened to use. But `load_dotenv()` with **no arguments** performs the
    same filesystem walk, so that version banned the instance rather than the class.
    Caught by review; it is the third guard-scoping error in this epic (see C-259, C-261),
    and the recurring question is always the same: *where could this defect occur* versus
    *where does the check look*.

    An explicit `load_dotenv(dotenv_path=...)` or `load_dotenv(path)` is a different thing
    entirely — the caller named the file. `model_path.py:371` and `dataloaders.py:931`
    both do that and are deliberately untouched.
    """
    import ast
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent.parent
    offenders = []
    for path in sorted((repo / "views_pipeline_core").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            where = f"{path.relative_to(repo)}:{node.lineno}"
            if name == "find_dotenv":
                offenders.append(f"{where} find_dotenv()")
            elif name == "load_dotenv" and not node.args and not any(
                kw.arg in {"dotenv_path", "stream"} for kw in node.keywords
            ):
                # No path given, so python-dotenv searches upward from the CWD.
                offenders.append(f"{where} load_dotenv() with no path")

    assert not offenders, (
        "these calls search for a .env from the working directory and mutate os.environ "
        "on import — PLATFORM-001 §3 clause 3 forbids that in importable code, because a "
        f"library must not read whatever .env its caller happens to be standing in: {offenders}"
    )


# ---------------------------------------------------------------------------
# C-240 — one seam, one guarantee
# ---------------------------------------------------------------------------


def _config(**overrides):
    base = dict(
        endpoint="https://cloud.appwrite.io/v1",
        project_id="p",
        credentials="k",
        auth_method=AuthMethod.API_KEY,
        bucket_id="b",
        bucket_name="B",
        collection_id="c",
        collection_name="C",
        database_id="d",
        database_name="D",
    )
    base.update(overrides)
    return AppwriteConfig(**base)


class TestAppwriteConfigIsFrozen:
    def test_a_coordinate_cannot_be_reassigned_after_construction(self):
        """Validation at construction is worthless if the value can change afterwards.

        `__post_init__` refuses a missing coordinate — but nothing stopped
        `config.bucket_id = "production_forecasts"` a line later, which is precisely the
        move C-229 was about.
        """
        config = _config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.bucket_id = "production_forecasts"

    def test_credentials_cannot_be_swapped_after_construction(self):
        config = _config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.credentials = "a-different-key"

    def test_the_string_auth_method_still_coerces(self):
        """Freezing must not break `__post_init__`'s coercion.

        `reconcile/targets.py` passes `auth_method="api_key"` as a string, so the
        coercion has a live caller and freezing requires `object.__setattr__`.
        """
        assert _config(auth_method="api_key").auth_method is AuthMethod.API_KEY

    def test_construction_validation_still_fails_loud(self):
        with pytest.raises(ConfigurationException, match="bucket_id"):
            _config(bucket_id=None)

    def test_it_matches_its_siblings_guarantee(self):
        """C-240's actual complaint: two configs for one seam with different rules."""
        from views_pipeline_core.configs.prediction_store import PredictionStoreConfig

        assert dataclasses.fields(AppwriteConfig)  # sanity
        assert AppwriteConfig.__dataclass_params__.frozen, "AppwriteConfig is not frozen"
        assert PredictionStoreConfig.__dataclass_params__.frozen, "sibling changed"


# ---------------------------------------------------------------------------
# C-11 — a preflight on EVERY consumer path
#
# The issue (#323) described this as unfinished: "the PFE publisher, `AppwriteSaver` and
# the loaders reach Appwrite with no preflight". Checked before implementing anything —
# it is no longer true. All three manager sites already build their config through
# `PredictionStoreConfig.from_environment()`, and every remaining construction site
# validates its own environment and names the missing variables.
#
# So the deliverable here is a GUARD rather than a change. The property holds today and
# nothing was keeping it true: a fourth consumer added tomorrow could construct a
# datastore from ad-hoc environment reads and nothing would object. That is how C-11 got
# written in the first place.
# ---------------------------------------------------------------------------

_PREFLIGHT_MARKERS = (
    "PredictionStoreConfig.from_environment",  # the validated env -> config path
    "ConfigurationException",                   # a site that validates its own env
)


def test_every_construction_of_the_datastore_is_preflighted():
    """Every module that builds a datastore must also validate its environment.

    Not a proof — a module could reference the marker and not use it. It is a tripwire
    for the realistic failure: someone wires a new consumer straight to
    `DatastoreModule(AppwriteConfig(...))` from raw `os.getenv` calls, and no error names
    the variable they forgot.
    """
    import ast
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent.parent
    unpreflighted = []
    for path in sorted((repo / "views_pipeline_core").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text()
        builds = any(
            isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", "")) in {
                "DatastoreModule",
                "AppWriteFileModule",
            }
            for n in ast.walk(ast.parse(source))
        )
        if not builds:
            continue
        # `datastore.py` itself receives an already-validated config from its caller.
        if path.name == "datastore.py" and "class DatastoreModule" in source:
            continue
        if not any(marker in source for marker in _PREFLIGHT_MARKERS):
            unpreflighted.append(str(path.relative_to(repo)))

    assert not unpreflighted, (
        "these modules construct an Appwrite client without any environment preflight, "
        "so a missing variable surfaces as a substrate error rather than as its own "
        f"name (register C-11, PLATFORM-001 §7): {unpreflighted}"
    )


def test_the_preflight_names_the_variable_it_is_missing(monkeypatch):
    """Fail-loud is only useful if it says WHICH variable."""
    from views_pipeline_core.configs.prediction_store import PredictionStoreConfig

    for var in list(dict(__import__("os").environ)):
        if var.startswith("APPWRITE_"):
            monkeypatch.delenv(var, raising=False)

    with pytest.raises(ConfigurationException) as excinfo:
        PredictionStoreConfig.from_environment()

    assert "APPWRITE_" in str(excinfo.value), (
        f"the preflight failed without naming a variable: {excinfo.value}"
    )


# ---------------------------------------------------------------------------
# Import-time side effects, as a CLASS rather than as the one instance a þing
# formed around. Found by the S0–S6 sweep.
# ---------------------------------------------------------------------------


def test_no_module_forces_a_log_level_at_import():
    """A library must not override the application's logging configuration.

    `modules/datastore/datastore.py` did `logger.setLevel(logging.INFO)` at module
    scope, so an operator who configured WARNING globally still got INFO from that
    module and never asked for it. Same class as the ambient `.env` load #346 removed —
    an import-time side effect the importer did not request — and it survived because
    that story was scoped to the dotenv line specifically.

    `modules/logging/` is exempt: deciding levels is its job, and it does so inside a
    method rather than at import.
    """
    import ast
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent.parent
    offenders = []
    for path in sorted((repo / "views_pipeline_core").rglob("*.py")):
        if "__pycache__" in path.parts or path.parent.name == "logging":
            continue
        for node in ast.parse(path.read_text()).body:  # MODULE SCOPE only
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "setLevel":
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue  # inside a definition — runs when called, not on import
                    offenders.append(f"{path.relative_to(repo)}:{node.lineno}")

    assert not offenders, (
        "these modules set a log level at import, overriding whatever the application "
        f"configured: {offenders}"
    )
