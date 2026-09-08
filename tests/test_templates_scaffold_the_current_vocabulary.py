"""A scaffolded source is born on the CURRENT vocabulary. #498, ADR-057, C-307.

Run: `conda run -n views_pipeline pytest tests/test_templates_scaffold_the_current_vocabulary.py -q`

## What this stops

ADR-057's transition window closes "when views-models reports no configs on the legacy
vocabulary". That count can never reach zero while this repo ships the generator that mints
new legacy configs: they migrate 123 sources, scaffold one model, and it returns to one.

`template_config_deployment.py` had **no tests at all**, in either family, while being the
thing that decides what every future source declares for its whole life (ADR-056: "whatever
the template emits is what runs in production for the lifetime of that model").

## Why these assert on generated OUTPUT, not on template source

The guard in `test_migrated_source_completes_a_run.py` AST-walks the package for the exact
constant `get_deployment_config`. It cannot see this defect: a template holds its output in
one large f-string, so the name is a substring of a multi-line constant and never matches.
Measured — that guard's offender list is identical with and without the `templates/`
exclusion it used to carry. Only generating the file and reading it back can see what a
template mints.

## Both families are discovered, never listed

#384 fixed `template_run_sh.py` in `model/` and left the identical defect in `ensemble/`.
The two config templates are character-identical, which is exactly the condition under which
a hand-listed test covers one twin and drifts on the other.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from views_pipeline_core.modules.validation.core_config_sniffer import (
    SUPPORTED_MATURITIES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_ROOT = REPO_ROOT / "views_pipeline_core" / "templates"

MATURITY_TEMPLATES = sorted(TEMPLATES_ROOT.glob("*/template_config_maturity.py"))
LEGACY_TEMPLATES = sorted(TEMPLATES_ROOT.glob("*/template_config_deployment.py"))
CONFIG_TEMPLATES = sorted(TEMPLATES_ROOT.glob("*/template_config_*.py"))

#: The families that exist today. Asserted rather than assumed, so a third family cannot
#: appear with no maturity template and no test noticing.
FAMILIES = {"model", "ensemble"}


def _load(template_path: Path):
    family = template_path.parent.name
    return importlib.import_module(
        f"views_pipeline_core.templates.{family}.{template_path.stem}"
    )


def _generate(template_path: Path, destination: Path) -> str:
    """Invoke a template's `generate()` and return what it wrote.

    Mirrors `test_generated_run_scripts_are_portable._generate`, including the reason for
    checking the return value: `save_python_script` writes the file BEFORE it py-compiles
    it and returns `False` rather than raising, so a file existing does not mean generation
    succeeded, and an unchecked return is a silent pass.
    """
    module = _load(template_path)

    parameters = list(inspect.signature(module.generate).parameters)
    extra = {"deployment_type": "shadow"}
    kwargs = {name: extra[name] for name in parameters[1:] if name in extra}
    unsupported = [
        name
        for name in parameters[1:]
        if name not in extra
        and inspect.signature(module.generate).parameters[name].default
        is inspect.Parameter.empty
    ]
    assert not unsupported, (
        f"{template_path.relative_to(REPO_ROOT)}::generate takes required parameter(s) "
        f"{unsupported} this test cannot supply, so it would be skipped silently."
    )

    wrote = module.generate(destination, **kwargs)
    assert wrote is not False, (
        f"{template_path.relative_to(REPO_ROOT)}::generate returned False. The file is "
        f"written before it is compiled, so its content cannot be trusted."
    )
    assert destination.exists()
    return destination.read_text()


def _ids(paths: list[Path]) -> list[str]:
    return [p.parent.name for p in paths]


# ----------------------------------------------------------------------------------
# The templates exist at all
# ----------------------------------------------------------------------------------


def test_every_family_has_a_maturity_template():
    """Derived, not listed: a third family must not arrive without one.

    The whole defect #498 records is that one family of generated file was left on the
    retired vocabulary while everything else moved. Asserting per-family coverage is the
    only thing that would say so.
    """
    assert {p.parent.name for p in MATURITY_TEMPLATES} == FAMILIES, (
        f"maturity templates found for {sorted(p.parent.name for p in MATURITY_TEMPLATES)}, "
        f"expected {sorted(FAMILIES)}"
    )


# ----------------------------------------------------------------------------------
# What a scaffolded source declares
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("template", MATURITY_TEMPLATES, ids=_ids(MATURITY_TEMPLATES))
def test_a_scaffolded_source_declares_the_current_vocabulary(template, tmp_path):
    """The regression #498 is about."""
    emitted = _generate(template, tmp_path / "config_maturity.py")

    assert "def get_maturity_config" in emitted
    assert "'maturity'" in emitted or '"maturity"' in emitted
    assert "get_deployment_config" not in emitted, (
        "a newly scaffolded source must not define the retired entry point"
    )
    assert "'deployment_status'" not in emitted, (
        "a newly scaffolded source must not declare the retired key"
    )


@pytest.mark.parametrize("template", MATURITY_TEMPLATES, ids=_ids(MATURITY_TEMPLATES))
def test_the_declared_maturity_is_one_the_platform_accepts(template, tmp_path):
    """Membership, derived from the sniffer — deliberately NOT set equality.

    The set a scaffolder may *mint* is a strict subset of the set a source may *hold*.
    Asserting equality against `SUPPORTED_MATURITIES` would force the scaffolder to be able
    to emit `retired`, which refuses to run — a source unrunnable the moment it is created.
    """
    emitted = _generate(template, tmp_path / "config_maturity.py")

    declared = {m for m in SUPPORTED_MATURITIES if f"'{m}'" in emitted}
    assert len(declared) == 1, (
        f"expected exactly one maturity in the emitted config, found {sorted(declared)}"
    )
    assert declared <= SUPPORTED_MATURITIES


@pytest.mark.parametrize("template", MATURITY_TEMPLATES, ids=_ids(MATURITY_TEMPLATES))
def test_a_new_source_is_born_a_candidate(template, tmp_path):
    """A separate assertion from the one above, and that is the point.

    Membership passing does not mean the value is right — `graduate` is a member too. This
    is the assertion that fails if the scaffolder ever promotes a source at birth, which is
    what views-models #444 did to thirteen models before #456 reverted it.
    """
    emitted = _generate(template, tmp_path / "config_maturity.py")

    assert "maturity_config = {'maturity': 'candidate'}" in emitted, (
        "a new source must be born a candidate. Maturity is earned; the scaffolder is not "
        "where a source is promoted."
    )


@pytest.mark.parametrize("template", MATURITY_TEMPLATES, ids=_ids(MATURITY_TEMPLATES))
def test_the_emitted_shape_is_the_one_views_models_specified(template, tmp_path):
    """views-models wrote the canonical text; this must not invent a variation.

    Their reference file (`8ac68a87:models/bad_blood/configs/config_maturity.py`) says in
    its own docstring: "the reference example for the migration — copy this shape exactly;
    do not invent variations." A generator emitting a different shape would leave their
    123-file migration with two shapes to reconcile.
    """
    emitted = _generate(template, tmp_path / "config_maturity.py")

    body = emitted[emitted.index("def get_maturity_config") :].strip()
    assert body == (
        "def get_maturity_config():\n"
        "    # Maturity settings\n"
        "    maturity_config = {'maturity': 'candidate'}\n"
        "    return maturity_config"
    ), f"emitted body diverges from views-models' reference shape:\n{body}"


# ----------------------------------------------------------------------------------
# The round trip — the loader and the validator, not a string comparison
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("template", MATURITY_TEMPLATES, ids=_ids(MATURITY_TEMPLATES))
def test_a_scaffolded_config_loads_and_satisfies_the_maturity_check(template, tmp_path):
    """Generate, load through the real loader, and put the value past the real check.

    Named for what it actually proves. It does **not** assert that a generated config
    passes `sniff_all()` — it cannot: the scaffold deliberately emits an incomplete config
    with `level` and `creator` commented out for the author to fill in, as
    `test_falsification_template_modelset_guard.py` already records. A test claiming
    otherwise would be asserting something it did not check, which is C-273's shape.

    What it does prove is the pairing that C-307 was about: `load_maturity_config` picks the
    entry point from the FILENAME, so a template writing `get_maturity_config` into a file
    the loader will ask `get_deployment_config` for produces `None` and a run refused for
    the wrong reason. Only loading it can see that.
    """
    from views_pipeline_core.managers.configuration.script_config import (
        load_maturity_config,
    )
    from views_pipeline_core.modules.validation.core_config_sniffer import (
        CoreConfigSniffer,
    )

    destination = tmp_path / "config_maturity.py"
    _generate(template, destination)

    loaded = load_maturity_config({"config_maturity.py": destination}, "scaffolded_source")
    assert loaded == {"maturity": "candidate"}, (
        f"the generated config did not load through the real loader: {loaded!r}"
    )

    sniffer = object.__new__(CoreConfigSniffer)
    sniffer._c = {"name": "scaffolded_source", **loaded}
    sniffer._check_deployment_status()  # must not raise


# ----------------------------------------------------------------------------------
# The filename and the entry point are decided in two different repositories
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("template", MATURITY_TEMPLATES, ids=_ids(MATURITY_TEMPLATES))
def test_the_maturity_template_refuses_the_legacy_filename(template, tmp_path):
    """The cross-repo pairing hole, closed at the only layer that can see both halves.

    views-models chooses the output path (`build_model_scaffold.py:175`), this repo chooses
    the entry point the file defines. So today they are one string literal away from a file
    named `config_deployment.py` that defines `get_maturity_config` — C-307 reproduced from
    the layer C-307's own guard cannot reach.
    """
    destination = tmp_path / "config_deployment.py"

    with pytest.raises(ValueError, match="config_maturity.py"):
        _load(template).generate(destination)

    assert not destination.exists(), "refused, but wrote the file anyway"


@pytest.mark.parametrize("template", LEGACY_TEMPLATES, ids=_ids(LEGACY_TEMPLATES))
def test_the_legacy_template_refuses_the_new_filename(template, tmp_path):
    """The same hole from the other side."""
    destination = tmp_path / "config_maturity.py"

    with pytest.raises(ValueError, match="template_config_maturity"):
        _load(template).generate(destination)

    assert not destination.exists(), "refused, but wrote the file anyway"


# ----------------------------------------------------------------------------------
# The census — a third legacy minter cannot appear
# ----------------------------------------------------------------------------------


def test_exactly_two_templates_still_mint_the_retired_vocabulary(tmp_path):
    """Derived from generated output, so it cannot be fooled by how a template is written.

    The two legacy templates are kept deliberately: views-models' scaffolders import them by
    module name and their fleet cannot accept `maturity` yet. This pins that set at exactly
    two, so a third cannot arrive unnoticed — and deleting the pair, when views-models
    flips, is a one-line edit here rather than a grep.
    """
    minters = set()
    for template in CONFIG_TEMPLATES:
        module = _load(template)
        parameters = list(inspect.signature(module.generate).parameters)
        # Supply whatever a template needs beyond the path; skip any that needs something
        # this census cannot invent, but say so rather than passing quietly.
        kwargs = {}
        for name in parameters[1:]:
            parameter = inspect.signature(module.generate).parameters[name]
            if parameter.default is not inspect.Parameter.empty:
                continue
            kwargs[name] = "census_probe"

        destination = tmp_path / template.parent.name / _emitted_name(template)
        destination.parent.mkdir(parents=True, exist_ok=True)
        module.generate(destination, **kwargs)
        emitted = destination.read_text()

        if "deployment_status" in emitted or "get_deployment_config" in emitted:
            minters.add(str(template.relative_to(TEMPLATES_ROOT)))

    assert minters == {
        "model/template_config_deployment.py",
        "ensemble/template_config_deployment.py",
    }, (
        f"the set of templates minting the retired vocabulary changed: {sorted(minters)}. "
        f"A new template must emit `maturity` (ADR-057, #498); the two legacy generators "
        f"are kept only until views-models' scaffolders stop importing them."
    )


def _emitted_name(template: Path) -> str:
    """`template_config_meta.py` -> `config_meta.py`.

    The templates that refuse a mismatched path (#498) need the real name; the rest do not
    care, and giving every template its own name keeps the census from tripping over its
    own probe.
    """
    return template.stem.replace("template_", "") + ".py"
