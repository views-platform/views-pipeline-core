"""Generated `run.sh` scripts must run where this platform actually runs. Issue #384.

NOTE ON THE ID: the originating issue calls this "C-39", which is **views-models' register
entry**, not this repo's. C-39 here is an unrelated concern about duplicated rolling-origin
step-mapping. Registers are per-repo and their numbers collide; the shared identifier for
this defect is issue #384.

## The incident this guard closes

views-models fixed it in its own tree on 2026-04-21 (`83fb3a2e`, *"replace #!/bin/zsh
with #!/usr/bin/env bash in all 79 scripts"*) and marked the risk resolved. **It then
regressed 24 times**, because the fix went to the *output* and not to the *generator*.
Every regressed file postdates the fix: `first_love`, `bad_romance`, `smol_cat` (May 4),
`fake_model` (May 19), the twelve r2darts models (June 28). The template here was last
touched on 2026-04-03 — eighteen days *before* the fix — so every model scaffolded since
was born with the defect.

views-models has since added its own `tests/test_run_sh_portability.py`, which fails on any
tracked `.sh` declaring zsh. That catches the next occurrence **after** a model has been
scaffolded and committed. This file is what stops it being emitted in the first place.

## Why zsh is wrong here

zsh is not installed on Linux servers, Docker containers or CI runners, which is where
this platform runs. `./run.sh` fails outright with
`bad interpreter: /bin/zsh: No such file or directory`.

The scripts were never really zsh anyway: the body called `eval "$(conda shell.bash hook)"`
— the *bash* hook — under a zsh shebang. The interpreter declaration and the code it ran
disagreed from the day it was written.

## What this checks

Both templates are **discovered by globbing** `templates/*/template_run_sh.py`, never
listed. #384 named only the model template; the ensemble template carried the identical
defect and went unmentioned — a hand-listed worklist missing a site is the failure mode
this epic hit repeatedly (C-259, C-261, C-264).

Each discovered template is invoked for real and its output checked:

1. the shebang is `#!/usr/bin/env bash` — `env` rather than `/bin/bash`, because macOS
   ships bash 3.2 at `/bin/bash` while Homebrew's modern bash lands on `PATH`;
2. **`bash -n` parses the result** — the check that actually matters, since a shebang can
   be changed without the body being valid bash;
3. no generated script mutates `~/.zshrc` or sources it.

Check 2 is the one that could fail for a reason we do not already believe. Checks 1 and 3
only confirm the string we just wrote.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_ROOT = REPO_ROOT / "views_pipeline_core" / "templates"

REQUIRED_SHEBANG = "#!/usr/bin/env bash"

# Every kind of run-script generator in the package, found rather than named. A new
# template family (a third alongside model/ and ensemble/) is covered the day it lands.
RUN_SH_TEMPLATES = sorted(TEMPLATES_ROOT.glob("*/template_run_sh.py"))


def _generate(template_path: Path, destination: Path) -> str:
    """Invoke a template's `generate()` and return what it wrote.

    The two templates have different signatures — the model one takes a package name,
    the ensemble one hardcodes its environment — so the arguments are derived from the
    signature rather than assumed. Assuming would make this guard silently skip whichever
    template did not match the assumption.
    """
    family = template_path.parent.name
    module = importlib.import_module(
        f"views_pipeline_core.templates.{family}.{template_path.stem}"
    )

    parameters = list(inspect.signature(module.generate).parameters)
    extra = {"package_name": f"views_{family}_probe"}
    kwargs = {name: extra[name] for name in parameters[1:] if name in extra}

    unsupported = [name for name in parameters[1:] if name not in extra]
    assert not unsupported, (
        f"{template_path.relative_to(REPO_ROOT)}::generate takes parameter(s) "
        f"{unsupported} that this test does not know how to supply, so it would be "
        f"skipped silently. Teach the test the new parameter."
    )

    module.generate(destination, **kwargs)
    assert destination.exists(), (
        f"{template_path.relative_to(REPO_ROOT)}::generate returned without writing "
        f"{destination}. Nothing below can be checked."
    )
    return destination.read_text()


def _ids() -> list[str]:
    return [f"{p.parent.name}" for p in RUN_SH_TEMPLATES]


@pytest.mark.parametrize("template_path", RUN_SH_TEMPLATES, ids=_ids())
def test_generated_script_declares_a_portable_shebang(
    template_path: Path, tmp_path: Path
) -> None:
    """zsh is absent on the Linux servers, containers and CI runners this platform uses."""
    content = _generate(template_path, tmp_path / "run.sh")
    first_line = content.splitlines()[0]

    assert first_line == REQUIRED_SHEBANG, (
        f"{template_path.relative_to(REPO_ROOT)} generates a script starting "
        f"{first_line!r}. It must be {REQUIRED_SHEBANG!r}. zsh is not installed on the "
        f"Linux servers, Docker containers and CI runners this platform runs on, so "
        f"./run.sh fails with 'bad interpreter'. This regressed into 24 scaffolded models "
        f"because it was fixed in the generated output and not in this generator (#384)."
    )


@pytest.mark.parametrize("template_path", RUN_SH_TEMPLATES, ids=_ids())
def test_generated_script_actually_parses_as_bash(
    template_path: Path, tmp_path: Path
) -> None:
    """The check that can fail for a reason we do not already believe.

    Declaring bash in the shebang and emitting something bash cannot parse would leave
    the scripts just as broken, in a way an assertion on the first line cannot see.
    `bash -n` reads the whole script and reports syntax errors without executing a
    single command, so this is safe to run against a generator's output.
    """
    destination = tmp_path / "run.sh"
    _generate(template_path, destination)

    result = subprocess.run(
        ["bash", "-n", str(destination)], capture_output=True, text=True
    )

    assert result.returncode == 0, (
        f"{template_path.relative_to(REPO_ROOT)} generates a script that bash cannot "
        f"parse:\n{result.stderr}"
    )


@pytest.mark.parametrize("template_path", RUN_SH_TEMPLATES, ids=_ids())
def test_generated_script_does_not_mutate_the_users_shell_profile(
    template_path: Path, tmp_path: Path
) -> None:
    """A script named "run this model" must not rewrite the user's shell config.

    With ~130 generated scripts per consuming repo, appending to `~/.zshrc` on every run
    is a side effect no reader would predict. Sourcing it under a bash shebang is
    additionally incoherent — it asks bash to read a zsh config, which on macOS can emit
    syntax errors from zsh-only constructs.

    The environment variables that block needed are exported directly instead, which
    achieves the actual goal (libomp is findable for this run) without touching global
    state that outlives the process.
    """
    content = _generate(template_path, tmp_path / "run.sh")

    assert ">> ~/.zshrc" not in content, (
        f"{template_path.relative_to(REPO_ROOT)} generates a script that appends to the "
        f"user's ~/.zshrc. Export the variables in the script instead — the values are "
        f"only needed for the run."
    )
    assert "source ~/.zshrc" not in content, (
        f"{template_path.relative_to(REPO_ROOT)} generates a script that sources "
        f"~/.zshrc under a bash shebang, asking bash to read a zsh config."
    )


@pytest.mark.parametrize("template_path", RUN_SH_TEMPLATES, ids=_ids())
def test_template_source_neither_emits_nor_advertises_zsh(template_path: Path) -> None:
    """Docstrings promising a zsh script are how the next contributor re-adds one.

    This checks the two things that are actually wrong — a zsh shebang in the emitted
    string, and a docstring describing the output as a zsh script — rather than banning
    the substring "zsh" outright. A blanket ban would forbid the docstring note that
    warns future contributors OFF zsh, which is the opposite of the intent, and is the
    same mistake as a guard that names its own scope wrongly (C-259, C-261, C-264).
    """
    source = template_path.read_text()

    assert "#!/bin/zsh" not in source, (
        f"{template_path.relative_to(REPO_ROOT)} emits a zsh shebang — the defect that "
        f"regressed into 24 scaffolded models (#384)."
    )
    assert "executed in a zsh shell" not in source, (
        f"{template_path.relative_to(REPO_ROOT)}'s docstring still describes the "
        f"generated script as a zsh script. That description is an instruction to "
        f"whoever edits it next."
    )


def test_the_guard_found_every_run_script_generator() -> None:
    """A parametrized test over an empty list reports as passing.

    #384 named one template; there were two. This asserts the discovery glob found both
    families that exist today, so a guard watching nothing cannot masquerade as a guard
    watching everything.
    """
    families = {p.parent.name for p in RUN_SH_TEMPLATES}
    assert {"model", "ensemble"} <= families, (
        f"Expected run-script templates for at least the model and ensemble families; "
        f"found {sorted(families) or 'none'} under {TEMPLATES_ROOT}. Either a template "
        f"was moved or the discovery glob is wrong — in both cases this file is now "
        f"checking less than it appears to."
    )


# ---------------------------------------------------------------------------
# #366 — a generated main must not silence the only signal upstream provides
# ---------------------------------------------------------------------------

MAIN_TEMPLATES = sorted(TEMPLATES_ROOT.glob("*/template_main.py"))


@pytest.mark.parametrize(
    "template_path", MAIN_TEMPLATES, ids=[p.parent.name for p in MAIN_TEMPLATES]
)
def test_generated_main_does_not_blanket_suppress_warnings(
    template_path: Path, tmp_path: Path
) -> None:
    """`warnings.filterwarnings("ignore")` in a generated main eats upstream's only signal.

    views-datafactory zero-fills coverage gaps by design — their ADR-047 states that a
    filled month is *"structurally indistinguishable from months where the source observed
    zero events"*. The mitigation they shipped for that is a `UserWarning` from
    `load_dataset()`, plus `first_valid_*_month_id` / `last_valid_*_month_id` provenance.

    A bare `filterwarnings("ignore")` at the top of every generated model main discards
    that warning in exactly the process that needed it. So the platform's answer to
    "how do I know these zeros are real?" was suppressed by the scaffolding, not by a
    decision — and it reached ~130 generated scripts per consuming repo.

    A targeted suppression is fine and this check permits it: silence the categories you
    have decided are noise, by name. What it refuses is the blanket form, which silences
    categories nobody has considered — including ones that do not exist yet.

    The template module holds the generated code as a *string literal*, so parsing the
    template file finds no calls at all — a guard written that way passes unconditionally.
    This one generates the file first and parses the output, which is what ships.
    """
    source = _generate(template_path, tmp_path / "main.py")
    tree = ast.parse(source)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", None) != "filterwarnings":
            continue
        # A blanket suppression is `filterwarnings("ignore")` with nothing narrowing it:
        # no `category=`, no `module=`, no `message=`.
        narrowed = {kw.arg for kw in node.keywords} & {"category", "module", "message"}
        positional_filter = len(node.args) > 1
        if not narrowed and not positional_filter:
            offenders.append(
                f"{template_path.relative_to(REPO_ROOT)} -> generated main.py:{node.lineno}"
            )

    assert not offenders, (
        f"generated mains blanket-suppress every warning: {offenders}. This discards "
        f"views-datafactory's coverage `UserWarning` (their ADR-047), which is the only "
        f"signal distinguishing a real zero from a filled gap — in the process that "
        f"consumes the data. Narrow it: name the categories you have decided are noise."
    )
