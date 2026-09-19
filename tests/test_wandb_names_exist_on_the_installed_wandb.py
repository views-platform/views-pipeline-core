"""Every wandb name this package evaluates exists on the installed wandb, and every call
binds against the installed signature.

Why this test exists (#519, #508): the manifest held `wandb ^0.18.7` (= `<0.19`) from
2.3.0 through 3.2.0, ten minor versions behind the wandb that views-r2darts2 0.2.x
requires (`>=0.28.2`), so no environment could resolve both. The ceiling was widened to
`<1.0` on 2026-09-18 after running the full suite and an offline end-to-end `WandBModule`
run on 0.30.0. What that measurement cannot do is repeat itself on the next wandb release —
this test can. It is the guard #508 asked for before the ceiling moved: "please do not
raise the ceiling without exercising those four" (`wandb.summary._as_dict`,
`wandb.sdk.wandb_run.Run`, `wandb.apis.public.runs.Run`, `wandb.old.summary`). It also
reaches the one path no measurement did: a sweep (`wandb.sweep`/`wandb.agent`) needs a
server, so offline mode cannot run one — but the keywords the package passes to them can
be bound against the installed signatures, and are.

The set of names and calls is DERIVED from the package's own source by walking every
module's AST — not hand-listed; a hand-list would be the next "guard wrong about its
own scope" in this register (C-259 family). What the walk sees: attribute chains rooted at
the name each module binds `wandb` to (`import wandb`, or `import wandb as w`),
`from wandb… import …`, and the positional count and keyword names of every call on such
a chain. What it deliberately does not see, stated so the next reader does not add it:
names inside docstrings and comments (`wandb.old.summary`, `wandb.Image`, `wandb.Histogram`
appear only there — never evaluated, so their absence cannot break anything); string
annotations (`'wandb.apis.public.runs.Run'` is never resolved at runtime; `wandb.Api`,
which is, is covered); and dynamic access — `getattr(wandb, …)`, `importlib.import_module`,
a `w = wandb` rebinding. None of the three exists in the package today (grepped
2026-09-18); if one is ever written, it is invisible here. A call that unpacks `*args` or
`**kwargs` into wandb is the one thing the walk sees but cannot check, so it refuses
rather than vouches: such a call fails this test until its arguments are written out.

Three roots are bound to a live run and hold a pre-init placeholder until `init` —
`wandb.run`, `wandb.summary`, `wandb.config` — so a chain through them is resolved on the
class wandb binds there after `init` (measured on 0.18.7 and 0.30.0:
`wandb.sdk.wandb_run.Run`, `wandb.sdk.wandb_summary.Summary`, `wandb.sdk.wandb_config.Config`).
That is the one place this test knows something the source does not say;
`wandb.summary._as_dict` is the private name that makes it worth knowing. The placeholder's
`__getattr__` raises wandb's own `Error`, not `AttributeError`, which is why the chain is
never walked through the placeholder itself.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest

import views_pipeline_core

PACKAGE_ROOT = Path(views_pipeline_core.__file__).parent

# The class wandb binds to each run-scoped module attribute once a run is active. Resolving
# `wandb.summary._as_dict` against the module would hit the placeholder; against the class
# it checks the thing the call site actually reaches.
RUN_BOUND_ROOTS = {
    "wandb.run": "wandb.sdk.wandb_run.Run",
    "wandb.summary": "wandb.sdk.wandb_summary.Summary",
    "wandb.config": "wandb.sdk.wandb_config.Config",
}

# A walk that found fewer than this has lost the package, not the package its wandb use;
# 26 chains and 28 calls on 2026-09-18. A floor, not a list — a list would rot when a call
# site is legitimately retired.
MIN_CHAINS_EXPECTED = 10


@dataclass(frozen=True)
class WandbCall:
    chain: str
    positional: int
    keywords: tuple[str, ...]
    unpacks: bool  # `*args` or `**kwargs` at the call — the arguments cannot be read statically
    site: str


def _wandb_bindings(tree: ast.Module) -> set[str]:
    """The local names this module binds the wandb package to (`wandb`, or an alias)."""
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "wandb"
    }


def _chain_of(node: ast.expr, roots: set[str]) -> str | None:
    """`wandb.sdk.wandb_run.Run` for an Attribute chain rooted at a wandb binding, else None."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name) and cur.id in roots and parts:
        return "wandb." + ".".join(reversed(parts))
    return None


def _wandb_use_in_the_package() -> tuple[dict[str, str], list[WandbCall]]:
    """Every dotted `wandb.…` name the package evaluates (→ one site), and every call on one."""
    chains: dict[str, str] = {}
    calls: list[WandbCall] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        roots = _wandb_bindings(tree)

        def site(node: ast.AST) -> str:
            return f"{path.relative_to(PACKAGE_ROOT.parent)}:{node.lineno}"

        for node in ast.walk(tree):
            # `from wandb import Api` / `from wandb.errors import CommError`; level > 0 is
            # the package's own `from .wandb import WandBModule`, not the SDK.
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and node.module.split(".")[0] == "wandb"
            ):
                for alias in node.names:
                    chains.setdefault(f"{node.module}.{alias.name}", site(node))
            elif isinstance(node, ast.Attribute):
                chain = _chain_of(node, roots)
                if chain:
                    chains.setdefault(chain, site(node))
            elif isinstance(node, ast.Call):
                chain = _chain_of(node.func, roots)
                if chain:
                    calls.append(
                        WandbCall(
                            chain=chain,
                            positional=len(node.args),
                            keywords=tuple(k.arg for k in node.keywords if k.arg is not None),
                            unpacks=any(isinstance(a, ast.Starred) for a in node.args)
                            or any(k.arg is None for k in node.keywords),
                            site=site(node),
                        )
                    )
    return chains, calls


def _resolve(chain: str) -> tuple[object, bool]:
    """Walk a dotted chain from the `wandb` module, importing submodules as needed.

    Returns the object and whether it was reached through a run-bound class (in which case
    a plain function found there is an unbound method whose first parameter is `self`).
    """
    through_run_bound = False
    for root, bound_class in RUN_BOUND_ROOTS.items():
        if chain.startswith(root + "."):
            chain = bound_class + chain[len(root):]
            through_run_bound = True
            break
    parts = chain.split(".")
    obj: object = importlib.import_module(parts[0])
    for i, part in enumerate(parts[1:], start=1):
        if hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        # A submodule not yet imported (`wandb.sdk.wandb_run` before anything touched it).
        # Only a module can have one; a class or function missing the name is just missing it.
        submodule = ".".join(parts[: i + 1])
        if not inspect.ismodule(obj):
            raise AttributeError(f"{'.'.join(parts[:i])} has no attribute {part!r}")
        try:
            obj = importlib.import_module(submodule)
        except ModuleNotFoundError as e:
            if e.name != submodule:
                raise  # the submodule exists; something IT imports is missing — not our finding
            raise AttributeError(f"{'.'.join(parts[:i])} has no attribute {part!r}") from e
    return obj, through_run_bound


_CHAINS, _CALLS = _wandb_use_in_the_package()


def test_the_derivation_sees_the_package():
    """The walk must find the names it exists to check; an empty set would pass vacuously."""
    assert len(_CHAINS) >= MIN_CHAINS_EXPECTED, sorted(_CHAINS)
    assert len(_CALLS) >= MIN_CHAINS_EXPECTED, _CALLS


@pytest.mark.parametrize("chain", sorted(_CHAINS))
def test_every_wandb_name_the_package_evaluates_exists_on_the_installed_wandb(chain: str):
    import wandb

    try:
        _resolve(chain)
    except AttributeError as e:
        pytest.fail(
            f"{chain} (used at {_CHAINS[chain]}) does not exist on wandb {wandb.__version__}: "
            f"{e}. Either the installed wandb moved it — cap the manifest below that version "
            f"and fix the call site — or the package grew a use of a name that never existed."
        )


@pytest.mark.parametrize("call", _CALLS, ids=[f"{c.chain}@{c.site}" for c in _CALLS])
def test_every_wandb_call_the_package_makes_binds_against_the_installed_signature(call: WandbCall):
    """The positional count and keyword names at each call site are accepted by the installed
    callable. This is what catches a keyword wandb removed (`sync` on `log`, `quiet` on
    `finish`, `goal` on `define_metric` all went between 0.18 and 0.30 — none used here) and
    it is the only check the sweep path gets, since a sweep cannot run offline."""
    import wandb

    if call.unpacks:
        pytest.fail(
            f"{call.chain} at {call.site} unpacks *args or **kwargs into wandb, so its arguments "
            f"cannot be checked against the installed signature. Write them out."
        )
    obj, through_run_bound = _resolve(call.chain)
    signature = inspect.signature(obj)
    positional = [None] * (call.positional + (1 if through_run_bound and inspect.isfunction(obj) else 0))
    try:
        signature.bind_partial(*positional, **{k: None for k in call.keywords})
    except TypeError as e:
        pytest.fail(
            f"{call.chain} at {call.site} is called with {call.positional} positional and "
            f"keywords {call.keywords}, which wandb {wandb.__version__} does not accept: {e}. "
            f"Installed signature: {signature}."
        )
