"""Every neighbouring repo this package names has a check, or a stated reason it does not.

Issue #429, epic #428.

Run: `conda run -n views_pipeline pytest tests/test_every_neighbour_has_a_conformance_check.py -q`

## What this is for

This package makes claims about roughly a dozen other repositories — it imports their
packages, builds their filenames, parses their configs, hands them frames. Some of those
boundaries have a conformance test. Some have a runtime `_require_*` probe. Some have
nothing, and **that was invisible**: nothing enumerated the boundaries, so nothing could
notice one was unguarded.

views-models was one of the unguarded ones, and it is the boundary the #422/#427 incident
went through — an agent in a third repo pushed a fix here because no check on either side
would have caught the mismatch it was working around.

A hand-written list of boundaries would rot the first time someone integrated an eleventh
repo. This codebase has hit that exact failure repeatedly: C-259, C-261, C-264, C-277 and
C-282 are all hand-listed worklists that missed a site, and #416 found `ModelPathManager`
respelling a convention the constants already owned. So the list is **derived**, and an
absence must be *justified in writing* rather than merely true.

## Where this deviates from #429, and why

#429 says the scan should not fire on documentation, on the grounds that "prose describing
a neighbour is not a claim about it." Implemented literally — excluding docstrings from
the derivation — the neighbour set loses **six of the twelve**: appwrite, baseline,
faoapi, hydranet, postprocessing and the wandb project token are named *only* in
docstrings. The guard would go blind precisely where the coverage gap is.

So docstrings count. A docstring in shipped code is a claim: it tells the next reader that
this package works with views-hydranet, and that reader will believe it. What is excluded
is `documentation/` and `reports/` — narrative trees that ship to nobody. The reference
implementation cited by the issue, `test_cache_name_has_one_spelling.py`, excludes
docstrings because it detects the *construction* of a filename and a docstring cannot
construct anything. This detects a *reference*, and a docstring plainly can refer.

## What "covered" does and does not mean

This test answers *"is there a check?"*, not *"is the check deep enough?"* — and the
difference is not academic. `appwrite` counts as covered on the strength of
`test_seam_contract_pin_is_coherent.py`, which verifies that the seam-contract citations
all point at one tag. That is a real check and it has caught real drift, but it is a
**documentation-citation** check: it says nothing about payload shape. The `models` and
`faoapi` exemptions below say exactly that about themselves ("the seam pin is checked, the
payload shape is not") and are counted as gaps; `appwrite` says the same thing and is
counted as covered.

Recorded rather than quietly graded, because a reader scanning "6 covered" would otherwise
draw a stronger conclusion than the data supports. Deepening it is B3's business (#430).

## Derive generously, allowlist explicitly

The scan pulls in things that are not repositories at all — the `views-platform`
organisation, a wandb project called `views-forecasting`. Rather than filter those out
with cleverness that could silently drop a real neighbour, they are allowlisted with the
reason written down. An over-inclusive scan plus an explicit reason is safe; a
hand-tuned exclusion rule is how a boundary disappears without anyone deciding it should.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "views_pipeline_core"
TESTS = REPO_ROOT / "tests"

#: `views-<name>` as it is spelled on GitHub. Greedy over hyphens so `views-pipeline-core`
#: yields `pipeline-core`, not `pipeline`.
_HYPHENATED = re.compile(r"views-([a-z0-9]+(?:-[a-z0-9]+)*)")

#: This repo. Not a neighbour of itself.
_SELF = {"pipeline-core"}

#: Filename markers that identify a file as a boundary check rather than a unit test.
_CONFORMANCE_MARKERS = ("contract", "conformance", "protocol", "seam", "canon")


@dataclass(frozen=True)
class Exemption:
    """A derived name with no conformance check, and why that is acceptable.

    Two different claims live here, and conflating them caused a real failure while
    building this file:

    - **A boundary with no check yet** (`not_a_repository=False`, the default). It could
      gain one, and when it does the exemption must be removed — `test_no_exemption_
      outlives_its_neighbour` enforces that, so a stale reason cannot sit next to a real
      check misleading the next reader.
    - **A token that is not a repository at all** (`not_a_repository=True`) — the GitHub
      organisation, a WandB project name. These can never gain a conformance check, so
      the same assertion must not apply. They are recorded rather than filtered out
      because an exclusion rule clever enough to drop them could also drop a real
      neighbour silently.
    """

    reason: str
    issue: str
    not_a_repository: bool = False


#: Neighbours deliberately without a conformance check.
#:
#: Each entry must say why and point somewhere a reader can follow. `test_every_exemption_
#: is_justified` enforces both — an entry saying only "skip" fails, which is the whole
#: point: an unexplained absence and a decided one must not look the same.
EXEMPT: dict[str, Exemption] = {
    "hydranet": Exemption(
        "Consumer, not a dependency: views-hydranet imports this package, not the "
        "reverse. The conformance check belongs on its side, where the expectation "
        "lives. Filed there.",
        "views-hydranet#257",
    ),
    "baseline": Exemption(
        "Consumer, not a dependency — same reasoning as views-hydranet. It pins this "
        "package by range and imports it; nothing here imports it.",
        "#428",
    ),
    "forecasts": Exemption(
        "`views_forecasts` is the prediction-store extension library, not a "
        "views-platform pipeline repo. Its surface here is `ViewsMetadata().get_runs()`, "
        "reached only on the prediction-store path, which is scheduled for retirement "
        "with the pandas tier (roadmap G5-G7).",
        "#313",
    ),
    "transformation-library": Exemption(
        "`views_transformation_library` is a pure function library — the shims in "
        "modules/dataloaders/update_viewser.py import it lazily precisely because it "
        "drags ingester3 and breaks CI without certificates. A conformance test would "
        "need those certificates to mean anything.",
        "#428",
    ),
    "frames-reconcile": Exemption(
        "Reached only through the injected `Reconciler` port "
        "(domain/reconciliation_port.py, #217), which is the abstraction a conformance "
        "test would otherwise have to invent. The port's contract is tested; the "
        "concrete package is substitutable by design.",
        "#217",
    ),
    "viewser": Exemption(
        "The upstream data source, not a views-platform repo. Its contract is exercised "
        "end-to-end by the loader tests rather than pinned by a fixture, because a "
        "frozen fixture of a live server's response is the thing that goes stale "
        "silently (C-218's lesson at the Appwrite seam).",
        "#428",
    ),
    "platform": Exemption(
        "`views-platform` is the GitHub organisation, not a repository. It appears in "
        "URLs. Recorded rather than filtered out: an exclusion rule clever enough to "
        "drop this could also drop a real neighbour without anyone noticing.",
        "n/a",
        not_a_repository=True,
    ),
    "forecasting": Exemption(
        "Not a repository — a WandB project name, `project='views-forecasting'` in a "
        "docstring example at modules/wandb/wandb.py:61. Same reasoning as "
        "`views-platform`: allowlisted with the reason rather than filtered.",
        "n/a",
        not_a_repository=True,
    ),
}


def _normalise(token: str) -> str:
    """`views_frames` / `views-frames` -> `frames`; `viewser` stays `viewser`."""
    for prefix in ("views_", "views-"):
        if token.startswith(prefix):
            return token[len(prefix):].replace("_", "-")
    return token


def declared_neighbours() -> set[str]:
    """Every other repo or package this package names, derived from its own source.

    Two kinds of claim, both machine-checkable:

    1. **An import.** `import views_frames` is the strongest possible claim about a
       neighbour — it asserts an API exists and has a shape.
    2. **A `views-<name>` mention** anywhere in a Python source file, docstrings
       included. See the module docstring for why docstrings count here.

    Scoped to `views_pipeline_core/`. `documentation/` and `reports/` are narrative and
    ship to nobody, so a repo named there is not a claim this package makes.
    """
    found: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # a template holding non-importable Python
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root.startswith("views"):
                        found.add(_normalise(root))
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                root = node.module.split(".")[0]
                if root.startswith("views"):
                    found.add(_normalise(root))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.update(_HYPHENATED.findall(node.value))
    return found - _SELF


def _conformance_files() -> list[Path]:
    """Boundary-check files, excluding this one.

    The exclusion is load-bearing and was found the hard way: this file's name contains
    "conformance" and its `EXEMPT` reasons name every uncovered neighbour, so on the first
    run it reported *itself* as the check for all of them. A guard that satisfies its own
    assertion is worse than no guard — it reads green while proving nothing.
    """
    return [
        p
        for p in sorted(TESTS.rglob("*.py"))
        if any(marker in p.name for marker in _CONFORMANCE_MARKERS)
        and p.resolve() != Path(__file__).resolve()
        and _defines_a_test(p)
    ]


def _defines_a_test(path: Path) -> bool:
    """Does this file actually assert anything, or does it just share constants?

    `tests/test_modules/contract_canon.py` matches the "canon" marker and is a
    shared-constants module with no test functions at all. It named `viewser`, and so
    conferred "coverage" on the upstream data source — a file that cannot fail was
    standing in for a check. Requiring a `test_*` function is the cheapest honest filter:
    a check that cannot fail is not a check.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(tree)
    )


def _require_probe_texts() -> list[tuple[str, str]]:
    """(`probe name`, `its source text`) for every `_require_*` function in the package.

    The probe body is what names the neighbour it guards — `_require_dense_report_consumer`
    checks views-reporting by importing it and inspecting what came back.
    """
    probes = []
    for path in sorted(PACKAGE.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("_require_")
            ):
                probes.append((f"{path.name}::{node.name}", ast.get_source_segment(source, node) or ""))
    return probes


def _spellings(neighbour: str) -> re.Pattern:
    """Every way this neighbour's name is actually written, as one pattern.

    Three cases, and getting this wrong is how a guard reads green while proving nothing:

    - `frames` -> `views-frames` or `views_frames`
    - `frames-reconcile` -> `views-frames-reconcile` or `views_frames_reconcile`, and any
      mix (`views-frames_reconcile`) because both spellings occur in the wild
    - `viewser` -> just `viewser`. It carries no `views-` prefix, so demanding one made it
      impossible to ever detect coverage for the upstream data source.

    **Built by splitting on the hyphen and escaping each part**, not by escaping the whole
    name and substituting afterwards. `re.escape` escapes `-` to `\-` (Python 3.7+), so
    `re.escape(n).replace('-', '[-_]')` produced `\[-_]` — a pattern matching the literal
    five characters `[-_]`. The first version of this function did exactly that, and every
    hyphenated neighbour was permanently undetectable. `test_the_matcher_matches_real_
    spellings` exists so that cannot recur silently.
    """
    separator = "[-_]"
    body = separator.join(re.escape(part) for part in neighbour.split("-"))
    if neighbour.startswith("views"):
        # e.g. `viewser` — already the whole package name, takes no prefix.
        return re.compile(rf"\b{body}\b")
    return re.compile(rf"views{separator}{body}\b")


def _names(text: str, neighbour: str) -> bool:
    return bool(_spellings(neighbour).search(text))


def _non_docstring_source(path: Path) -> str:
    """A file's source with its docstrings blanked out.

    Used only on the *detection* side, and the asymmetry with `declared_neighbours` is
    deliberate rather than an oversight:

    - deriving a neighbour, a docstring **is** a claim — "this package works with
      views-hydranet" is something the next reader will believe;
    - detecting a *check*, a docstring is **not** a test. It cannot assert anything.

    Without this, `tests/test_data/test_views_frames_conformance.py` conferred coverage on
    views-reporting because its docstring says *"We do NOT import views-reporting"* — a
    scope disclaimer counted as a check of the very boundary it disclaims.
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            for i in range(body[0].lineno - 1, body[0].end_lineno):
                lines[i] = "\n"
    return "".join(lines)


def checks_for(neighbour: str) -> list[str]:
    """Every conformance file or runtime probe that names this neighbour, in code.

    Docstrings are excluded — see `_non_docstring_source`. A file that only mentions a
    neighbour in prose is not checking it.
    """
    hits = [
        p.name for p in _conformance_files() if _names(_non_docstring_source(p), neighbour)
    ]
    hits += [name for name, text in _require_probe_texts() if _names(text, neighbour)]
    return hits


# ----------------------------------------------------------------------------------
# The scan must be able to see something before any assertion about it means anything
# ----------------------------------------------------------------------------------


def test_the_scan_finds_neighbours_at_all():
    """Population check. A scan returning nothing would make every assertion vacuous.

    Copied in spirit from `test_seam_contract_pin_is_coherent.py`. This repo has shipped
    a guard whose marker matched nothing and therefore passed on every input (#415, C-59).
    """
    neighbours = declared_neighbours()
    assert len(neighbours) >= 8, (
        f"only {len(neighbours)} neighbours derived ({sorted(neighbours)}) — the scan has "
        f"stopped seeing most of them, and every assertion below is now vacuous."
    )
    # Named spot-checks: if the derivation broke in a way that still returned enough
    # tokens, these are the ones whose absence would matter most.
    for expected in ("frames", "datafactory", "models", "reporting"):
        assert expected in neighbours, f"{expected} vanished from the derivation"


def test_the_scan_finds_conformance_files_at_all():
    files = _conformance_files()
    assert len(files) >= 5, f"only {len(files)} conformance-shaped test files found: {files}"


def test_the_scan_finds_require_probes_at_all():
    """Floor set just under the real count, and named spot-checks alongside it.

    A floor of 3 against an actual 4 tolerated losing a quarter of the probes without
    complaint — review's point, and a fair one. Names matter more than the count: a
    partial regression that still returned enough probes would slip a bare threshold.
    """
    probes = _require_probe_texts()
    assert len(probes) >= 4, f"only {len(probes)} `_require_*` probes found: {probes}"
    found = {name.split("::")[-1] for name, _ in probes}
    for expected in ("_require_dense_report_consumer", "_require_evaluation_source_consumer"):
        assert expected in found, f"{expected} vanished from the probe scan; found {sorted(found)}"


def test_the_matcher_matches_real_spellings():
    """The matcher's own unit test, and the reason it exists.

    The first version built its pattern as `re.escape(n).replace('-', '[-_]')`. Because
    `re.escape` escapes `-` to `\\-`, that produced `\\[-_]` — a pattern matching the
    literal characters `[-_]`. Every hyphenated neighbour became permanently undetectable,
    and `viewser` was unmatchable besides, since the pattern demanded a `views-` prefix it
    does not carry.

    Nothing failed. `checks_for('frames-reconcile')` simply returned `[]` forever, and the
    exemption-staleness assertion built on it was inert for three of eleven entries. That
    is this guard's own failure mode — silent success — reproduced inside the guard.
    """
    assert _names("from views_frames_reconcile import R", "frames-reconcile")
    assert _names("views-frames-reconcile", "frames-reconcile")
    assert _names("views-frames_reconcile", "frames-reconcile")
    assert _names("import views_transformation_library", "transformation-library")
    assert _names("from viewser import Queryset", "viewser")
    assert _names("views-frames", "frames") and _names("views_frames", "frames")
    # and it must not over-match:
    assert not _names("views-framesomething", "frames")
    assert not _names("views_frames_reconcile", "frames"), (
        "`frames` must not match the longer `frames-reconcile` spelling"
    )
    assert not _names("nothing to see", "frames")


def test_this_repo_is_not_its_own_neighbour():
    assert not (declared_neighbours() & _SELF)


def test_documentation_does_not_add_neighbours():
    """#429's must-not-fire criterion.

    `documentation/` and `reports/` name repos this package has no code relationship
    with. If the scan reached them, integrating nothing would still fail the test, and a
    guard that fails for reasons the author cannot act on gets deleted.
    """
    derived = declared_neighbours()
    doc_only = set()
    for tree_dir in (REPO_ROOT / "documentation", REPO_ROOT / "reports"):
        if not tree_dir.is_dir():
            continue
        for path in tree_dir.rglob("*.md"):
            doc_only.update(_HYPHENATED.findall(path.read_text(encoding="utf-8")))
    doc_only -= _SELF

    assert doc_only, "no repo names found in documentation/ — this control proves nothing"
    assert doc_only - derived, (
        "every repo named in documentation/ is also named in source, so this control "
        "cannot distinguish a scan that reads documentation from one that does not"
    )


# ----------------------------------------------------------------------------------
# Consumers this repo has never named — the blind spot the derivation cannot cover
# ----------------------------------------------------------------------------------
#
# Added 2026-08-13, after views-impact turned up as a real consumer that subclasses
# `ForecastingModelManager`, calls four other internals, and appears **nowhere** in this
# package's source.
#
# `declared_neighbours()` derives from what this repo *says*. That is the right basis for
# a dependency — naming one is a claim about it. It is structurally blind to a *consumer*,
# because being imported by someone leaves no trace in your own source. views-hydranet and
# views-baseline are in the derived set only because docstrings happen to mention them;
# had nobody written those sentences, they would be invisible too.
#
# So consumers are found the only way they can be: by looking at the repositories beside
# this one. That works where checkouts are colocated and skips in CI, the same limit as
# the wire-fixture drift check and the SessionAuth registry ratchet — and, as there,
# whoever has the sibling checked out is whoever can act.

_SIBLINGS = REPO_ROOT.parent

#: Consumers that import this package and need no conformance check here, with the reason.
#: Same discipline as EXEMPT: a reason long enough to disagree with, and somewhere to look.
KNOWN_CONSUMERS: dict[str, Exemption] = {
    "views-hydranet": Exemption(
        "Engine. Imports this package; nothing here imports it. The conformance check "
        "belongs on its side, where the expectation lives, and is filed there.",
        "views-hydranet#257",
    ),
    "views-baseline": Exemption(
        "Engine, same reasoning as views-hydranet. Pins a range and imports us.",
        "#428",
    ),
    "views-r2darts2": Exemption(
        "Engine. Also rebuilds the cache filename itself rather than reading the loader's "
        "exposed path, which is filed there as views-r2darts2#25 (the C-59 class).",
        "views-r2darts2#25",
    ),
    "views-stepshifter": Exemption(
        "Engine. Imports this package; no reciprocal claim is made here about it.",
        "#428",
    ),
    "views-impact": Exemption(
        "Engine, and the reason this check exists — it subclasses ForecastingModelManager "
        "and calls four internals while being named nowhere in our source. It is pinned "
        "<3.0.0 and uses the `targets` key retired in #380, so it cannot adopt 3.x "
        "unchanged; filed as views-impact#5.",
        "views-impact#5",
    ),
    "views-crafdapi": Exemption(
        "Consumer of published artifacts, like views-faoapi. Its payload contract is the "
        "ADR-013 wire canon, verified here by the three-way byte agreement in "
        "test_wire_fixture_conformance.py rather than by a per-repo fixture.",
        "#454",
    ),
}


def _sibling_consumers() -> list[str]:
    """Sibling repositories that import or pin this package.

    Excludes this repo, virtualenvs and site-packages — a vendored copy inside someone's
    `envs/` is not a consumer, it is an install.
    """
    if not _SIBLINGS.is_dir():  # pragma: no cover
        return []
    found = []
    for candidate in sorted(_SIBLINGS.iterdir()):
        if not candidate.is_dir() or candidate.resolve() == REPO_ROOT.resolve():
            continue
        if not candidate.name.startswith("views-"):
            continue
        for path in list(candidate.rglob("*.py")) + list(candidate.rglob("pyproject.toml")):
            parts = set(path.parts)
            if parts & {"envs", "site-packages", ".git", "node_modules", "wandb"}:
                continue
            try:
                if "views_pipeline_core" in path.read_text(encoding="utf-8", errors="ignore"):
                    found.append(candidate.name)
                    break
            except OSError:  # pragma: no cover
                continue
    return found


def test_the_sibling_scan_finds_consumers_at_all():
    """Control. A scan returning nothing would make the assertion below vacuous."""
    if not any(p.name.startswith("views-") and p.is_dir() for p in _SIBLINGS.iterdir()):
        pytest.skip("no sibling checkouts beside this repo")
    consumers = _sibling_consumers()
    assert len(consumers) >= 5, (
        f"only {len(consumers)} sibling consumers found ({consumers}) — the scan has "
        f"stopped seeing most of them."
    )


def test_every_consumer_on_disk_is_accounted_for():
    """A repository that imports this package must be known to this repo.

    Not "must have a conformance test" — most consumers are correctly checked from their
    own side. It must be *known*: either this repo names it (so the derivation sees it) or
    it is listed above with a reason. Anything else is a repo depending on us that nobody
    here has thought about, which is how views-impact ended up pinned two majors behind on
    a config key we retired, with nothing to tell it.
    """
    if not _SIBLINGS.is_dir():  # pragma: no cover
        pytest.skip("no sibling checkouts")
    consumers = _sibling_consumers()
    if not consumers:
        pytest.skip("no sibling consumers on disk")

    derived = {f"views-{n}" for n in declared_neighbours()}
    unaccounted = [c for c in consumers if c not in derived and c not in KNOWN_CONSUMERS]

    assert not unaccounted, (
        f"these repositories import or pin views_pipeline_core and this repo knows nothing "
        f"about them: {unaccounted}. Add each to KNOWN_CONSUMERS with a reason and an "
        f"issue, or give it a conformance check. A consumer leaves no trace in our source, "
        f"so the derivation cannot find it — this scan is the only thing that can."
    )


@pytest.mark.parametrize("consumer", sorted(KNOWN_CONSUMERS))
def test_every_known_consumer_is_justified(consumer):
    """Same bar as EXEMPT — an entry saying "skip" fails."""
    entry = KNOWN_CONSUMERS[consumer]
    assert len(entry.reason) >= 40, f"KNOWN_CONSUMERS['{consumer}'] reason is too thin"
    assert entry.issue, f"KNOWN_CONSUMERS['{consumer}'] has no issue reference"


# ----------------------------------------------------------------------------------
# The assertion itself
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("neighbour", sorted(declared_neighbours()))
def test_neighbour_has_a_conformance_check_or_a_stated_reason(neighbour):
    """#428's DoD-2, per neighbour.

    Adding a reference to a new `views-*` repo fails this until it has a check or an
    entry in `EXEMPT` saying why not. That is the intended friction: integrating a
    repository is the moment to decide how the boundary is verified, not later.
    """
    checks = checks_for(neighbour)
    if checks:
        return

    exemption = EXEMPT.get(neighbour)
    assert exemption is not None, (
        f"`views-{neighbour}` is named in this package's source but has no conformance "
        f"test, no `_require_*` probe, and no entry in EXEMPT. Either add a check, or "
        f"add an EXEMPT entry saying why one is not needed and where the decision is "
        f"recorded. An unguarded boundary and a deliberately unguarded one must not "
        f"look the same — that is how views-models stayed probe-only until #422."
    )


@pytest.mark.parametrize("neighbour", sorted(EXEMPT))
def test_every_exemption_is_justified(neighbour):
    """An entry must say something. `Exemption("skip", "")` fails here."""
    exemption = EXEMPT[neighbour]
    assert len(exemption.reason) >= 40, (
        f"EXEMPT['{neighbour}'] reason is {len(exemption.reason)} chars: "
        f"{exemption.reason!r}. Say why the boundary needs no check, in a sentence the "
        f"next reader can disagree with."
    )
    assert exemption.issue, f"EXEMPT['{neighbour}'] has no issue reference"


@pytest.mark.parametrize("neighbour", sorted(EXEMPT))
def test_no_exemption_outlives_its_neighbour(neighbour):
    """An allowlist that keeps entries for neighbours nobody references any more is how
    it stops describing reality. Also fails if a neighbour *gains* a check while keeping
    its exemption — the reason is then stale and misleads the next reader."""
    derived = declared_neighbours()
    assert neighbour in derived, (
        f"EXEMPT carries '{neighbour}' but nothing in the package names it any more. "
        f"Remove the entry."
    )
    if EXEMPT[neighbour].not_a_repository:
        # Cannot gain a conformance check — it is not a repository. A name-match against
        # some other file (views-platform appears in a URL inside the seam-pin test) is
        # not a check of it, and asserting otherwise is what made this test fail first
        # time round.
        return
    assert not checks_for(neighbour), (
        f"'{neighbour}' now has {checks_for(neighbour)} but is still in EXEMPT. Remove "
        f"the exemption — its stated reason is no longer true."
    )


def test_the_gap_is_visible_as_a_number():
    """DoD-2's headline, computed rather than claimed.

    Deliberately not asserted against a target: a test that failed when coverage improved
    would be gamed. It fails only if the derivation collapses, and otherwise exists so the
    ratio is printed somewhere a reader will find it.
    """
    neighbours = sorted(declared_neighbours())
    covered = [n for n in neighbours if checks_for(n)]
    assert len(covered) >= 4, (
        f"only {len(covered)} of {len(neighbours)} neighbours have a check "
        f"({covered}) — coverage has gone backwards."
    )
    real_repos = [n for n in neighbours if not (EXEMPT.get(n) and EXEMPT[n].not_a_repository)]
    covered = [n for n in real_repos if checks_for(n)]
    assert set(neighbours) == set(covered) | set(EXEMPT), (
        f"unaccounted neighbours: {sorted(set(neighbours) - set(covered) - set(EXEMPT))}"
    )
