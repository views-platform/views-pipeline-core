"""Every `filters=` example in this package must name attributes that exist.

**Why this guard exists.** views-crafdapi measured its own serving queries and reported
that the `loa` filter in this package's documented examples is reached by nothing on
their serving path; views-appwrite then grepped the platform seam contract and the
coordinate registry and found `loa` in neither, while the contract's substrate annex has
named `name`/`category`/`type` as the metadata collection's filterable attributes all
along. So this package documented a filter that was never a platform fact.

Checking that, it turned out to be the smaller half. Several examples filtered on
`"model"`, which is not an attribute of the metadata collection at all — the field is
`name`. A consumer copying that example queries a column that does not exist.

`get_predictions_by_metadata` turns any caller-supplied `filters` dict straight into
`Query.equal(attribute, value)` (`modules/appwrite/file.py:1046-1050`), so an example is
not decoration here: it is the only specification a new consumer has of what may be
filtered on. The scope is DERIVED by reading the files, because a hand-listed one is how
eight guards in this repo came to be wrong about their own subject.
"""

import re
from pathlib import Path

import pytest

from views_pipeline_core.modules.appwrite.provisioning import (
    DECLARED_METADATA_ATTRIBUTES,
)

_REPO = Path(__file__).resolve().parent.parent

#: Everywhere a consumer reads a filter example. An audit planted
#: `filters={"model": ...}` in `documentation/CICs/DatastoreModule.md` and in the root
#: `README.md` and this guard stayed green both times — it scanned `views_pipeline_core/`
#: only. A consumer reads the CIC and the root README BEFORE the package docstrings, and
#: `DatastoreModule.md` was in fact already carrying two `loa` filter examples that the
#: narrower scan never saw. Derived over the three roots a reader actually uses.
_ROOTS = [_REPO / "views_pipeline_core", _REPO / "documentation", _REPO / "README.md"]

#: Appwrite's own document fields, which are queryable without being declared by us.
_SYSTEM_FIELDS = {"$id", "$createdAt", "$updatedAt", "$permissions"}

#: `filters={...}` / `array_filters={...}`, in docstrings and in markdown alike. Matched
#: textually on purpose: a docstring example is text, and `ast` will not see inside it.
#:
#: **Coverage is the keyword-literal form, and that is a limit rather than an oversight.**
#: An audit showed this guard green against `download_latest_file({"model": ...})`
#: (positional) and `filters=dict(model=...)`. Recognising those means deciding what an
#: arbitrary expression denotes, which is the prose-parsing trade register C-355 declined
#: after measuring it. Every example in this repository uses the keyword-literal form, so
#: the guard covers what is written rather than what could be; if that changes, the fix is
#: to normalise the examples, not to grow a parser here.
_FILTER_START = re.compile(r"\b(?:array_)?filters\s*=\s*\{")
_KEY = re.compile(r"""['"]([A-Za-z_$][A-Za-z0-9_]*)['"]\s*:""")


def _balanced_block(text: str, open_brace: int) -> str:
    """Return the `{...}` starting at `open_brace`, counting nested braces.

    The first version used `\{[^{}]*\}`, which stops at the first inner `{`. An audit
    planted `filters={"extra": {"name": "x"}, "model": "fatalities"}` and the guard saw
    only the inner dict, so the offending key was never examined — the pattern was blind
    to exactly the shape that hides a key from it.
    """
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace : i + 1]
    return text[open_brace:]


def _declared() -> set:
    return {a["key"] for a in DECLARED_METADATA_ATTRIBUTES} | _SYSTEM_FIELDS


def _documented_filter_keys():
    """Every attribute name any `filters=` example mentions, across every reader-facing root."""
    found = {}
    paths = []
    for root in _ROOTS:
        if root.is_file():
            paths.append(root)
        else:
            paths.extend(
                p
                for p in root.rglob("*")
                if p.suffix in {".py", ".md"} and "__pycache__" not in p.parts
            )
    for path in sorted(set(paths)):
        text = path.read_text()
        for match in _FILTER_START.finditer(text):
            block = _balanced_block(text, match.end() - 1)
            line = text[: match.start()].count("\n") + 1
            for key in _KEY.findall(block):
                found.setdefault(key, []).append(f"{path.relative_to(_REPO)}:{line}")
    return found


def test_the_scan_finds_examples_at_all():
    """A derived scope that derives nothing passes vacuously."""
    found = _documented_filter_keys()
    assert len(found) >= 3, (
        f"only {sorted(found)} found — the `filters=` pattern no longer matches this "
        f"package's examples, so this guard is not checking anything"
    )


def test_every_documented_filter_names_a_declared_attribute():
    declared = _declared()
    offenders = {
        key: sites
        for key, sites in _documented_filter_keys().items()
        if key not in declared
    }
    assert not offenders, (
        "these documented filter keys are not attributes of the metadata collection, so "
        "a consumer copying the example queries a column that does not exist: "
        + "; ".join(f"{k} at {', '.join(v)}" for k, v in sorted(offenders.items()))
        + f". Declared attributes are {sorted(declared - _SYSTEM_FIELDS)}."
    )


@pytest.mark.parametrize("attribute", ["category", "type", "name"])
def test_the_attributes_the_consumer_measured_are_declared_and_indexed(attribute):
    """The three views-crafdapi actually filters on, and the platform contract's §8
    annex names as filterable. Pinned so a schema edit cannot quietly drop one."""
    from views_pipeline_core.modules.appwrite.provisioning import DECLARED_INDEXES

    assert attribute in _declared(), f"{attribute} is not a declared attribute"
    assert any(
        attribute in index["attributes"] for index in DECLARED_INDEXES
    ), f"{attribute} is filtered on by the consumer but no declared index covers it"
