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

PACKAGE = Path(__file__).resolve().parent.parent / "views_pipeline_core"

#: Appwrite's own document fields, which are queryable without being declared by us.
_SYSTEM_FIELDS = {"$id", "$createdAt", "$updatedAt", "$permissions"}

#: `filters={...}` / `array_filters={...}`, in docstrings and in markdown alike. Matched
#: textually on purpose: a docstring example is text, and `ast` will not see inside it.
_FILTER_BLOCK = re.compile(
    r"\b(?:array_)?filters\s*=\s*(\{[^{}]*\})", re.DOTALL
)
_KEY = re.compile(r"""['"]([A-Za-z_$][A-Za-z0-9_]*)['"]\s*:""")


def _declared() -> set:
    return {a["key"] for a in DECLARED_METADATA_ATTRIBUTES} | _SYSTEM_FIELDS


def _documented_filter_keys():
    """Every attribute name any `filters=` example in the package mentions."""
    found = {}
    for path in sorted(PACKAGE.rglob("*")):
        if path.suffix not in {".py", ".md"} or "__pycache__" in path.parts:
            continue
        text = path.read_text()
        for block in _FILTER_BLOCK.finditer(text):
            line = text[: block.start()].count("\n") + 1
            for key in _KEY.findall(block.group(1)):
                found.setdefault(key, []).append(
                    f"{path.relative_to(PACKAGE.parent)}:{line}"
                )
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
