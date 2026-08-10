"""Our datafactory format strings are verified against theirs. Issue #415, register C-62.

## The duplication this replaces

`LOA_TO_OUTPUT_FORMAT` maps a level of analysis to the `output_format` we ask
views-datafactory for. views-datafactory validates that argument against its own set.
The two were independent string literals in two repositories with nothing linking them, so
a rename on either side broke the other at runtime — and only when a datafactory model
actually ran.

The fix is not to copy their set here. A copy is the same defect with an extra step: it
goes stale silently, and nothing fails until production. This asks the **installed
package** what it accepts.

## Three outcomes, and they must not collapse into two

| Situation | Result |
|---|---|
| views-datafactory installed, vocabulary found | verify against it |
| views-datafactory **not installed** | **skip**, naming the package |
| installed, but the vocabulary **cannot be located** | **fail** |

The last row is the one that would ordinarily be got wrong. views-datafactory is an
optional dependency here, so failing on its absence would break the `test-core-only` CI
job. But "their package is not installed" and "their package is installed and I cannot
find the thing I am supposed to check" are different facts, and a check that skips on both
quietly stops checking the day they rename an attribute — while still reporting green.
That conflation is the defect this whole epic is about, so this file must not commit it.
"""

from __future__ import annotations

import pytest

from views_pipeline_core.data.constants import LOA_TO_OUTPUT_FORMAT

#: Where the vocabulary has lived, newest first. Probed in order rather than assumed:
#: `_VALID_FORMATS` is private and may move or vanish; `OutputFormat` is the public enum
#: it is derived from and is the better thing to bind to.
_VOCABULARY_SOURCES = (
    ("OutputFormat", lambda obj: {member.value for member in obj}),
    ("_VALID_FORMATS", set),
)


def _installed_datafactory():
    """The datafactory dataset module, or None when the package is absent."""
    try:
        import datafactory_query.dataset as dataset
    except ImportError:
        return None
    return dataset


def _their_valid_formats(dataset) -> set:
    """Every format the installed views-datafactory accepts.

    Raises `AssertionError` — not a skip — when the package is present but its vocabulary
    cannot be found. See the module docstring: absent and unlocatable are different facts.
    """
    for attribute, extract in _VOCABULARY_SOURCES:
        candidate = getattr(dataset, attribute, None)
        if candidate is not None:
            return extract(candidate)

    raise AssertionError(
        f"views-datafactory is installed but none of "
        f"{[name for name, _ in _VOCABULARY_SOURCES]} was found on "
        f"{dataset.__name__}. This check has stopped verifying anything. Find where they "
        f"declare valid output formats now and add it to _VOCABULARY_SOURCES — do NOT "
        f"turn this into a skip, which would leave the check green and blind."
    )


def test_every_format_we_request_is_one_they_accept():
    """The contract. A rename on their side fails here, not in a production run."""
    dataset = _installed_datafactory()
    if dataset is None:
        pytest.skip(
            "views-datafactory is not installed (it is an optional dependency here, and "
            "the test-core-only CI job runs without it). Nothing to verify against."
        )

    theirs = _their_valid_formats(dataset)
    ours = set(LOA_TO_OUTPUT_FORMAT.values())

    unknown = ours - theirs
    assert not unknown, (
        f"We ask views-datafactory for {sorted(unknown)}, which the installed version "
        f"does not accept. It accepts {sorted(theirs)}. Either they renamed a format or "
        f"LOA_TO_OUTPUT_FORMAT is wrong — a datafactory run would fail on this, and only "
        f"at fetch time."
    )


def test_the_vocabulary_probe_can_actually_find_something():
    """The control. If the probe silently found nothing, the test above would be vacuous."""
    dataset = _installed_datafactory()
    if dataset is None:
        pytest.skip("views-datafactory is not installed.")

    theirs = _their_valid_formats(dataset)
    assert theirs, "the probe returned an empty vocabulary, so nothing above is verified"
    assert isinstance(theirs, set)


def test_an_installed_package_with_no_locatable_vocabulary_fails_rather_than_skips():
    """The distinction the module docstring insists on, tested rather than asserted.

    A check that skips when it cannot find what it is checking stops checking silently.
    This proves the code path raises instead.
    """

    class _Moved:
        """Their module after a rename this file does not know about."""

        __name__ = "datafactory_query.dataset"

    with pytest.raises(AssertionError, match="stopped verifying"):
        _their_valid_formats(_Moved())


def test_we_do_not_keep_our_own_copy_of_their_vocabulary():
    """The point of the story: verify against theirs, never duplicate it.

    Looks for a literal COLLECTION of their formats — a set, list or tuple containing all
    of them. Copying their vocabulary back in would reintroduce C-62 while leaving the
    contract test above passing, because the copy would agree with itself.

    The first version of this check just looked for the three strings anywhere in a file,
    and flagged `constants.py` and `dataloaders.py` — both of which mention the names in
    prose and in our own two-entry mapping. A guard that fires on documentation gets
    switched off, so it parses the source instead of grepping it.
    """
    import ast
    import pathlib

    package = pathlib.Path(__file__).resolve().parents[1] / "views_pipeline_core"
    their_vocabulary = {"feature_frame", "dataframe", "country_month"}
    offenders = []

    for path in package.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a template, not importable Python
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
                continue
            literals = {
                element.value
                for element in node.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
            if their_vocabulary <= literals:
                offenders.append(f"{path.relative_to(package.parent)}:{node.lineno}")

    assert not offenders, (
        f"{offenders} enumerate views-datafactory's format vocabulary as a literal "
        f"collection. Ask the installed package instead — a copy is C-62 with an extra "
        f"step: it goes stale silently and nothing fails until a datafactory model runs."
    )


def test_our_mapping_covers_the_levels_of_analysis_we_support():
    """A mapping that lost an entry would fail loudly at fetch time, but late.

    `_check_level` accepts cm and pgm; the datafactory descriptor spells those as
    `country_month` and `priogrid_month`.
    """
    assert set(LOA_TO_OUTPUT_FORMAT) == {"priogrid_month", "country_month"}
