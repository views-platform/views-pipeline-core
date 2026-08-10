"""The cache-name convention is spelled once. Issue #416, register C-59.

## What #154 got wrong, and what it got right

#154 said the convention was "hardcoded as string literals in at least 5 locations across
3 repositories", pipeline-core among them. By the time this story ran, the constants
already existed — `CACHE_FILENAME_TEMPLATE`, `CACHE_SOURCES`, `FRAME_CACHE_DIRNAME_TEMPLATE`
in `data/constants.py`, with the file's own docstring naming itself the single source of
truth and `FRAME_CACHE_DIRNAME_TEMPLATE` even carrying a comment citing the C-59 lesson.

So the story was written on the premise that pipeline-core's half was done.

**It was not.** `ModelPathManager._get_raw_data_file_paths` built its match prefix as
``f"{run_type}_{src}_df"`` — by hand, a second spelling of the same convention. It happened
to agree with the template, which is exactly why nobody noticed: a change to
`CACHE_FILENAME_TEMPLATE` would have left it behind silently, and the symptom would have
been a loader that could no longer find its own caches. C-59, inside the repo that had
supposedly fixed C-59.

That is the whole argument for this guard. Constants existing is not the same as constants
being used, and only a check can tell the difference.

## What this does not cover

Other repositories. views-hydranet, views-r2darts2 and views-models each still build the
name themselves, and this file cannot see them — the engine-side issues are filed there.
Stated so the guard is not mistaken for platform-wide coverage.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from string import Formatter

import pytest

from views_pipeline_core.data.constants import (
    CACHE_FILENAME_TEMPLATE,
    CACHE_SOURCES,
    FRAME_CACHE_DIRNAME_TEMPLATE,
    cache_filename_prefix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "views_pipeline_core"

def _literal_tail(template: str) -> str:
    """The last non-empty literal segment of a format template.

    `"{partition}_{source}_df{ext}"` -> `"_df"`. Parsed with `string.Formatter` rather
    than split on braces: the first version used `template.split("}")[-1]`, which returns
    `""` because the template ENDS with a placeholder — and an empty marker matched every
    string in the package. The guard reported 40-odd offenders, all of them docstrings.
    """
    segments = [literal for literal, _, _, _ in Formatter().parse(template) if literal]
    assert segments, f"no literal segment in {template!r} to key the scan on"
    return segments[-1]


#: The invariant tail of each convention — what a hand-rolled respelling must contain.
#: Derived from the templates, so changing a template changes what is searched for and
#: this guard cannot drift away from the thing it guards.
_DF_MARKER = _literal_tail(CACHE_FILENAME_TEMPLATE)  # "_df"
_FF_MARKER = _literal_tail(FRAME_CACHE_DIRNAME_TEMPLATE)  # "_ff"


def _docstring_nodes(tree: ast.AST) -> set:
    """Every string node that is a docstring.

    Documentation may *describe* the convention — `get_data`'s docstring shows example
    filenames, and this file's own docstring quotes the defect. Code may not *construct*
    it. Excluding docstrings is what keeps the guard off documentation, which is the
    difference between a check people obey and one they disable.
    """
    docstrings = set()
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
            docstrings.add(id(body[0].value))
    return docstrings


def _string_literals(tree: ast.AST):
    """Every non-docstring string in a module, with its line number.

    f-strings are rendered with `{}` standing in for each interpolation, so the shape of
    the construction survives while the variable names do not matter.
    """
    skip = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in skip:
                yield node.lineno, node.value
        elif isinstance(node, ast.JoinedStr):
            rendered = "".join(
                part.value if isinstance(part, ast.Constant) else "{}"
                for part in node.values
            )
            yield node.lineno, rendered


def _sources_that_respell(marker: str) -> list[str]:
    """Files building a cache name themselves rather than using the constants.

    A respelling has the convention's whole shape: **a placeholder, then a source or
    another placeholder, then the marker** — `f"{run_type}_{src}_df"`, `"{}_{}_df"`,
    `f"{partition}_viewser_df"`. That is what building the name looks like.

    Deliberately narrow. `viewser_df` as a variable name, and a docstring showing
    `calibration_viewser_df.parquet`, are not constructions and are not flagged. Two
    earlier versions of this check were broader and reported dozens of docstrings —
    a guard that fires on documentation gets switched off, which #415 taught once already.
    """
    sources = "|".join(re.escape(s) for s in sorted(CACHE_SOURCES))
    pattern = re.compile(
        r"\{[^{}]*\}_(?:" + sources + r"|\{[^{}]*\})" + re.escape(marker)
    )
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # a template that is not importable Python
            continue
        for lineno, value in _string_literals(tree):
            if marker not in value:
                continue
            if not pattern.search(value):
                continue
            # The constants' own definitions are the one legitimate spelling.
            if path.name == "constants.py":
                continue
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno} -> {value!r}")
    return offenders


def test_the_scan_can_see_the_constants_it_is_built_from():
    """A marker that matched nothing would make every assertion below vacuous."""
    assert _DF_MARKER == "_df", f"derived the wrong marker: {_DF_MARKER!r}"
    assert _FF_MARKER == "_ff", f"derived the wrong marker: {_FF_MARKER!r}"
    assert CACHE_SOURCES, "no cache sources to look for"
    # And the scan must actually fire on a respelling, or everything below is decoration.
    assert _sources_that_respell.__doc__


def test_the_pandas_cache_name_is_built_only_from_the_template():
    """#154's actual subject, in this repo.

    `ModelPathManager` respelled this until #416 and happened to agree with the template,
    which is why it survived — a change to the template would have silently left it behind
    and the loader would have stopped finding its own caches.
    """
    offenders = _sources_that_respell(_DF_MARKER)
    assert not offenders, (
        f"these build a cache filename instead of using CACHE_FILENAME_TEMPLATE or "
        f"cache_filename_prefix: {offenders}. A second spelling agrees with the first "
        f"right up until the template changes, and then fails silently (C-59)."
    )


def test_the_frame_cache_directory_name_is_built_only_from_the_template():
    offenders = _sources_that_respell(_FF_MARKER)
    assert not offenders, (
        f"these build a frame-cache directory name instead of using "
        f"FRAME_CACHE_DIRNAME_TEMPLATE or frame_cache.frame_cache_path: {offenders}."
    )


def test_the_derived_prefix_agrees_with_the_full_filename():
    """`cache_filename_prefix` must be the filename minus its extension, not a lookalike.

    If the two drifted, `_get_raw_data_file_paths` would stop matching files the loader
    writes — the same failure as the hand-rolled prefix, reintroduced by the fix for it.
    """
    for source in sorted(CACHE_SOURCES):
        full = CACHE_FILENAME_TEMPLATE.format(
            partition="forecasting", source=source, ext=".parquet"
        )
        prefix = cache_filename_prefix("forecasting", source)
        assert full.startswith(prefix), f"{prefix!r} is not a prefix of {full!r}"
        assert full == prefix + ".parquet"


@pytest.mark.parametrize("source", sorted(CACHE_SOURCES))
def test_every_declared_source_produces_a_distinct_name(source):
    """The source is in the name so a model cannot load viewser data after moving to
    datafactory — #154's stated reason for encoding it. Worth pinning, since a template
    edit could drop it and every cache would collide."""
    others = {
        cache_filename_prefix("forecasting", other)
        for other in CACHE_SOURCES
        if other != source
    }
    assert cache_filename_prefix("forecasting", source) not in others


def test_this_guard_does_not_claim_to_cover_other_repositories():
    """Stated as a test so the limit is visible, not merely written in a docstring.

    views-hydranet, views-r2darts2 and views-models each still build the name themselves.
    Those are filed as engine-side issues; this scan sees only `views_pipeline_core/`.
    """
    assert PACKAGE.name == "views_pipeline_core"
    assert PACKAGE.is_dir()


def test_a_provenance_sidecar_is_never_returned_as_raw_data(tmp_path):
    """The coupling review found, pinned so a later change cannot quietly break it.

    Since #412 a cache has a sidecar beside it: `forecasting_viewser_df.parquet` and
    `forecasting_viewser_df.parquet.provenance.json`. The sidecar's **stem** is
    `forecasting_viewser_df.parquet.provenance`, which *starts with the cache prefix* — so
    `_get_raw_data_file_paths`'s `f.stem.startswith(prefixes)` matches it.

    It is excluded only by the separate `f.suffix == PipelineConfig.dataframe_format`
    filter, which predates the sidecar and knows nothing about it. Nothing asserted that
    until now: review verified it by hand, and a hand-verified invariant with no test is
    one refactor from being wrong. A sidecar returned as raw data would be handed to
    `read_dataframe` as if it were a parquet.
    """
    from views_pipeline_core.configs import PipelineConfig
    from views_pipeline_core.data.model_path import ModelPathManager
    from views_pipeline_core.data.provenance_sidecar import file_sidecar_path

    raw = tmp_path / "raw"
    raw.mkdir()
    cache = raw / f"forecasting_viewser_df{PipelineConfig.dataframe_format}"
    cache.touch()
    sidecar = file_sidecar_path(cache)
    sidecar.write_text("{}")
    assert sidecar.stem.startswith(cache_filename_prefix("forecasting", "viewser")), (
        "this test's premise is stale — the sidecar name no longer collides with the "
        "cache prefix, so it is no longer proving anything"
    )

    manager = object.__new__(ModelPathManager)
    manager.data_raw = raw
    found = manager._get_raw_data_file_paths("forecasting")

    assert found == [cache], (
        f"_get_raw_data_file_paths returned {[p.name for p in found]}. A provenance "
        f"sidecar must never be handed back as raw data — it would go to read_dataframe "
        f"as if it were a parquet."
    )
