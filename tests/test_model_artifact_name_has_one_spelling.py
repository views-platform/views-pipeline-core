"""The artifact-name convention is spelled once, and the reader can read the writer. #459.

Run: `conda run -n views_pipeline pytest tests/test_model_artifact_name_has_one_spelling.py -q`

## What this stops

`files.utils.generate_model_file_name` **writes** model artifact names.
`ModelPathManager._get_artifact_files` **reads** them, and until #459 it did so with a
hand-built ``f"{run_type}_model_"`` — a second spelling of the same convention. The two
happened to agree, which is exactly how C-59 survives: change one and the other is left
behind silently, and the symptom is a loader that cannot find artifacts it just wrote.

#459 added a target discriminator to both sides. Two hand-rolled spellings of a *more*
complicated convention is how you get a real outage, so both now derive from
`MODEL_ARTIFACT_TEMPLATE`, and this file asserts they still agree.

The sibling of `test_cache_name_has_one_spelling.py`, for the other convention.
"""

from __future__ import annotations

import pytest

from views_pipeline_core.data.constants import (
    MODEL_ARTIFACT_TIMESTAMP_PATTERN,
    model_artifact_filename,
    model_artifact_stem_pattern,
)
from views_pipeline_core.files.utils import generate_model_file_name

RUN_TYPES = ["calibration", "validation", "forecasting"]


@pytest.mark.parametrize("run_type", RUN_TYPES)
@pytest.mark.parametrize("suffix", ["", "lr_sb", "lr_sb_best_lr_ns_best"])
def test_what_the_writer_writes_is_what_the_reader_matches(run_type, suffix):
    """The agreement that had no test. If it breaks, artifacts become unfindable."""
    name = generate_model_file_name(run_type, ".pt", targets_suffix=suffix)
    stem = name[: -len(".pt")]

    assert model_artifact_stem_pattern(run_type, suffix).match(stem), (
        f"the reader cannot match what the writer produced: {stem!r}. The two spellings "
        f"of the artifact convention have drifted (C-59)."
    )
    # and a caller asking for no particular suffix must still find it
    assert model_artifact_stem_pattern(run_type).match(stem)


@pytest.mark.parametrize("run_type", RUN_TYPES)
def test_the_no_suffix_name_is_byte_identical_to_the_pre_459_convention(run_type):
    """#459 must not rename a single existing artifact.

    The old format was `f"{run_type}_model_{timestamp}{ext}"`. If the empty-suffix case
    changed, every artifact ever written would stop being found — the outage this whole
    file exists to prevent, caused by the change meant to prevent it.
    """
    stamp = "20241105_143022"
    assert (
        model_artifact_filename(run_type, stamp, ".pt")
        == f"{run_type}_model_{stamp}.pt"
    )


def test_a_suffix_does_not_match_a_longer_suffix_that_starts_with_it():
    """The reason the match is anchored on both ends.

    `startswith(f"{run_type}_model_sb_")` matches an artifact written for `sb_best`,
    because `sb_best_...` starts with `sb_`. Asking for one model's artifact and getting
    another's is a silently wrong answer — the class this codebase keeps finding.
    """
    sb_only = model_artifact_stem_pattern("calibration", "sb")

    assert sb_only.match("calibration_model_sb_20241105_143022")
    assert not sb_only.match("calibration_model_sb_best_20241105_143022")
    assert not sb_only.match("calibration_model_20241105_143022")


def test_the_bare_pattern_still_finds_suffixed_artifacts():
    """Callers that do not ask for a suffix keep seeing everything, as before #459."""
    anything = model_artifact_stem_pattern("calibration")

    for stem in (
        "calibration_model_20241105_143022",
        "calibration_model_sb_20241105_143022",
        "calibration_model_lr_sb_best_lr_ns_best_20241105_143022",
    ):
        assert anything.match(stem), stem

    assert not anything.match("validation_model_20241105_143022")
    assert not anything.match("calibration_model_notatimestamp")


def test_every_real_artifact_shape_on_this_machine_is_matched():
    """A control against the shape assumption, not against the code.

    The pattern requires `YYYYMMDD_HHMMSS`. That is what the writer produces — but the
    assumption is worth checking against reality rather than against itself, because two
    test fixtures in this repo used a date-only stamp the writer cannot emit, and #459's
    anchoring broke them. Every one of the 20 real artifacts on this machine matched;
    the fixtures were wrong, not the pattern.

    Skips where no sibling checkouts exist, like every other cross-repo check here.
    """
    import re
    from pathlib import Path

    siblings = Path(__file__).resolve().parents[2]
    if not siblings.is_dir():  # pragma: no cover
        pytest.skip("no sibling checkouts")

    stamps = []
    for artifact in siblings.rglob("*_model_*.pt"):
        parts = set(artifact.parts)
        if parts & {"envs", "site-packages", ".git"}:
            continue
        stamps.append(artifact.stem.split("_model_", 1)[1])
        if len(stamps) >= 200:
            break

    if not stamps:
        pytest.skip("no real artifacts on disk to check the shape against")

    bad = [s for s in stamps if not re.fullmatch(rf"(?:.+_)?{MODEL_ARTIFACT_TIMESTAMP_PATTERN}", s)]
    assert not bad, (
        f"{len(bad)} of {len(stamps)} real artifacts do not match the timestamp shape the "
        f"reader assumes: {bad[:5]}. The anchoring in `model_artifact_stem_pattern` would "
        f"make these unfindable."
    )
