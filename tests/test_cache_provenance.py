"""The provenance record and its digest. Issue #411, epic #410, register C-61.

## What is actually at risk here

This module is pure, so the usual worry — does it integrate — does not apply. The risks
are narrower and nastier:

1. **A digest that is not stable across processes.** Caches outlive the run that wrote
   them. A digest keyed on anything process-local (``id()``, ``hash()`` of a str, memory
   address) compares equal within one interpreter and unequal on the next run. Every
   in-process test would pass and every real cache would refetch forever. So stability is
   tested by digesting in a **subprocess** and comparing, which is the only way to see it.

2. **A digest that is stable but insensitive.** One that returns a constant is perfectly
   stable and useless. Tested by mutation, not by inspection.

3. **A record that silently drops a field.** ``to_dict``/``from_dict`` are derived from
   ``dataclasses.fields``; the tests derive their expectations the same way, so a field
   added without handling fails rather than passing quietly.

4. **A fallback creeping into the digest.** The refusal is the point. If someone later
   adds ``except Exception: return repr(...)`` to make an awkward type work, a mismatched
   cache starts looking verified — the exact defect (C-61) this record exists to detect.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest

from views_pipeline_core.data.cache_provenance import (
    PROVENANCE_VERSION,
    CacheProvenance,
    queryset_digest,
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

DESCRIPTOR = {
    "source": "views-datafactory",
    "dataset": "views_ged",
    "loa": "priogrid_month",
    "features": ["ged_sb_best", "ged_ns_best"],
    "region": "africa_me_legacy",
}


def _provenance(**overrides) -> CacheProvenance:
    base = {
        "queryset_digest": "abc123",
        "source": "datafactory",
        "partition": "forecasting",
        "month_first": 121,
        "month_last": 550,
        "level": "pgm",
    }
    base.update(overrides)
    return CacheProvenance(**base)


# ── the digest: stability ─────────────────────────────────────────────────────


def test_the_digest_is_stable_across_processes():
    """The only test that can see a process-local digest. See the module docstring.

    A digest computed from ``id()`` or from a salted ``hash()`` would be identical within
    one interpreter and different in the next — passing every in-process assertion while
    making every real cache miss.
    """
    program = (
        "import json,sys;"
        "sys.path.insert(0, %r);"
        "from views_pipeline_core.data.cache_provenance import queryset_digest;"
        "print(queryset_digest(json.loads(sys.argv[1])))" % str(REPO_ROOT)
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", program, json.dumps(DESCRIPTOR)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(2)
    ]

    assert runs[0] == runs[1] != "", f"digest differed between processes: {runs}"
    assert runs[0] == queryset_digest(DESCRIPTOR), (
        "the in-process digest disagrees with a fresh interpreter's — the digest depends "
        "on interpreter state, so a cache written by one run can never verify in another."
    )


def test_key_order_does_not_change_the_digest():
    """Key order is not a property of the specification.

    If it were significant, a descriptor rebuilt in a different order would report a
    spurious mismatch — and a check that cries wolf gets switched off.
    """
    reordered = dict(reversed(list(DESCRIPTOR.items())))
    assert reordered != DESCRIPTOR or list(reordered) != list(DESCRIPTOR)
    assert queryset_digest(reordered) == queryset_digest(DESCRIPTOR)


# ── the digest: sensitivity ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mutation, description",
    [
        ({"features": ["ged_sb_best"]}, "a feature removed"),
        ({"features": ["ged_sb_best", "ged_ns_best", "ged_os_best"]}, "a feature added"),
        ({"region": "global"}, "the region changed"),
        ({"loa": "country_month"}, "the level of analysis changed"),
        ({"dataset": "views_acled"}, "the dataset changed"),
        ({"source": "viewser"}, "the source changed"),
    ],
)
def test_the_digest_changes_when_the_specification_changes(mutation, description):
    """Every scenario #155 describes as 'happens during normal model development'."""
    changed = {**DESCRIPTOR, **mutation}
    assert queryset_digest(changed) != queryset_digest(DESCRIPTOR), (
        f"{description} did not change the digest, so a cache from the previous "
        f"specification would be served as a match."
    )


def test_feature_order_is_significant():
    """Deliberately asserted rather than assumed either way.

    Column order is part of the frame a model receives, so two querysets differing only in
    feature order are not interchangeable. Recording the decision here means a future
    change to it is visible rather than incidental.
    """
    reordered = {**DESCRIPTOR, "features": list(reversed(DESCRIPTOR["features"]))}
    assert queryset_digest(reordered) != queryset_digest(DESCRIPTOR)


# ── the digest: refusal ───────────────────────────────────────────────────────


def test_none_is_refused_and_not_digested_to_a_constant():
    """`detect_data_source` already treats None as 'no queryset'. A stable digest for it
    would let every model without a queryset share one cache identity."""
    with pytest.raises(TypeError, match="None"):
        queryset_digest(None)


@pytest.mark.parametrize(
    "unsupported",
    ["a string", 42, ["a", "list"], object(), (1, 2)],
    ids=["str", "int", "list", "object", "tuple"],
)
def test_unserialisable_querysets_are_refused_by_type(unsupported):
    """No fallback exists, and this is what stops one being added quietly."""
    with pytest.raises(TypeError) as excinfo:
        queryset_digest(unsupported)
    assert type(unsupported).__name__ in str(excinfo.value), (
        "the refusal does not name the offending type, so a caller cannot tell what to fix"
    )


def test_a_pydantic_style_model_is_accepted_via_model_dump():
    """viewser's Queryset is a pydantic model — it has `model_dump`, and specifically has
    no `to_dict`, `to_json` or `name`. Verified against the installed package 2026-08-09.

    Stubbed rather than constructed: building a real Queryset would make this a test of
    viewser's constructor, and it would need a live connection.
    """

    class _PydanticLike:
        def model_dump(self, mode=None):
            assert mode == "json", (
                "model_dump must be called with mode='json'; without it pydantic returns "
                "Python objects (dates, enums) that canonical JSON rejects, and a "
                "perfectly serialisable queryset would be refused."
            )
            return {"name": "fatalities003", "themes": ["conflict"]}

    digest = queryset_digest(_PydanticLike())
    assert len(digest) == 64, "not a sha256 hexdigest"
    assert digest == queryset_digest({"name": "fatalities003", "themes": ["conflict"]}), (
        "a model and the mapping it dumps to must digest identically — otherwise the "
        "digest keys on the wrapper rather than on the specification."
    )


def test_a_model_dump_that_returns_something_unserialisable_still_refuses():
    """The refusal must survive the pydantic branch, not only the else branch."""

    class _Awkward:
        def model_dump(self, mode=None):
            return {"when": object()}

    with pytest.raises(TypeError, match="canonically serialised"):
        queryset_digest(_Awkward())


# ── the record ────────────────────────────────────────────────────────────────


def test_the_record_is_frozen():
    provenance = _provenance()
    with pytest.raises(dataclasses.FrozenInstanceError):
        provenance.source = "viewser"


def test_every_field_round_trips():
    """Field list derived from the dataclass, so a new unhandled field fails here."""
    original = _provenance()
    restored = CacheProvenance.from_dict(original.to_dict())

    assert restored == original
    for field in dataclasses.fields(CacheProvenance):
        assert getattr(restored, field.name) == getattr(original, field.name), (
            f"field '{field.name}' did not survive the round trip"
        )


def test_to_dict_emits_exactly_the_declared_fields():
    assert set(_provenance().to_dict()) == {
        field.name for field in dataclasses.fields(CacheProvenance)
    }


def test_the_record_is_json_serialisable():
    """It is written to a sidecar in #412; a record that cannot be serialised is useless."""
    payload = json.dumps(_provenance().to_dict(), sort_keys=True)
    assert CacheProvenance.from_dict(json.loads(payload)) == _provenance()


def test_an_unknown_field_is_refused():
    """A sidecar from a future version records something this one cannot compare."""
    payload = {**_provenance().to_dict(), "drift_detection_ran": True}
    with pytest.raises(ValueError, match="unknown field"):
        CacheProvenance.from_dict(payload)


@pytest.mark.parametrize(
    "dropped",
    [
        field.name
        for field in dataclasses.fields(CacheProvenance)
        if field.default is dataclasses.MISSING
    ],
)
def test_a_missing_required_field_is_refused(dropped):
    """Partial reconstruction would compare equal on whatever fields survived."""
    payload = {k: v for k, v in _provenance().to_dict().items() if k != dropped}
    with pytest.raises(ValueError, match="missing required field"):
        CacheProvenance.from_dict(payload)


def test_the_version_defaults_and_is_recorded():
    assert _provenance().provenance_version == PROVENANCE_VERSION
    assert "provenance_version" in _provenance().to_dict()


def test_drift_detection_ran_is_not_in_this_version():
    """#414 adds it. Pinned so that story demonstrably changes the record rather than
    silently finding the field already present."""
    assert "drift_detection_ran" not in {
        field.name for field in dataclasses.fields(CacheProvenance)
    }


# ── differences ───────────────────────────────────────────────────────────────


def test_identical_records_report_no_differences():
    assert _provenance().differences_from(_provenance()) == {}


@pytest.mark.parametrize(
    "field_name, other_value",
    [
        ("queryset_digest", "different"),
        ("source", "viewser"),
        ("partition", "calibration"),
        ("month_first", 1),
        ("month_last", 999),
        ("level", "cm"),
        ("provenance_version", PROVENANCE_VERSION + 1),
    ],
)
def test_each_field_is_compared_independently(field_name, other_value):
    """One case per field. #413's refusal names what changed, so every field must be
    capable of being the thing that changed."""
    differences = _provenance().differences_from(_provenance(**{field_name: other_value}))
    assert field_name in differences, f"a change to '{field_name}' was not detected"
    assert differences[field_name] == (
        getattr(_provenance(), field_name),
        other_value,
    ), "the difference does not carry both values, so a message cannot report them"


def test_the_difference_check_covers_every_declared_field():
    """Derived, so a field added without a comparison is impossible rather than unlikely."""
    covered = {
        "queryset_digest",
        "source",
        "partition",
        "month_first",
        "month_last",
        "level",
        "provenance_version",
    }
    declared = {field.name for field in dataclasses.fields(CacheProvenance)}
    assert declared == covered, (
        f"CacheProvenance declares {sorted(declared)} but the per-field comparison test "
        f"covers {sorted(covered)}. Unverified: {sorted(declared - covered)}."
    )


# ── purity ────────────────────────────────────────────────────────────────────


def test_the_module_imports_without_pandas():
    """It sits under `data/`, which the package's import-purity guard governs, and both
    cache paths — pandas and FeatureFrame — must be able to use it."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
            "import views_pipeline_core.data.cache_provenance as m; "
            "sys.exit(1 if 'pandas' in sys.modules else 0)",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, (
        f"importing cache_provenance pulls in pandas: {result.stderr.decode()[-400:]}"
    )
