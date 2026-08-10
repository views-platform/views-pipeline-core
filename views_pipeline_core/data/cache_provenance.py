"""What produced a cached artifact. Issue #411, epic #410, register C-61.

## Why this exists

`ViewsDataLoader` caches fetched data and serves it back on ``use_saved=True`` after
checking two things: that a file exists, and that its name matches
``{partition}_{source}_df{ext}``. Nothing records what *produced* the cache, so a
developer who edits ``config_queryset.py`` and re-runs with ``--saved`` trains on the
previous specification's data. The result is complete, correctly shaped, correctly
indexed, and wrong — it passes every check the loader has.

This module is the record that makes that detectable. It is deliberately **pure**: no
filesystem, no pandas, no `views_frames`. Writing the record is #412's; comparing it is
#413's. Keeping the three apart means each is independently reviewable and independently
revertable.

## Everything here is already resolved elsewhere

`ViewsDataLoader._resolve_fetch_context` reads ``get_queryset()`` exactly once (the #289
single-read contract) and returns a frozen `FetchContext` carrying the partition, the
resolved month range, the detected source, and the queryset object itself. Source
detection and the fetch therefore always describe one snapshot. Nothing in this module
needs to re-read anything; it turns values that already exist into a record.

## The digest refuses rather than degrades

`queryset_digest` raises on anything it cannot canonically serialise. It has no fallback
to ``repr()``, ``str()`` or ``id()``, and adding one would be actively harmful: a weak
digest makes a *mismatched* cache look verified, converting a loud problem into a silent
one. That is the defect this epic exists to fix, so reintroducing it here would be
self-defeating. The two shapes that reach it in practice are viewser's ``Queryset`` (a
pydantic model) and views-datafactory's descriptor (a plain dict); anything else is
something nobody has thought about, and saying so is the correct response.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

#: Bumped whenever the record's field set changes. A sidecar written at a different
#: version is a recognisable mismatch rather than a parse error, so the read path (#413)
#: treats it as "cannot be verified" and refetches instead of crashing on an unexpected key.
#:
#: 1 -> 2 (#414): gained ``drift_detection_ran``. Every v1 record on disk becomes a version
#: mismatch and is refetched once — which is the whole reason the read path distinguishes
#: a version difference from a corrupt record.
PROVENANCE_VERSION = 2

#: Canonical JSON: sorted keys (recursively — verified, not assumed), no incidental
#: whitespace, non-ASCII escaped. Two descriptors differing only in key order must digest
#: identically — key order is not a property of the specification, and treating it as one
#: would report spurious mismatches until people stopped believing the check.
#:
#: ``allow_nan=False`` matters more than it looks. Python's ``json`` emits bare ``NaN`` and
#: ``Infinity`` by default, which is **not JSON** — no other parser is obliged to read it.
#: Left on, a queryset carrying a NaN would digest happily to a value derived from a
#: string nothing else can parse. This module refuses what it cannot faithfully represent
#: everywhere else; accepting it here purely because CPython is lenient would be the same
#: quiet degradation the digest exists to prevent.
_CANONICAL_JSON = {
    "sort_keys": True,
    "separators": (",", ":"),
    "ensure_ascii": True,
    "allow_nan": False,
}


class ProvenanceRecordInvalid(ValueError):
    """The record is malformed — missing required fields, or otherwise unreadable.

    Distinct from `ProvenanceVersionMismatch` because the two demand different responses
    and #413 has to tell them apart: a malformed record means something is wrong and the
    operator should look at it, while an older record is an ordinary consequence of
    upgrading. Collapsing both into a bare ``ValueError`` would force the read path to
    guess from the message text.
    """


class ProvenanceVersionMismatch(ValueError):
    """The record was written at a different `PROVENANCE_VERSION`.

    Not an error in the artifact — an artifact from before (or after) this code. It cannot
    be compared field-for-field, because the two versions do not agree on what the fields
    are, so the honest outcome is "cannot be verified" and the safe response is to refetch.
    """

    def __init__(self, found: Any, expected: int) -> None:
        self.found = found
        self.expected = expected
        super().__init__(
            f"Provenance record is version {found!r}; this code writes and compares "
            f"version {expected}. The record cannot be verified field-for-field across "
            f"versions — refetch rather than trusting or rejecting it."
        )


def _canonical(payload: Any) -> str:
    """Serialise to canonical JSON, or raise naming what could not be serialised."""
    try:
        return json.dumps(payload, **_CANONICAL_JSON)
    except RecursionError as exc:
        # A structure too deep for the encoder. Caught explicitly because a bare
        # RecursionError escaping here would surface far from the queryset that caused
        # it, and this function's contract is that it names what it could not serialise.
        raise TypeError(
            "Queryset is nested too deeply to serialise for a provenance digest. It is "
            "either far larger than any real queryset or self-referential."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Queryset could not be canonically serialised for a provenance digest: "
            f"{exc}. The digest has no weaker fallback on purpose — one would make a "
            f"mismatched cache look verified, which is the failure (C-61) this record "
            f"exists to detect."
        ) from exc


def queryset_digest(queryset: Any) -> str:
    """A stable sha256 over the identity of a queryset or datafactory descriptor.

    Stable means *across processes*, not merely within one: caches outlive the run that
    wrote them, so a digest keyed on anything process-local would compare unequal on the
    next run and refetch forever.

    Accepts:
        - a **pydantic model** — viewser's ``Queryset``. It exposes ``model_dump``; it has
          no ``to_dict``/``to_json``/``name``, which is worth stating because those are
          the obvious things to reach for and they are not there.
        - a **mapping** — views-datafactory's descriptor, ``{"source": "views-datafactory",
          ...}``.

    Raises:
        TypeError: for anything else, including ``None``, naming the type. ``None`` in
            particular must never digest to a constant: `detect_data_source` already
            treats it as "no queryset found", and giving it a stable digest here would let
            two models with no queryset share a cache identity.
    """
    if queryset is None:
        raise TypeError(
            "Cannot digest a None queryset. A missing queryset is not an identity — "
            "digesting it to a constant would make every model without one share a "
            "cache identity."
        )

    dump = getattr(queryset, "model_dump", None)
    if callable(dump):
        # ``mode="json"`` resolves enums, dates and nested models to JSON-native values.
        # Without it, model_dump returns Python objects that json.dumps rejects, and the
        # refusal above would fire on a queryset that is perfectly serialisable.
        payload = dump(mode="json")
    elif isinstance(queryset, Mapping):
        payload = dict(queryset)
    else:
        raise TypeError(
            f"Cannot digest a queryset of type {type(queryset).__name__!r}. Expected a "
            f"pydantic model (viewser Queryset) or a mapping (views-datafactory "
            f"descriptor). There is deliberately no fallback: a digest that degrades "
            f"quietly would certify caches it never actually checked."
        )

    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheProvenance:
    """What produced a cached artifact, in the fields that make it identifying.

    Every field is here because changing it changes what the data *is*, so a cache that
    disagrees on any of them is not the cache this run asked for:

    - ``queryset_digest`` — the specification itself; the thing a developer edits
    - ``source`` — viewser and datafactory produce different data for the same queryset
    - ``partition`` — belt and braces with the filename, which also encodes it
    - ``month_first`` / ``month_last`` — a widened range must not read as a hit
    - ``level`` — ``cm`` and ``pgm`` are different rows entirely; Optional because
      ``get_data`` accepts None and the record states what was actually used
    - ``drift_detection_ran`` — whether the safety net actually ran (#414)
    - ``provenance_version`` — so a field set change is a mismatch, not a crash
    """

    queryset_digest: str
    source: str
    partition: str
    month_first: int
    month_last: int
    level: Optional[str]
    #: Did the fetch that produced this cache actually run drift detection? (#414)
    #:
    #: Deliberately **required**, with no default. A default would have to assert
    #: something about a cache nobody checked: ``True`` claims a safety net that may never
    #: have run, and ``False`` is equally an assertion. Requiring it means every
    #: construction site states what it knows, and `tests/conftest.py`'s fixture guard
    #: catches any that does not.
    #:
    #: **Not identifying.** It records something about how the cache was produced, not
    #: about *what* it contains, and the reader cannot know it in advance — a run reading
    #: a cache has not fetched, so it has no alerts to derive it from. Comparing it would
    #: refuse every viewser cache on the first read. See `IDENTIFYING` below.
    drift_detection_ran: bool = field(metadata={"identifying": False})
    provenance_version: int = PROVENANCE_VERSION

    @classmethod
    def identifying_fields(cls) -> tuple:
        """The fields that determine whether two caches are the same data.

        Derived from each field's metadata rather than listed, so a field added without
        deciding which kind it is defaults to *identifying* — the safe direction. A new
        identifying field wrongly treated as informational would silently stop being
        checked; the reverse merely refuses a cache that could have been served, which is
        loud and fixable.
        """
        return tuple(
            f.name for f in dataclasses.fields(cls) if f.metadata.get("identifying", True)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for the sidecar.

        Derived from ``dataclasses.fields`` rather than written out by hand: a field added
        later and forgotten here would be silently absent from every sidecar, which is a
        gap in exactly the record that exists to close gaps.
        """
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CacheProvenance":
        """Reconstruct from a sidecar, refusing anything that does not round-trip.

        **The version is checked first, and that ordering is load-bearing.** A record from
        another version differs in its field set *by definition* — #414 adds
        `drift_detection_ran` — so field-level validation would report it as malformed and
        the read path could not distinguish "this cache predates the current code" from
        "this cache is corrupt". Those need opposite responses: refetch versus stop and
        look. Checking the version first makes the distinction available instead of
        leaving #413 to parse an error message.

        After that, both directions are checked explicitly. An unknown key means the record
        knows something this code does not; a missing key means the reverse. Either way it
        cannot be trusted to describe the artifact — not partially reconstructed and
        compared.
        """
        supplied = set(payload)

        found_version = payload.get("provenance_version")
        # `type(...) is not int`, not `!=` and not `isinstance`. Python's numeric equality
        # makes `1.0 == 1` and `True == 1` both true, so a float or boolean version would
        # sail through a plain comparison and be stored as-is — a corrupt record accepted
        # as verified, which is the exact class of defect this check exists to catch.
        # `isinstance` would not help either: `bool` is a subclass of `int`.
        if type(found_version) is not int or found_version != PROVENANCE_VERSION:
            # A record with no version at all is also a version mismatch, not a missing
            # field: every record this code has ever written carries one, so its absence
            # means the writer predates the field.
            #
            # `expected` is passed explicitly rather than defaulted. A default argument is
            # bound at class-definition time, so an exception raised after
            # PROVENANCE_VERSION changed would report the value from import time — the
            # comparison above reading one number while the message reports another.
            raise ProvenanceVersionMismatch(found_version, PROVENANCE_VERSION)

        expected = {field.name for field in dataclasses.fields(cls)}
        unknown = supplied - expected
        if unknown:
            raise ProvenanceRecordInvalid(
                f"Provenance record declares version {PROVENANCE_VERSION} but carries "
                f"unknown field(s) {sorted(unknown)}. Same version, different fields — "
                f"that is a malformed record rather than an older one."
            )

        required = {
            field.name
            for field in dataclasses.fields(cls)
            if field.default is dataclasses.MISSING
        }
        missing = required - supplied
        if missing:
            raise ProvenanceRecordInvalid(
                f"Provenance record is missing required field(s) {sorted(missing)}. A "
                f"partially reconstructed record would compare equal on the fields it "
                f"happens to have, which is worse than refusing."
            )

        return cls(**{name: payload[name] for name in supplied})

    def differences_from(self, other: "CacheProvenance") -> Dict[str, tuple]:
        """Field-by-field differences, as ``{field: (self_value, other_value)}``.

        Order stated explicitly rather than left to the parameter names, because #413
        renders it into an operator-facing message. Call it as
        ``expected.differences_from(found)`` and each tuple reads ``(expected, found)``;
        reversed, the refusal would tell the operator their new queryset was the stale
        one, which is worse than saying nothing.

        Returned rather than reduced to a bool so the read path can name what changed. A
        refusal that says only "cache mismatch" sends the reader to this source file to
        find out what it means, which is not a remediation.

        Only **identifying** fields are compared. `drift_detection_ran` records how a
        cache was produced, not what it contains, and a run that is about to read a cache
        has not fetched anything and so cannot know its value — comparing it would refuse
        every viewser cache on first read.

        Fields are enumerated from the dataclass, so a field added without a comparison is
        impossible rather than merely unlikely.
        """
        return {
            name: (getattr(self, name), getattr(other, name))
            for name in self.identifying_fields()
            if getattr(self, name) != getattr(other, name)
        }
