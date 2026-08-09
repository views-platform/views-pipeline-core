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
from dataclasses import dataclass
from typing import Any, Dict, Mapping

#: Bumped whenever the record's field set changes. A sidecar written at a different
#: version is a recognisable mismatch rather than a parse error, so the read path (#413)
#: can treat it as "cannot be verified" instead of crashing on an unexpected key.
PROVENANCE_VERSION = 1

#: Canonical JSON: sorted keys, no incidental whitespace, non-ASCII escaped. Two
#: descriptors differing only in key order must digest identically — key order is not a
#: property of the specification, and treating it as one would report spurious mismatches
#: until people stopped believing the check.
_CANONICAL_JSON = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": True}


def _canonical(payload: Any) -> str:
    """Serialise to canonical JSON, or raise naming what could not be serialised."""
    try:
        return json.dumps(payload, **_CANONICAL_JSON)
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
    - ``level`` — ``cm`` and ``pgm`` are different rows entirely
    - ``provenance_version`` — so a field set change is a mismatch, not a crash

    ``drift_detection_ran`` is **not** here. It is #414's, held back deliberately so that
    story can demonstrate a field appearing and the version bump being honoured.
    """

    queryset_digest: str
    source: str
    partition: str
    month_first: int
    month_last: int
    level: str
    provenance_version: int = PROVENANCE_VERSION

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

        Both directions are checked explicitly. An unknown key means the sidecar was
        written by a version that knows something this one does not; a missing key means
        the reverse. Either way the honest answer is that this record cannot be trusted to
        describe that artifact — not that it can be partially reconstructed and compared.
        """
        expected = {field.name for field in dataclasses.fields(cls)}
        supplied = set(payload)

        unknown = supplied - expected
        if unknown:
            raise ValueError(
                f"Provenance record carries unknown field(s) {sorted(unknown)}. It was "
                f"written by a version that records something this one does not "
                f"understand, so it cannot be compared field-for-field."
            )

        required = {
            field.name
            for field in dataclasses.fields(cls)
            if field.default is dataclasses.MISSING
        }
        missing = required - supplied
        if missing:
            raise ValueError(
                f"Provenance record is missing required field(s) {sorted(missing)}. A "
                f"partially reconstructed record would compare equal on the fields it "
                f"happens to have, which is worse than refusing."
            )

        return cls(**{name: payload[name] for name in supplied})

    def differences_from(self, other: "CacheProvenance") -> Dict[str, tuple]:
        """Field-by-field differences, as ``{field: (mine, theirs)}``.

        Returned rather than reduced to a bool so the read path (#413) can name what
        changed. A refusal that says only "cache mismatch" sends the reader to this source
        file to find out what it means, which is not a remediation.

        Fields are enumerated from the dataclass, so a field added without a comparison is
        impossible rather than merely unlikely.
        """
        return {
            field.name: (getattr(self, field.name), getattr(other, field.name))
            for field in dataclasses.fields(self)
            if getattr(self, field.name) != getattr(other, field.name)
        }
