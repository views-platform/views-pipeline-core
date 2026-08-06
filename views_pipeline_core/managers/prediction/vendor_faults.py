"""Resolving vendor exception types without importing the vendor.

## Why this file exists

`appwrite` is an **optional extra** (#345, register C-253). Before that change it sat in
`[tool.poetry.dependencies]`, so **views-hydranet, views-baseline and views-evaluation —
three repos containing zero references to Appwrite — installed its SDK anyway.** That is
CRP violated, measured rather than argued, and SDP inverted: the platform's
most-depended-upon package depended on a vendor SDK whose `databases.list*` surface
deprecated at Appwrite server 1.8.0.

The DIP seam was already right — `PredictionSaver` is a Protocol and `AppwriteSaver`
implements it. Only the packaging was wrong. But the module *defining* the Protocol did
`from appwrite.exception import AppwriteException` at module scope, so making the
dependency optional would have broken importing the Protocol itself. That was the
blocker found by falsification before this story started (F1), and this module is what
removes it.

## Why a shared helper rather than two copies

Two call sites need the identical tuple — `savers.py::AppwriteSaver.save` and
`io.py::PredictionIOManager`. WET before DRY says duplicate on first contact and extract
when a **second** incident shows the real shape. This is that second incident, the shape
is known and small, and the reasoning below is the part worth writing once: a copy that
drifted would silently stop catching a fault on one of the two upload paths.
"""

from __future__ import annotations

from typing import Tuple, Type

# Faults meaning "the upload never reached the substrate". These are stdlib and always
# available; the vendor's own exception is appended only when the extra is installed.
_TRANSPORT_FAULTS: Tuple[Type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def upload_transport_faults() -> Tuple[Type[BaseException], ...]:
    """Exception types an upload call may raise before the SDK converts them.

    `DatastoreModule.upload_data` reports failure by **return value** — the SDK's
    `AppwriteException` is turned into `OperationResult(success=False)` deep inside the
    storage module — so the vendor exception is mostly defensive here. It is included
    anyway, because "mostly" is not "never" and a fault that escapes the conversion must
    not take down a run that ADR-047 says should survive an Appwrite failure.

    When the extra is absent the tuple simply omits it: **an `AppwriteException` cannot
    be raised by an SDK that is not installed**, so the narrower tuple loses nothing.
    That is what makes this safe to resolve lazily rather than guarding every call site.

    Only `ImportError` is caught. If the SDK is installed but BROKEN — raising something
    else at import time — that error propagates out of the `except` clause and replaces
    the upload's original exception. Python preserves the original as `__context__`, so
    nothing is lost, and an SDK that cannot be imported is a larger problem than a masked
    upload fault. Widening this to `except Exception` would be the bare catch the
    Cluster J guard bans, so the narrow catch stands and the trade is written down.

    Returns:
        A tuple usable directly in an `except (...)` clause.
    """
    try:
        from appwrite.exception import AppwriteException
    except ImportError:
        return _TRANSPORT_FAULTS
    return _TRANSPORT_FAULTS + (AppwriteException,)
