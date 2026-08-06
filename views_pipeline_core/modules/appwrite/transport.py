"""Bounding every Appwrite call in time. Register C-15, issue #248, story #347.

## What the drill found

Before any value was chosen, a transport that never returns was installed and each
Appwrite path exercised:

    PATH                         OUTCOME                            SECONDS
    list_files (read)            STILL BLOCKED after 3s             (no timeout)
    search_files_by_metadata     STILL BLOCKED after 3s             (no timeout)
    get_bucket                   STILL BLOCKED after 3s             (no timeout)

No timeout, no error, no recovery. A hung call hangs the delivery indefinitely rather
than failing — and an indefinite hang is worse than an error, because nothing downstream
ever learns anything. views-models, reaching the same service, has always set a timeout;
the package everything else depends on did not.

## Why this module exists rather than a constructor argument

**The SDK provides no hook.** `appwrite.client.Client` calls `requests.request(...)`
directly with no `timeout=` and exposes no `set_timeout` — verified against the installed
version, and pinned by a test that fails if that ever changes. There is no supported way
to pass a deadline through its API.

So the timeout is injected at the reference the SDK resolves: `appwrite.client.requests`
is replaced with a thin proxy that forwards everything and adds `timeout=` when the caller
did not supply one. That is invasive, and it is deliberately the *narrowest* invasive
option available:

* it does **not** touch the global `requests` module, so nothing outside the SDK changes;
* it does **not** use `socket.setdefaulttimeout()`, which would silently bound viewser,
  wandb and every other client in the process;
* it does **not** copy `Client.call`, which would need re-verifying on every SDK bump;
* it is idempotent, so repeated client construction cannot stack proxies;
* it defers to an explicit `timeout=` if the SDK ever starts passing one.

**A vendor workaround that outlives its reason becomes folklore**, so
`test_appwrite_timeout.py::TestTheWorkaroundIsTemporary` fails the day the SDK grows
native support, with instructions to delete this file.

## Why a timeout is a Cluster J concern

A timeout means **"I could not tell"** — never **"there is nothing there"**. Every path
that can now raise one must report it as a failure, which is the rule C-241 and C-258
exist to enforce, arriving by a different route.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Seconds to wait for a single Appwrite HTTP call before giving up.
#
# 30 is chosen against what this platform actually does: uploads are the slowest
# operation and a forecast shard is small enough that thirty seconds means something has
# gone wrong rather than something is large. Too low would convert a slow link into a
# failed delivery; too high recreates the hang the drill found, just politely.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0

# Operators on slow or distant links must be able to raise this without a release.
TIMEOUT_ENV_VAR = "APPWRITE_REQUEST_TIMEOUT_SECONDS"

_PROXY_MARKER = "_views_pipeline_core_timeout_proxy"


def resolve_timeout_seconds() -> float:
    """The configured timeout, or the default.

    A malformed override raises rather than falling back. Silently defaulting would be
    the Cluster J shape at the configuration boundary: the operator asked for something,
    we could not honour it, and we proceeded anyway without saying so.
    """
    from views_pipeline_core.exceptions.exceptions import ConfigurationException

    raw = os.getenv(TIMEOUT_ENV_VAR)
    if raw is None or raw == "":
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as e:
        raise ConfigurationException(
            f"{TIMEOUT_ENV_VAR}={raw!r} is not a number of seconds. Unset it to use the "
            f"default of {DEFAULT_REQUEST_TIMEOUT_SECONDS}s, or set a positive value."
        ) from e
    if value <= 0:
        raise ConfigurationException(
            f"{TIMEOUT_ENV_VAR}={raw!r} must be positive. A zero or negative timeout "
            f"would either fail every call or restore the unbounded hang this setting "
            f"exists to prevent."
        )
    return value


class _TimeoutInjectingRequests:
    """Forwards to `requests`, adding a default `timeout=` to `request()` calls."""

    def __init__(self, wrapped: Any, timeout: float) -> None:
        self._wrapped = wrapped
        self._timeout = timeout
        setattr(self, _PROXY_MARKER, True)

    def request(self, *args: Any, **kwargs: Any) -> Any:
        # `setdefault`, not assignment: if the SDK ever starts passing its own timeout,
        # it wins and this proxy becomes a no-op rather than silently overriding it.
        kwargs.setdefault("timeout", self._timeout)
        return self._wrapped.request(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def install_request_timeout(timeout: float | None = None) -> float:
    """Bound every Appwrite HTTP call. Idempotent; safe to call per client.

    Returns the timeout now in force, so a caller can log what it got rather than what
    it asked for.
    """
    import appwrite.client

    seconds = resolve_timeout_seconds() if timeout is None else float(timeout)
    current = appwrite.client.requests

    if getattr(current, _PROXY_MARKER, False):
        # Already wrapped. Update the value rather than nesting proxies — repeated
        # client construction is normal and must not build a chain.
        current._timeout = seconds
        return seconds

    appwrite.client.requests = _TimeoutInjectingRequests(current, seconds)
    logger.debug("Appwrite HTTP calls bounded at %ss", seconds)
    return seconds
