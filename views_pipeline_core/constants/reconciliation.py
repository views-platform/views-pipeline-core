"""Reconciliation-related constants.

Centralized per the user's directive.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Fail-loud message when reconciliation is configured but no concrete `Reconciler`
# was injected at the composition root. Single source of truth for the ensemble
# managers and the reconciliation port (no silent-off — see #194/#195).
# ---------------------------------------------------------------------------
RECONCILER_NOT_INJECTED_MSG: str = (
    "Reconciliation is configured but no Reconciler was injected. "
    "The composition root (the views-models ensemble main) must inject a concrete "
    "Reconciler. See issue #195."
)