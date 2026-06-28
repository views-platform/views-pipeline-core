"""The reconciliation injection **port**: the `Reconciler` protocol (DIP).

Ensemble managers depend on this abstraction, never on a concrete reconciliation package.
The concrete, frames-native reconciler (`views_frames_reconcile.ReconciliationModule`, the
views-frames sibling, ≥1.8 for native point-broadcast + `reconcile_result`) is injected at the
composition root (views-models);
pipeline-core never imports it (ADP — no cycle). Geography (the `(time, priogrid) -> country`
mapping) is baked into the injected instance (views-frames ADR-014), so this port is frames-only.

Split from the former `domain.reconciliation` (#237): the port is a *stable abstraction* (SAP)
and changes for different reasons than the concrete `ReconciliationInvariants` value object,
which now lives in `domain.reconciliation_invariants`.
"""
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from views_frames import PredictionFrame

if TYPE_CHECKING:  # type-only — the concrete reconciler is injected at the composition root
    from views_frames_reconcile import ReconciliationResult


@runtime_checkable
class Reconciler(Protocol):
    """Port for hierarchical pgm→cm forecast reconciliation (the DIP abstraction).

    Both methods take a country-level (cm) frame and a grid-level (pgm) frame and scale the
    grid values so they sum to the cm country totals, returning a **new** frame. A point cm
    (`sample_count == 1`) is broadcast across the grid's draws (views-frames ≥1.8). Frames-only
    by design: the geography mapping is held by the concrete implementation, injected at the
    composition root.
    """

    def reconcile(
        self, cm_frame: PredictionFrame, pgm_frame: PredictionFrame
    ) -> PredictionFrame:
        """Scale pgm forecasts to cm country totals; return a new pgm frame.

        Used by the DataFrame ensemble path (`adapter.reconcile_datasets`).
        """
        ...

    def reconcile_result(
        self, cm_frame: PredictionFrame, pgm_frame: PredictionFrame
    ) -> "ReconciliationResult":
        """Reconcile and also report the mode — a `ReconciliationResult` (frame + mode + method).

        The frames-native path (`reconcile_frames`) uses this so the **reconciler** is the
        authority on the mode (`point-broadcast` vs `aligned-draws`), not pipeline-core.
        """
        ...


#: Fail-loud message when reconciliation is configured but no concrete `Reconciler`
#: was injected at the composition root. Single source of truth for both ensemble
#: managers (no silent-off — see #194/#195).
RECONCILER_NOT_INJECTED_MSG = (
    "Reconciliation is configured but no Reconciler was injected. "
    "The composition root (the views-models ensemble main) must inject a concrete "
    "Reconciler. See issue #195."
)
