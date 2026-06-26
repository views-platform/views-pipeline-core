"""The reconciliation domain contract: the `Reconciler` port + its invariants.

Two tightly-coupled pieces of one concept — "what reconciliation must satisfy and
how it is invoked":
  * `Reconciler` — the injection **port** (DIP). Ensemble managers depend on this
    abstraction, never on a concrete reconciliation package. The concrete,
    frames-native reconciler (views-postprocessing) is injected at the composition
    root; pipeline-core never imports it (ADP — no cycle). Geography (the
    `(time, priogrid) -> country` mapping) is baked into the injected instance
    (views-frames ADR-014), so this port is frames-only.
  * `ReconciliationInvariants` — the named mathematical constraints the reconciled
    output must satisfy (sum tolerance, zero preservation, non-negativity).
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from views_frames import PredictionFrame


@runtime_checkable
class Reconciler(Protocol):
    """Port for hierarchical pgm→cm forecast reconciliation (the DIP abstraction).

    `reconcile` takes a country-level (cm) frame and a grid-level (pgm) frame and
    returns a **new** pgm frame whose grid values are scaled so they sum to the cm
    country totals. Frames-only by design: the geography mapping is held by the
    concrete implementation, injected at the composition root.
    """

    def reconcile(
        self, cm_frame: PredictionFrame, pgm_frame: PredictionFrame
    ) -> PredictionFrame:
        """Scale pgm forecasts to cm country totals; return a new pgm frame."""
        ...


#: Fail-loud message when reconciliation is configured but no concrete `Reconciler`
#: was injected at the composition root. Single source of truth for both ensemble
#: managers (no silent-off — see #194/#195).
RECONCILER_NOT_INJECTED_MSG = (
    "Reconciliation 'pgm_cm_point' is configured but no Reconciler was injected. "
    "The composition root (the views-models ensemble main) must inject a concrete "
    "Reconciler. See issue #195."
)


@dataclass(frozen=True)
class ReconciliationInvariants:
    """Mathematical constraints for hierarchical forecast reconciliation.

    Attributes:
        sum_tolerance: Maximum allowed absolute difference between
            PGM aggregate and CM total (default 1e-2).
        zero_atol: Absolute tolerance for treating a value as zero
            (default 1e-8).
        enforce_non_negativity: Whether reconciled values must be >= 0
            (default True).
        preserve_zeros: Whether zero inputs must remain zero after
            reconciliation (default True).
    """

    sum_tolerance: float = 1e-2
    zero_atol: float = 1e-8
    enforce_non_negativity: bool = True
    preserve_zeros: bool = True

    def check_sum_constraint(self, pgm_sum: float, cm_total: float) -> bool:
        """Check if PGM aggregate matches CM total within tolerance."""
        return abs(pgm_sum - cm_total) <= self.sum_tolerance

    def check_zero_preservation(self, original: float, reconciled: float) -> bool:
        """Check that zero inputs remain zero after reconciliation.

        Returns True if the constraint holds or is disabled.
        """
        if not self.preserve_zeros:
            return True
        if abs(original) <= self.zero_atol:
            return abs(reconciled) <= self.zero_atol
        return True
