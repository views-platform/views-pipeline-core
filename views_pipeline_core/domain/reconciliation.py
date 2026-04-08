"""Reconciliation invariants domain value object.

Named mathematical constraints for hierarchical PGM-CM forecast
reconciliation.  Makes implicit tolerance values and constraint
flags explicit and testable.
"""
from dataclasses import dataclass


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
