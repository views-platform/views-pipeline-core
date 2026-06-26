"""Tests for ReconciliationInvariants domain value object."""
from dataclasses import FrozenInstanceError

import pytest

from views_pipeline_core.domain.reconciliation_invariants import ReconciliationInvariants


class TestDefaults:
    def test_default_values(self):
        r = ReconciliationInvariants()
        assert r.sum_tolerance == 1e-2
        assert r.zero_atol == 1e-8
        assert r.enforce_non_negativity is True
        assert r.preserve_zeros is True

    def test_frozen(self):
        r = ReconciliationInvariants()
        with pytest.raises(FrozenInstanceError):
            r.sum_tolerance = 0.5


class TestSumConstraint:
    def test_within_tolerance(self):
        r = ReconciliationInvariants()
        assert r.check_sum_constraint(pgm_sum=100.005, cm_total=100.0) is True

    def test_outside_tolerance(self):
        r = ReconciliationInvariants()
        assert r.check_sum_constraint(pgm_sum=100.02, cm_total=100.0) is False

    def test_exact_match(self):
        r = ReconciliationInvariants()
        assert r.check_sum_constraint(pgm_sum=50.0, cm_total=50.0) is True

    def test_custom_tolerance(self):
        r = ReconciliationInvariants(sum_tolerance=0.1)
        assert r.check_sum_constraint(pgm_sum=100.05, cm_total=100.0) is True


class TestZeroPreservation:
    def test_zero_stays_zero(self):
        r = ReconciliationInvariants()
        assert r.check_zero_preservation(original=0.0, reconciled=0.0) is True

    def test_zero_becomes_nonzero_fails(self):
        r = ReconciliationInvariants()
        assert r.check_zero_preservation(original=0.0, reconciled=0.1) is False

    def test_nonzero_unchecked(self):
        r = ReconciliationInvariants()
        assert r.check_zero_preservation(original=5.0, reconciled=6.0) is True

    def test_disabled(self):
        r = ReconciliationInvariants(preserve_zeros=False)
        assert r.check_zero_preservation(original=0.0, reconciled=0.1) is True

    def test_near_zero_treated_as_zero(self):
        r = ReconciliationInvariants(zero_atol=1e-6)
        assert r.check_zero_preservation(original=1e-7, reconciled=1e-7) is True
