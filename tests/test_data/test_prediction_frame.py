import pytest
import numpy as np
from views_pipeline_core.data.prediction_frame import PredictionFrame

class TestPredictionFrame:
    def test_initialization_success(self):
        """Verify successful initialization with valid shapes."""
        y_pred = np.random.rand(10, 5) # 10 rows, 5 samples
        identifiers = {
            'time': np.arange(10),
            'unit': np.arange(10)
        }

        pf = PredictionFrame(y_pred=y_pred, identifiers=identifiers)

        assert pf.n_rows == 10
        assert pf.sample_count == 5
        assert pf.y_pred.dtype == np.float32
        np.testing.assert_allclose(pf.y_pred, y_pred.astype(np.float32))
        assert pf.identifier_keys == {'time', 'unit'}

    def test_shape_mismatch_raises(self):
        """Verify failure when y_pred and identifiers have different row counts."""
        y_pred = np.random.rand(10, 5)
        identifiers = {
            'time': np.arange(9), # Mismatch: 9 vs 10
            'unit': np.arange(10)
        }
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            PredictionFrame(y_pred=y_pred, identifiers=identifiers)

    def test_non_2d_y_pred_raises(self):
        """Verify failure when y_pred is not 2D."""
        y_pred = np.random.rand(10) # 1D array
        identifiers = {'time': np.arange(10)}
        
        with pytest.raises(ValueError, match="y_pred must be a 2D array"):
            PredictionFrame(y_pred=y_pred, identifiers=identifiers)

    def test_missing_required_identifiers_raises(self):
        """Verify failure when required identifiers are missing."""
        y_pred = np.random.rand(10, 1)
        identifiers = {'something_else': np.arange(10)} # missing time/unit
        
        with pytest.raises(ValueError, match="Missing required identifier"):
            PredictionFrame(y_pred=y_pred, identifiers=identifiers)

    def test_nan_in_identifiers_raises(self):
        """Verify failure when identifiers contain NaNs."""
        y_pred = np.random.rand(2, 1)
        identifiers = {
            'time': np.array([100, np.nan]),
            'unit': np.array([1, 2])
        }

        with pytest.raises(ValueError, match="NaN detected in identifier"):
            PredictionFrame(y_pred=y_pred, identifiers=identifiers)

    def test_zero_rows_raises(self):
        """A frame with shape (0, S) is not a valid prediction."""
        y_pred = np.empty((0, 1))
        identifiers = {'time': np.array([]), 'unit': np.array([])}
        with pytest.raises(ValueError, match="at least one row"):
            PredictionFrame(y_pred=y_pred, identifiers=identifiers)

    def test_zero_samples_raises(self):
        """A frame with shape (N, 0) is not a valid prediction."""
        y_pred = np.empty((3, 0))
        identifiers = {'time': np.arange(3), 'unit': np.arange(3)}
        with pytest.raises(ValueError, match="at least one sample column"):
            PredictionFrame(y_pred=y_pred, identifiers=identifiers)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pf(n_rows: int = 4, n_samples: int = 8, value: float = None) -> PredictionFrame:
    """Build a minimal PredictionFrame for testing. Value=None → random."""
    if value is not None:
        y_pred = np.full((n_rows, n_samples), value, dtype=float)
    else:
        rng = np.random.default_rng(42)
        y_pred = rng.standard_normal((n_rows, n_samples))
    return PredictionFrame(
        y_pred=y_pred,
        identifiers={
            "time": np.arange(n_rows, dtype=int),
            "unit": np.arange(n_rows, dtype=int) + 100,
        },
    )


# ---------------------------------------------------------------------------
# TestPredictionFrameStochastic — GREEN tier
# ---------------------------------------------------------------------------

class TestPredictionFrameStochastic:
    """Verify that a PredictionFrame holding full posterior samples works as intended."""

    def test_stochastic_sample_count_equals_s(self):
        """PF with S=100 samples: pf.sample_count == 100."""
        pf = _make_pf(n_rows=10, n_samples=100)
        assert pf.sample_count == 100

    def test_stochastic_preserves_all_sample_values(self):
        """y_pred values are numerically identical after storage — no accidental reduction."""
        y_pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pf = PredictionFrame(
            y_pred=y_pred,
            identifiers={"time": np.array([0, 1]), "unit": np.array([10, 11])},
        )
        assert np.array_equal(pf.y_pred, y_pred)

    def test_stochastic_y_pred_columns_independently_accessible(self):
        """Each column (posterior draw) is addressable: pf.y_pred[:, i] returns correct slice."""
        n_rows, n_samples = 5, 4
        y_pred = np.arange(n_rows * n_samples, dtype=float).reshape(n_rows, n_samples)
        pf = PredictionFrame(
            y_pred=y_pred,
            identifiers={"time": np.arange(n_rows), "unit": np.arange(n_rows)},
        )
        for i in range(n_samples):
            assert np.array_equal(pf.y_pred[:, i], y_pred[:, i])

    def test_stochastic_single_sample_is_valid(self):
        """S=1 is a legal stochastic PF. sample_count == 1, not the same as 'collapsed'."""
        pf = _make_pf(n_rows=6, n_samples=1)
        assert pf.sample_count == 1
        assert pf.y_pred.shape == (6, 1)

    def test_stochastic_large_sample_count_accepted(self):
        """S=1000, N=500 constructs without error — no artificial upper bound enforced."""
        pf = _make_pf(n_rows=500, n_samples=1000)
        assert pf.sample_count == 1000
        assert pf.n_rows == 500

    def test_stochastic_identifiers_preserved_unchanged(self):
        """Identifiers on a stochastic PF are bitwise-equal to what was passed in."""
        time_arr = np.array([100, 200, 300])
        unit_arr = np.array([1, 2, 3])
        pf = PredictionFrame(
            y_pred=np.ones((3, 5)),
            identifiers={"time": time_arr, "unit": unit_arr},
        )
        assert np.array_equal(pf.identifiers["time"], time_arr)
        assert np.array_equal(pf.identifiers["unit"], unit_arr)


# ---------------------------------------------------------------------------
# TestPredictionFrameCollapse — GREEN tier
# ---------------------------------------------------------------------------

class TestPredictionFrameCollapse:
    """Core correctness tests for PredictionFrame.collapse()."""

    def test_collapse_arithmetic_mean_shape(self):
        """collapse('arithmetic_mean') produces y_pred of shape (N, 1)."""
        pf = _make_pf(n_rows=4, n_samples=10)
        collapsed = pf.collapse("arithmetic_mean")
        assert collapsed.y_pred.shape == (4, 1)

    def test_collapse_arithmetic_mean_values_correct(self):
        """Collapsed values equal np.mean(y_pred, axis=1, keepdims=True) exactly."""
        pf = _make_pf(n_rows=4, n_samples=8)
        expected = pf.y_pred.mean(axis=1, keepdims=True)
        collapsed = pf.collapse("arithmetic_mean")
        np.testing.assert_array_equal(collapsed.y_pred, expected)

    def test_collapse_known_values(self):
        """Hand-verifiable: [[1,3],[2,4]] → [[2.0],[3.0]]."""
        y_pred = np.array([[1.0, 3.0], [2.0, 4.0]])
        pf = PredictionFrame(
            y_pred=y_pred,
            identifiers={"time": np.array([0, 1]), "unit": np.array([10, 11])},
        )
        collapsed = pf.collapse("arithmetic_mean")
        np.testing.assert_array_equal(collapsed.y_pred, np.array([[2.0], [3.0]]))

    def test_collapse_preserves_all_identifiers(self):
        """Every key/value in identifiers is present and numerically equal after collapse."""
        pf = _make_pf(n_rows=4, n_samples=5)
        collapsed = pf.collapse("arithmetic_mean")
        assert collapsed.identifier_keys == pf.identifier_keys
        for key in pf.identifier_keys:
            np.testing.assert_array_equal(collapsed.identifiers[key], pf.identifiers[key])

    def test_collapse_returns_valid_pf(self):
        """The returned object is a PredictionFrame instance with sample_count == 1."""
        pf = _make_pf(n_rows=4, n_samples=6)
        collapsed = pf.collapse("arithmetic_mean")
        assert isinstance(collapsed, PredictionFrame)
        assert collapsed.sample_count == 1
        assert collapsed.n_rows == pf.n_rows

    def test_collapse_is_immutable_original_unchanged(self):
        """After collapse(), original pf.y_pred is unchanged in values and shape."""
        pf = _make_pf(n_rows=4, n_samples=8)
        original_shape = pf.y_pred.shape
        original_values = pf.y_pred.copy()
        pf.collapse("arithmetic_mean")
        assert pf.y_pred.shape == original_shape
        np.testing.assert_array_equal(pf.y_pred, original_values)

    def test_collapse_s1_pf_produces_same_values(self):
        """Collapsing a PF with S=1 returns the same y_pred values (mean of one element)."""
        y_pred = np.array([[5.0], [3.0], [7.0]])
        pf = PredictionFrame(
            y_pred=y_pred,
            identifiers={"time": np.arange(3), "unit": np.arange(3)},
        )
        collapsed = pf.collapse("arithmetic_mean")
        np.testing.assert_array_equal(collapsed.y_pred, y_pred)

    def test_collapse_returns_new_object(self):
        """collapse() returns a distinct object — not the same instance."""
        pf = _make_pf()
        collapsed = pf.collapse("arithmetic_mean")
        assert collapsed is not pf


# ---------------------------------------------------------------------------
# TestPredictionFrameCollapseBeige — BEIGE tier (edge / boundary)
# ---------------------------------------------------------------------------

class TestPredictionFrameCollapseBeige:
    """Edge cases and boundary conditions for collapse()."""

    def test_collapse_single_row_pf(self):
        """N=1, S=50 → collapsed shape (1, 1). No crash for single observation."""
        pf = _make_pf(n_rows=1, n_samples=50)
        collapsed = pf.collapse("arithmetic_mean")
        assert collapsed.y_pred.shape == (1, 1)

    def test_collapse_large_s_stable_mean(self):
        """S=10000 values from N(0,1). Collapsed mean is close to 0 (no blowup)."""
        rng = np.random.default_rng(0)
        y_pred = rng.standard_normal((5, 10_000))
        pf = PredictionFrame(
            y_pred=y_pred,
            identifiers={"time": np.arange(5), "unit": np.arange(5)},
        )
        collapsed = pf.collapse("arithmetic_mean")
        # By CLT, sample mean of 10000 draws is within 3σ/√n ≈ 0.03 of 0
        assert np.all(np.abs(collapsed.y_pred) < 0.1)

    def test_collapse_negative_values(self):
        """y_pred with all-negative values: mean is also negative."""
        y_pred = np.full((3, 4), -5.0)
        pf = PredictionFrame(
            y_pred=y_pred,
            identifiers={"time": np.arange(3), "unit": np.arange(3)},
        )
        collapsed = pf.collapse("arithmetic_mean")
        np.testing.assert_array_equal(collapsed.y_pred, np.full((3, 1), -5.0))

    def test_collapse_all_identical_values(self):
        """Mean of a constant is that constant: all 7.5 → all 7.5."""
        pf = _make_pf(n_rows=3, n_samples=10, value=7.5)
        collapsed = pf.collapse("arithmetic_mean")
        np.testing.assert_array_almost_equal(collapsed.y_pred, np.full((3, 1), 7.5))

    def test_collapse_mixed_positive_negative_sums_to_zero(self):
        """Symmetric rows (e.g. [-1, 1]) collapse to 0."""
        y_pred = np.tile([-1.0, 1.0], (4, 1))  # shape (4, 2)
        pf = PredictionFrame(
            y_pred=y_pred,
            identifiers={"time": np.arange(4), "unit": np.arange(4)},
        )
        collapsed = pf.collapse("arithmetic_mean")
        np.testing.assert_array_almost_equal(collapsed.y_pred, np.zeros((4, 1)))

    def test_collapse_float32_input_accepted(self):
        """float32 y_pred does not crash — dtype is float32 or float64 after collapse."""
        y_pred = np.ones((3, 4), dtype=np.float32)
        pf = PredictionFrame(
            y_pred=y_pred,
            identifiers={"time": np.arange(3), "unit": np.arange(3)},
        )
        collapsed = pf.collapse("arithmetic_mean")
        assert collapsed.y_pred.dtype in (np.float32, np.float64)
        assert collapsed.y_pred.shape == (3, 1)

    def test_stochastic_and_collapsed_share_same_identifier_values(self):
        """Stochastic PF and its collapsed counterpart have bitwise-identical identifiers."""
        pf = _make_pf(n_rows=5, n_samples=20)
        collapsed = pf.collapse("arithmetic_mean")
        for key in pf.identifier_keys:
            np.testing.assert_array_equal(collapsed.identifiers[key], pf.identifiers[key])

    def test_collapse_nan_in_y_pred_propagates_to_mean(self):
        """NaN in any sample propagates to collapsed mean. PF itself accepts NaN in y_pred."""
        y_pred = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 6.0]])
        pf = PredictionFrame(
            y_pred=y_pred,
            identifiers={"time": np.array([0, 1]), "unit": np.array([10, 11])},
        )
        collapsed = pf.collapse("arithmetic_mean")
        assert np.isnan(collapsed.y_pred[0, 0])          # NaN propagated
        assert not np.isnan(collapsed.y_pred[1, 0])      # Clean row unaffected

    def test_collapse_inf_in_y_pred_propagates_to_mean(self):
        """Inf in any sample propagates to collapsed mean."""
        y_pred = np.array([[1.0, np.inf], [2.0, 4.0]])
        pf = PredictionFrame(
            y_pred=y_pred,
            identifiers={"time": np.array([0, 1]), "unit": np.array([10, 11])},
        )
        collapsed = pf.collapse("arithmetic_mean")
        assert np.isinf(collapsed.y_pred[0, 0])
        assert not np.isinf(collapsed.y_pred[1, 0])


# ---------------------------------------------------------------------------
# TestPredictionFrameCollapseRedTeam — RED TEAM tier (adversarial / failure)
# ---------------------------------------------------------------------------

class TestPredictionFrameCollapseRedTeam:
    """Adversarial and failure-mode coverage for collapse()."""

    def test_collapse_unknown_method_raises_value_error(self):
        """Unsupported aggregate_method raises ValueError mentioning the method name."""
        pf = _make_pf()
        with pytest.raises(ValueError, match="geometric_mean"):
            pf.collapse("geometric_mean")

    def test_collapse_empty_string_method_raises(self):
        """Empty string is not a supported method."""
        pf = _make_pf()
        with pytest.raises(ValueError):
            pf.collapse("")

    def test_collapse_none_method_raises(self):
        """None is not a valid method argument."""
        pf = _make_pf()
        with pytest.raises((ValueError, TypeError)):
            pf.collapse(None)

    def test_collapse_does_not_mutate_original_y_pred(self):
        """collapse() must not alter the original y_pred array in place."""
        y_pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pf = PredictionFrame(
            y_pred=y_pred.copy(),
            identifiers={"time": np.array([0, 1]), "unit": np.array([10, 11])},
        )
        values_before = pf.y_pred.copy()
        pf.collapse("arithmetic_mean")
        np.testing.assert_array_equal(pf.y_pred, values_before)

    def test_collapse_identifier_arrays_are_deep_copies(self):
        """Collapsed PF's identifier arrays are not the same objects as originals."""
        pf = _make_pf()
        collapsed = pf.collapse("arithmetic_mean")
        assert collapsed.identifiers["time"] is not pf.identifiers["time"]
        assert collapsed.identifiers["unit"] is not pf.identifiers["unit"]

    def test_stochastic_nan_in_y_pred_does_not_raise(self):
        """NaN in y_pred values is NOT a construction error — PF is a transport container."""
        y_pred = np.array([[np.nan, 1.0], [2.0, 3.0]])
        # Must not raise:
        pf = PredictionFrame(
            y_pred=y_pred,
            identifiers={"time": np.array([0, 1]), "unit": np.array([10, 11])},
        )
        assert pf.sample_count == 2

    def test_stochastic_inf_in_y_pred_does_not_raise(self):
        """Inf in y_pred values is NOT a construction error — only identifiers are checked."""
        y_pred = np.array([[np.inf, 1.0], [2.0, -np.inf]])
        pf = PredictionFrame(
            y_pred=y_pred,
            identifiers={"time": np.array([0, 1]), "unit": np.array([10, 11])},
        )
        assert pf.n_rows == 2

    def test_collapse_called_twice_is_idempotent_on_values(self):
        """Collapsing an already-collapsed PF (S=1) a second time preserves values."""
        pf = _make_pf(n_rows=3, n_samples=10)
        collapsed_once = pf.collapse("arithmetic_mean")     # (3, 1)
        collapsed_twice = collapsed_once.collapse("arithmetic_mean")  # still (3, 1)
        assert collapsed_twice.y_pred.shape == (3, 1)
        np.testing.assert_array_equal(collapsed_twice.y_pred, collapsed_once.y_pred)

    def test_collapse_result_passes_full_pf_validation(self):
        """The result of collapse() satisfies all PredictionFrame invariants."""
        pf = _make_pf(n_rows=5, n_samples=20)
        collapsed = pf.collapse("arithmetic_mean")
        # Verify invariants explicitly:
        assert collapsed.y_pred.ndim == 2
        assert collapsed.n_rows == 5
        assert collapsed.sample_count == 1
        assert PredictionFrame.REQUIRED_IDENTIFIERS <= collapsed.identifier_keys
        for key, arr in collapsed.identifiers.items():
            assert len(arr) == collapsed.n_rows
