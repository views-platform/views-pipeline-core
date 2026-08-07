"""pipeline-core PredictionFrame tests after issue #188.

The local PredictionFrame class was retired; ``views_pipeline_core.data.prediction_frame.PredictionFrame``
now re-exports the published leaf ``views_frames.PredictionFrame``. The leaf owns
its own construction / validation / ``__repr__`` / type tests, so this module is
trimmed to what is still pipeline-core's concern:

  * the re-export identity (pipeline-core's symbol IS the leaf type);
  * sample-preservation behaviour that the pipeline relies on (a PF is a
    transport container holding the full posterior — no accidental reduction);
  * sample-axis reduction, which moved off the class onto
    ``views_frames_summarize.collapse`` (the old ``collapse("arithmetic_mean")``
    method == ``collapse(pf, np.mean)``).
"""
import numpy as np

import views_frames
from views_frames import SpatialLevel, SpatioTemporalIndex
from views_frames_summarize import collapse

from views_pipeline_core.data.prediction_frame import PredictionFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pf(n_rows: int = 4, n_samples: int = 8, value: float = None) -> PredictionFrame:
    """Build a minimal leaf PredictionFrame for testing. value=None → random."""
    if value is not None:
        y_pred = np.full((n_rows, n_samples), value, dtype=float)
    else:
        rng = np.random.default_rng(42)
        y_pred = rng.standard_normal((n_rows, n_samples))
    index = SpatioTemporalIndex(
        time=np.arange(n_rows, dtype=np.int64),
        unit=np.arange(n_rows, dtype=np.int64) + 100,
        level=SpatialLevel.PGM,
    )
    return PredictionFrame(y_pred, index)


def _pf(y_pred, time, unit):
    """Construct a leaf PredictionFrame from raw arrays."""
    index = SpatioTemporalIndex(
        time=np.asarray(time, dtype=np.int64),
        unit=np.asarray(unit, dtype=np.int64),
        level=SpatialLevel.PGM,
    )
    return PredictionFrame(y_pred, index)


# ---------------------------------------------------------------------------
# Re-export identity — pipeline-core's responsibility post-#188
# ---------------------------------------------------------------------------

class TestPredictionFrameReExport:
    def test_pipeline_core_symbol_is_the_leaf(self):
        """``data.prediction_frame.PredictionFrame`` IS ``views_frames.PredictionFrame``."""
        assert PredictionFrame is views_frames.PredictionFrame


# ---------------------------------------------------------------------------
# TestPredictionFrameStochastic — sample-preservation transport contract
# ---------------------------------------------------------------------------

class TestPredictionFrameStochastic:
    """Verify that a PredictionFrame holding full posterior samples works as intended."""

    def test_stochastic_sample_count_equals_s(self):
        """PF with S=100 samples: pf.sample_count == 100."""
        pf = _make_pf(n_rows=10, n_samples=100)
        assert pf.sample_count == 100

    def test_stochastic_preserves_all_sample_values(self):
        """values are numerically identical after storage — no accidental reduction."""
        y_pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        pf = _pf(y_pred, time=[0, 1], unit=[10, 11])
        assert np.array_equal(pf.values, y_pred)

    def test_stochastic_values_columns_independently_accessible(self):
        """Each column (posterior draw) is addressable: pf.values[:, i] returns correct slice."""
        n_rows, n_samples = 5, 4
        y_pred = np.arange(n_rows * n_samples, dtype=np.float32).reshape(n_rows, n_samples)
        pf = _pf(y_pred, time=np.arange(n_rows), unit=np.arange(n_rows))
        for i in range(n_samples):
            assert np.array_equal(pf.values[:, i], y_pred[:, i])

    def test_stochastic_single_sample_is_valid(self):
        """S=1 is a legal stochastic PF. sample_count == 1, not the same as 'collapsed'."""
        pf = _make_pf(n_rows=6, n_samples=1)
        assert pf.sample_count == 1
        assert pf.values.shape == (6, 1)

    def test_stochastic_large_sample_count_accepted(self):
        """S=1000, N=500 constructs without error — no artificial upper bound enforced."""
        pf = _make_pf(n_rows=500, n_samples=1000)
        assert pf.sample_count == 1000
        assert pf.n_rows == 500

    def test_stochastic_identifiers_preserved_unchanged(self):
        """Identifiers on a stochastic PF are equal to what was passed in."""
        time_arr = np.array([100, 200, 300], dtype=np.int64)
        unit_arr = np.array([1, 2, 3], dtype=np.int64)
        pf = _pf(np.ones((3, 5)), time=time_arr, unit=unit_arr)
        assert np.array_equal(pf.identifiers["time"], time_arr)
        assert np.array_equal(pf.identifiers["unit"], unit_arr)


# ---------------------------------------------------------------------------
# TestPredictionFrameCollapse — sample-axis reduction via views_frames_summarize
# ---------------------------------------------------------------------------

class TestPredictionFrameCollapse:
    """Core correctness tests for ``collapse(pf, np.mean)`` (the old arithmetic_mean)."""

    def test_collapse_arithmetic_mean_shape(self):
        """collapse produces values of shape (N, 1)."""
        pf = _make_pf(n_rows=4, n_samples=10)
        collapsed = collapse(pf, np.mean)
        assert collapsed.values.shape == (4, 1)

    def test_collapse_arithmetic_mean_values_correct(self):
        """Collapsed values equal np.mean(values, axis=1, keepdims=True) exactly."""
        pf = _make_pf(n_rows=4, n_samples=8)
        expected = pf.values.mean(axis=1, keepdims=True)
        collapsed = collapse(pf, np.mean)
        np.testing.assert_array_equal(collapsed.values, expected)

    def test_collapse_known_values(self):
        """Hand-verifiable: [[1,3],[2,4]] → [[2.0],[3.0]]."""
        y_pred = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        pf = _pf(y_pred, time=[0, 1], unit=[10, 11])
        collapsed = collapse(pf, np.mean)
        np.testing.assert_array_equal(collapsed.values, np.array([[2.0], [3.0]]))

    def test_collapse_preserves_all_identifiers(self):
        """Every key/value in identifiers is present and numerically equal after collapse."""
        pf = _make_pf(n_rows=4, n_samples=5)
        collapsed = collapse(pf, np.mean)
        assert set(collapsed.identifiers) == set(pf.identifiers)
        for key in pf.identifiers:
            np.testing.assert_array_equal(collapsed.identifiers[key], pf.identifiers[key])

    def test_collapse_returns_valid_pf(self):
        """The returned object is a PredictionFrame instance with sample_count == 1."""
        pf = _make_pf(n_rows=4, n_samples=6)
        collapsed = collapse(pf, np.mean)
        assert isinstance(collapsed, PredictionFrame)
        assert collapsed.sample_count == 1
        assert collapsed.n_rows == pf.n_rows

    def test_collapse_is_immutable_original_unchanged(self):
        """After collapse, the original pf.values is unchanged in values and shape."""
        pf = _make_pf(n_rows=4, n_samples=8)
        original_shape = pf.values.shape
        original_values = pf.values.copy()
        collapse(pf, np.mean)
        assert pf.values.shape == original_shape
        np.testing.assert_array_equal(pf.values, original_values)

    def test_collapse_s1_pf_produces_same_values(self):
        """Collapsing a PF with S=1 returns the same values (mean of one element)."""
        y_pred = np.array([[5.0], [3.0], [7.0]], dtype=np.float32)
        pf = _pf(y_pred, time=np.arange(3), unit=np.arange(3))
        collapsed = collapse(pf, np.mean)
        np.testing.assert_array_equal(collapsed.values, y_pred)

    def test_collapse_returns_new_object(self):
        """collapse returns a distinct object — not the same instance."""
        pf = _make_pf()
        collapsed = collapse(pf, np.mean)
        assert collapsed is not pf


# ---------------------------------------------------------------------------
# TestPredictionFrameCollapseBeige — edge / boundary
# ---------------------------------------------------------------------------

class TestPredictionFrameCollapseBeige:
    """Edge cases and boundary conditions for collapse."""

    def test_collapse_single_row_pf(self):
        """N=1, S=50 → collapsed shape (1, 1). No crash for single observation."""
        pf = _make_pf(n_rows=1, n_samples=50)
        collapsed = collapse(pf, np.mean)
        assert collapsed.values.shape == (1, 1)

    def test_collapse_large_s_stable_mean(self):
        """S=10000 values from N(0,1). Collapsed mean is close to 0 (no blowup)."""
        rng = np.random.default_rng(0)
        y_pred = rng.standard_normal((5, 10_000))
        pf = _pf(y_pred, time=np.arange(5), unit=np.arange(5))
        collapsed = collapse(pf, np.mean)
        # By CLT, sample mean of 10000 draws is within 3σ/√n ≈ 0.03 of 0
        assert np.all(np.abs(collapsed.values) < 0.1)

    def test_collapse_negative_values(self):
        """values with all-negative entries: mean is also negative."""
        y_pred = np.full((3, 4), -5.0)
        pf = _pf(y_pred, time=np.arange(3), unit=np.arange(3))
        collapsed = collapse(pf, np.mean)
        np.testing.assert_array_equal(collapsed.values, np.full((3, 1), -5.0))

    def test_collapse_all_identical_values(self):
        """Mean of a constant is that constant: all 7.5 → all 7.5."""
        pf = _make_pf(n_rows=3, n_samples=10, value=7.5)
        collapsed = collapse(pf, np.mean)
        np.testing.assert_array_almost_equal(collapsed.values, np.full((3, 1), 7.5))

    def test_collapse_mixed_positive_negative_sums_to_zero(self):
        """Symmetric rows (e.g. [-1, 1]) collapse to 0."""
        y_pred = np.tile([-1.0, 1.0], (4, 1))  # shape (4, 2)
        pf = _pf(y_pred, time=np.arange(4), unit=np.arange(4))
        collapsed = collapse(pf, np.mean)
        np.testing.assert_array_almost_equal(collapsed.values, np.zeros((4, 1)))

    def test_collapse_float32_input_accepted(self):
        """float32 values does not crash — dtype is float32 or float64 after collapse."""
        y_pred = np.ones((3, 4), dtype=np.float32)
        pf = _pf(y_pred, time=np.arange(3), unit=np.arange(3))
        collapsed = collapse(pf, np.mean)
        assert collapsed.values.dtype in (np.float32, np.float64)
        assert collapsed.values.shape == (3, 1)

    def test_stochastic_and_collapsed_share_same_identifier_values(self):
        """Stochastic PF and its collapsed counterpart have identical identifiers."""
        pf = _make_pf(n_rows=5, n_samples=20)
        collapsed = collapse(pf, np.mean)
        for key in pf.identifiers:
            np.testing.assert_array_equal(collapsed.identifiers[key], pf.identifiers[key])

    def test_collapse_nan_in_values_propagates_to_mean(self):
        """NaN in any sample propagates to collapsed mean. PF itself accepts NaN in values."""
        y_pred = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32)
        pf = _pf(y_pred, time=[0, 1], unit=[10, 11])
        collapsed = collapse(pf, np.mean)
        assert np.isnan(collapsed.values[0, 0])          # NaN propagated
        assert not np.isnan(collapsed.values[1, 0])      # Clean row unaffected

    def test_collapse_inf_in_values_propagates_to_mean(self):
        """Inf in any sample propagates to collapsed mean."""
        y_pred = np.array([[1.0, np.inf], [2.0, 4.0]], dtype=np.float32)
        pf = _pf(y_pred, time=[0, 1], unit=[10, 11])
        collapsed = collapse(pf, np.mean)
        assert np.isinf(collapsed.values[0, 0])
        assert not np.isinf(collapsed.values[1, 0])


# ---------------------------------------------------------------------------
# TestPredictionFrameCollapseRedTeam — adversarial / non-mutation
# ---------------------------------------------------------------------------

class TestPredictionFrameCollapseRedTeam:
    """Non-mutation and value-tolerance coverage for collapse / construction."""

    def test_collapse_does_not_mutate_original_values(self):
        """collapse must not alter the original values array in place."""
        y_pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        pf = _pf(y_pred.copy(), time=[0, 1], unit=[10, 11])
        values_before = pf.values.copy()
        collapse(pf, np.mean)
        np.testing.assert_array_equal(pf.values, values_before)

    def test_stochastic_nan_in_values_does_not_raise(self):
        """NaN in values is NOT a construction error — PF is a transport container."""
        y_pred = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
        pf = _pf(y_pred, time=[0, 1], unit=[10, 11])  # must not raise
        assert pf.sample_count == 2

    def test_stochastic_inf_in_values_does_not_raise(self):
        """Inf in values is NOT a construction error — only identifiers are checked."""
        y_pred = np.array([[np.inf, 1.0], [2.0, -np.inf]], dtype=np.float32)
        pf = _pf(y_pred, time=[0, 1], unit=[10, 11])
        assert pf.n_rows == 2

    def test_collapse_called_twice_is_idempotent_on_values(self):
        """Collapsing an already-collapsed PF (S=1) a second time preserves values."""
        pf = _make_pf(n_rows=3, n_samples=10)
        collapsed_once = collapse(pf, np.mean)              # (3, 1)
        collapsed_twice = collapse(collapsed_once, np.mean)  # still (3, 1)
        assert collapsed_twice.values.shape == (3, 1)
        np.testing.assert_array_equal(collapsed_twice.values, collapsed_once.values)