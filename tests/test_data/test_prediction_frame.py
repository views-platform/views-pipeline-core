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
        assert np.array_equal(pf.y_pred, y_pred)
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
