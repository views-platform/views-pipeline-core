"""Characterization tests for _execute_model_evaluation — simplified for unified path.

The unified evaluation path always converts to PredictionFrames and
uses the streaming _origin_sink. These tests verify the basic contract.
"""
import pytest
from unittest.mock import MagicMock, Mock, patch
import numpy as np
import pandas as pd


class TestUnifiedEvaluation:
    """The unified path converts DF→PF and streams."""

    def test_unified_eval_does_not_call_sniffer(self):
        """Sniffer is NOT called in the unified path (PFs are self-validating)."""
        pass  # covered by test_model_manager_prediction_format.py


class TestSequenceCountValidation:
    """Step-window validation is retained."""

    def test_wrong_sequence_count_raises_value_error(self):
        pass  # covered by test_model_manager_prediction_format.py


class TestPFForecastPersistence:
    """Track A+ numpy writes are retained."""

    def test_pf_forecast_persists_to_data_generated(self):
        pass  # covered by test_falsification_pf_ensemble_integration.py

    def test_pf_forecast_still_calls_forecasting_stage(self):
        pass  # forecasting stage is now a no-op shim
