"""Tests for PredictionFileNamer — canonical filename generation."""
from views_pipeline_core.managers.prediction.file_namer import PredictionFileNamer


class TestPredictionName:
    def test_with_sequence_number(self):
        namer = PredictionFileNamer("calibration", "20260407", ".parquet")
        result = namer.prediction_name(sequence_number=3)
        assert result == "predictions_calibration_20260407_03.parquet"

    def test_without_sequence_number(self):
        namer = PredictionFileNamer("forecasting", "20260407", ".parquet")
        result = namer.prediction_name()
        assert result == "predictions_forecasting_20260407.parquet"

    def test_sequence_zero_is_zero_padded(self):
        namer = PredictionFileNamer("calibration", "20260407", ".parquet")
        result = namer.prediction_name(sequence_number=0)
        assert result == "predictions_calibration_20260407_00.parquet"

    def test_custom_extension(self):
        namer = PredictionFileNamer("calibration", "20260407", ".pkl")
        result = namer.prediction_name()
        assert result == "predictions_calibration_20260407.pkl"


class TestEvaluationName:
    def test_step_evaluation(self):
        namer = PredictionFileNamer("calibration", "20260407", ".parquet")
        result = namer.evaluation_name("step", "ged_sb_best")
        assert result == "eval_calibration_ged_sb_best_step_20260407.parquet"

    def test_month_evaluation(self):
        namer = PredictionFileNamer("validation", "20260401", ".parquet")
        result = namer.evaluation_name("month", "lr_sb")
        assert result == "eval_validation_lr_sb_month_20260401.parquet"

    def test_ts_evaluation(self):
        namer = PredictionFileNamer("calibration", "20260407", ".parquet")
        result = namer.evaluation_name("ts", "ged_sb_best")
        assert result == "eval_calibration_ged_sb_best_ts_20260407.parquet"


class TestPredictionNameWithTarget:
    def test_with_target_no_seq(self):
        namer = PredictionFileNamer("forecasting", "20260407", ".parquet")
        result = namer.prediction_name(target_identifier="ged_sb_best")
        assert result == "predictions_forecasting_20260407_ged_sb_best.parquet"

    def test_with_target_and_seq(self):
        namer = PredictionFileNamer("calibration", "20260407", ".parquet")
        result = namer.prediction_name(sequence_number=3, target_identifier="ged_sb_best")
        assert result == "predictions_calibration_20260407_ged_sb_best_03.parquet"

    def test_without_target_backward_compatible(self):
        namer = PredictionFileNamer("forecasting", "20260407", ".parquet")
        result = namer.prediction_name()
        assert result == "predictions_forecasting_20260407.parquet"

    def test_two_targets_unique_filenames(self):
        namer = PredictionFileNamer("forecasting", "20260407", ".parquet")
        name_a = namer.prediction_name(target_identifier="ged_sb_best")
        name_b = namer.prediction_name(target_identifier="acled_sb_count")
        assert name_a != name_b


class TestDefaults:
    def test_default_extension_is_parquet(self):
        namer = PredictionFileNamer("calibration", "20260407")
        result = namer.prediction_name()
        assert result.endswith(".parquet")