import os
import logging
import numpy as np
import pandas as pd
from views_evaluation.evaluation.evaluation_manager import EvaluationManager
from views_pipeline_core.managers.ensemble import (
    EnsembleManager, EnsemblePathManager, ModelPathManager, ForecastingModelManager
)
from views_pipeline_core.files.utils import read_dataframe

logger = logging.getLogger(__name__)


class EvalReportGenerator:
    """Generate evaluation reports for ensemble or single model forecasts."""

    def __init__(self, config: dict, target: str, template_dir=None):
        self.config = config
        self.target = target
        self.conflict_type = ForecastingModelManager._get_conflict_type(target)
        self.level = config.get("level")
        self.run_type = config.get("run_type")
        self.template_dir = template_dir or os.path.dirname(os.path.abspath(__file__))
        self.is_ensemble = "models" in config

        if self.is_ensemble:
            self.path_manager = EnsemblePathManager(config["name"])
            self.model_manager = EnsembleManager(self.path_manager)
        else:
            self.path_manager = ModelPathManager(config["name"])
            self.model_manager = ForecastingModelManager(config["name"])

    def generate_eval_report_dict(self, eval_type: str):
        """Return a dictionary with evaluation report data."""
        eval_report = {
            "Target": self.target,
            "Forecast Type": self._forecast_type(),
            "Level of Analysis": self.level,
            "Data Partition": self.run_type,
            "Training Period": self._partition("train"),
            "Testing Period": self._partition("test"),
            "Forecast Horizon": len(self.config.get("steps", [])),
            "Number of Rolling Origins": self.model_manager._resolve_evaluation_sequence_number(eval_type),
            "Evaluation Results": []
        }

        eval_report["Evaluation Results"].append(
            self._single_result(
                "Ensemble" if self.is_ensemble else "Model",
                self.config["name"],
                self._eval_ts(self.path_manager),
                self._pred(self.path_manager)
            )
        )

        if self.is_ensemble:
            for model_name in self.config["models"]:
                pm = ModelPathManager(model_name)
                eval_report["Evaluation Results"].append(
                    self._single_result(
                        "Constituent",
                        model_name,
                        self._eval_ts(pm),
                        self._pred(pm)
                    )
                )
        return eval_report

    def _forecast_type(self):
        df_pred = self._pred(self.path_manager)
        arr = EvaluationManager.convert_to_array(df_pred, f"pred_{self.target}")
        return "point" if not EvaluationManager.get_evaluation_type([arr], f"pred_{self.target}") else "uncertainty"

    def _partition(self, key: str):
        return self.model_manager._partition_dict[self.run_type][key]

    def _eval_ts(self, path_manager):
        path = path_manager._get_eval_file_paths(self.run_type, self.conflict_type)[0]
        return read_dataframe(path)
        

    def _pred(self, path_manager):
        path = path_manager._get_generated_predictions_data_file_paths(self.run_type)[0]
        return read_dataframe(path)

    def _single_result(self, model_type: str, model_name: str, df_eval_ts: pd.DataFrame, df_pred: pd.DataFrame):
        # mse = df_eval_ts.loc["ts00", "MSE"] # Add back after publishing latest version of views-evaluation
        msle = np.sqrt(df_eval_ts.loc["ts00", "RMSLE"])
        mean_pred = df_pred.get(f"pred_{self.target}", [None]).mean()
        
        return {
            "Type": model_type,
            "Model Name": model_name,
            # "MSE": mse,
            "MSLE": msle,
            "mean ŷ": mean_pred
        }


