import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader
from views_evaluation.evaluation.evaluation_manager import EvaluationManager
from views_evaluation.evaluation.metric_calculators import (
    POINT_METRIC_FUNCTIONS,
    UNCERTAINTY_METRIC_FUNCTIONS,
)
from views_pipeline_core.managers.ensemble import (
    EnsemblePathManager,
    ModelPathManager,
)
from views_pipeline_core.files.utils import read_dataframe
from argparse import Namespace
import os
import logging

logger = logging.getLogger(__name__)

class AnalysisReportGenerator:
    def __init__(self, config, template_dir=None):
        self._args = Namespace(run_type="calibration")
        self._required_metrics = ["RMSLE", "CRPS"]  # TODO: handle AUPR and Brier score
        self.config = config
        self.targets = config["targets"]
        self.ensemble_path = EnsemblePathManager(config["name"])
        self.template_dir = template_dir or os.path.dirname(os.path.abspath(__file__))
        self.actual, self.prediction = self._load_data()
        self.is_uncertainty = EvaluationManager.get_evaluation_type([self.prediction])
        self.functions = (
            POINT_METRIC_FUNCTIONS
            if not self.is_uncertainty
            else UNCERTAINTY_METRIC_FUNCTIONS
        )

    def _load_data(self):
        actual = read_dataframe(
            file_path=ModelPathManager(
                self.config["models"][0]
            )._get_raw_data_file_paths(run_type=self._args.run_type)[0]
        )
        prediction = read_dataframe(
            self.ensemble_path._get_generated_predictions_data_file_paths(
                run_type=self._args.run_type
            )[0]
        )
        actual = self._transform(actual, self.targets)
        prediction = self._transform(prediction, [f"pred_{t}" for t in self.targets])
        return actual, prediction

    def _transform(self, data, col):
        return EvaluationManager.transform_data(
            EvaluationManager.convert_to_array(data, col), col
        )

    def get_model_config_link(self):
        return f"https://github.com/views-platform/views-models/tree/main/ensembles/{self.config['name']}/configs"

    def prepare_metadata(self):
        return {
            "name": self.config["name"],
            "level": self.config["level"],
            "targets": self.config["targets"],
            "aggregation": self.config["aggregation"],
            "constituent_models": self.config.get("models", []),
            "reconciliation": self.config.get("reconciliation", None),
            "detailed_model_config": self.get_model_config_link(),
        }

    def prepare_worst_predictions(self, target):
        matched_actual, matched_pred = EvaluationManager._match_actual_pred(
            self.actual, self.prediction, target
        )
        worst = {}
        for metric in self.config["metrics"]:
            if metric not in self._required_metrics:
                continue
            metric_func = self.functions[metric]
            actual_col = matched_actual[target]
            pred_col = matched_pred[f"pred_{target}"]
            indices = matched_actual.index
            metrics = []
            for i in range(len(matched_actual)):
                actual_row = matched_actual.iloc[[i]]
                pred_row = matched_pred.iloc[[i]]
                metric_value = metric_func(actual_row, pred_row, target)
                metrics.append(metric_value)
            df = pd.DataFrame(
                {
                    "actual": [round(float(x), 3) for x in actual_col.values],
                    "prediction": [round(float(x), 3) for x in pred_col.values],
                    f"{metric}": [round(float(x), 3) for x in metrics],
                },
                index=indices,
            )
            worst_n = df.sort_values(f"{metric}", ascending=False).head(5)
            worst[metric] = worst_n.reset_index().to_dict(orient="records")
        return worst

    def prepare_correlation_matrix(self, target):
        pred_dfs = []
        model_names = []
        pred_dfs.append(
            self.prediction[f"pred_{target}"]
            .reset_index(drop=True)
            .apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
        )
        model_names.append(self.config["name"])
        for model in self.config["models"]:
            model_pred = self._transform(
                read_dataframe(
                    file_path=ModelPathManager(
                        model
                    )._get_generated_predictions_data_file_paths(
                        run_type=self._args.run_type
                    )[
                        0
                    ]
                ),
                f"pred_{target}",
            )
            pred_dfs.append(
                model_pred[f"pred_{target}"]
                .reset_index(drop=True)
                .apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
            )
            model_names.append(model)
        all_preds = pd.concat(pred_dfs, axis=1)
        all_preds.columns = model_names
        corr_matrix = all_preds.corr()

        columns = list(corr_matrix.columns)
        rows = [(idx, [f"{v:.3f}" for v in row]) for idx, row in corr_matrix.iterrows()]
        return {"columns": columns, "rows": rows}

    def generate_analysis_report(self):
        env = Environment(loader=FileSystemLoader(self.template_dir))
        template = env.get_template("analysis_report_template.html")

        metadata = self.prepare_metadata()

        worst_predictions = {}
        correlation_matrix = {}
        for target in self.targets:
            worst_predictions[target] = self.prepare_worst_predictions(target)
            correlation_matrix[target] = self.prepare_correlation_matrix(target)

        html = template.render(
            metadata=metadata,
            worst_predictions=worst_predictions,
            correlation_matrix=correlation_matrix,
        )

        output_path = self.ensemble_path.reports / f"{self._args.run_type}_report.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Report written to {output_path}")
