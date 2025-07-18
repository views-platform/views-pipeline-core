from typing import Dict, List, Optional
from pathlib import Path
import wandb
import pandas as pd
from views_pipeline_core.managers.model import ModelPathManager, ForecastingModelManager
from views_pipeline_core.wandb.utils import get_latest_run, format_evaluation_dict, format_metadata_dict, timestamp_to_date
from views_pipeline_core.reports.utils import ( 
    get_conflict_type_from_feature_name,
    filter_metrics_from_dict,
)
from views_pipeline_core.files.utils import (
    generate_model_file_name,
)
from views_pipeline_core.managers.report import ReportManager
from views_pipeline_core.configs.pipeline import PipelineConfig
import logging

logger = logging.getLogger(__name__)

class EvaluationReportTemplate:
    def __init__(self, config: Dict, model_path: ModelPathManager, run_type: str):
        self.config = config
        self.model_path = model_path
        self.run_type = run_type

    def generate(
        self, wandb_run: "wandb.apis.public.runs.Run", target: str
    ) -> Path:
        """Generate an evaluation report based on the evaluation DataFrame."""
        evaluation_dict = format_evaluation_dict(dict(wandb_run.summary))
        metadata_dict = format_metadata_dict(dict(wandb_run.config))
        conflict_code, type_of_conflict = get_conflict_type_from_feature_name(target)
        metrics = metadata_dict.get("metrics", [])
        
        report_manager = ReportManager()
        report_manager.add_heading(
            f"Evaluation report for {self.model_path.target} {self.model_path.model_name}", level=1
        )
        _timestamp = dict(wandb_run.summary).get("_timestamp", None)
        run_date_str = f"{timestamp_to_date(_timestamp)}" if _timestamp else "N/A"
        report_manager.add_heading("Run Summary", level=2)
        report_manager.add_markdown(
            markdown_text=(
            f"**Run ID**: [{wandb_run.id}]({wandb_run.url})  \n"
            f"**Owner**: {wandb_run.user.name} ({wandb_run.user.username})  \n"
            f"**Run Date**: {run_date_str}  \n"
            f"**Constituent Models**: {metadata_dict.get('models', None)}  \n" if self.model_path.target == "ensemble" else ""
            f"**Pipeline Version**: {PipelineConfig().current_version}"
            )
        )

        methodology_md = (
            f"- **Target Variable**: {target}"
            + (f" ({type_of_conflict.title()})" if type_of_conflict else "")
            + "\n"
            f"- **Level of Analysis (resolution)**: {metadata_dict.get('level', 'N/A')}\n"
            f"- **Evaluation Scheme**: `Rolling-Origin Holdout`\n"
            f"    - **Forecast Horizon**: {metadata_dict.get('steps', 'N/A')}\n"
            f"    - **Number of Rolling Origins**: {ForecastingModelManager._resolve_evaluation_sequence_number(str(metadata_dict.get('eval_type', 'standard')).lower())}\n"
        )
        report_manager.add_heading("Methodology", level=2)
        report_manager.add_markdown(markdown_text=methodology_md)
        
        # Evaluation scheme description
        eval_scheme_md = (
            "This evaluation uses a **rolling-origin holdout strategy** with an **expanding input window** and a **fixed model artifact**.\n\n"
            f"- A single model is trained once on historical data up to a cutoff date and then saved (no retraining).\n"
            f"- The model generates forecasts for a fixed forecast horizon of 36 months starting immediately after the training period.\n"
            f"- For each evaluation step, both the input data and the forecast window are shifted forward by one month, expanding the input by adding the newly available data point.\n"
            f"- The model is re-run {ForecastingModelManager._resolve_evaluation_sequence_number(str(metadata_dict.get('eval_type', 'standard')).lower())} times, each time using the same trained model artifact but with updated input data and a new rolling forecast origin.\n"
            f"- Forecast accuracy is assessed by comparing each forecast window to the corresponding true observations in the holdout test set.\n"
            f"- This scheme tests the stability and robustness of the fixed model when re-applied to updated data without retraining, simulating how the model would perform if deployed as-is and used to re-forecast each month."
        )
        report_manager.add_heading("Evaluation Scheme Description", level=2)
        report_manager.add_markdown(markdown_text=eval_scheme_md)
        
        # Model-specific report content
        if self.model_path.target == "model":
            self._add_model_report_content(report_manager, metadata_dict, evaluation_dict, conflict_code, metrics)
        elif self.model_path.target == "ensemble":
            self._add_ensemble_report_content(report_manager, metadata_dict, evaluation_dict, conflict_code, metrics)
        else:
            raise ValueError(
                f"Invalid target type: {self.model_path.target}. Expected 'model' or 'ensemble'."
            )

        # Generate report path
        report_path = (
            self.model_path.reports
            / f"report_{generate_model_file_name(run_type=self.run_type, file_extension='')}_{conflict_code}.html"
        )
        report_manager.export_as_html(report_path)
        return report_path

    def _add_model_report_content(
        self,
        report_manager: ReportManager,
        metadata_dict: Dict,
        evaluation_dict: Dict,
        conflict_code: str,
        metrics: List[str]
    ) -> None:
        """Add model-specific content to the evaluation report."""
        try:
            partition_metadata = {
                k: v
                for k, v in metadata_dict.items()
                if k.lower() in {"calibration", "validation", "forecasting"}
            }
            report_manager.add_heading("Data Partitions", level=2)
            report_manager.add_table(partition_metadata)
        except Exception:
            logger.warning("Could not find partition metadata in the run summary")

        report_manager.add_heading("Model Metrics", level=2)
        for metric in metrics:
            report_manager.add_heading(f"{str(metric).upper()}", level=3)
            report_manager.add_table(data=filter_metrics_from_dict(
                evaluation_dict=evaluation_dict,
                metric=metric,
                conflict_code=conflict_code,
                model_name=metadata_dict.get('name', None)
            ))

    def _add_ensemble_report_content(
        self,
        report_manager: ReportManager,
        metadata_dict: Dict,
        evaluation_dict: Dict,
        conflict_code: str,
        metrics: List[str]
    ) -> None:
        """Add ensemble-specific content to the evaluation report."""
        models = self.config.get("models", [])
        verified_partition_dict = None
        
        # Get constituent model runs
        constituent_model_runs = []
        for model in models:
            latest_run = get_latest_run(entity="views_pipeline", model_name=model, run_type="calibration")
            if latest_run:
                constituent_model_runs.append(latest_run)
        
        # Verify partition metadata consistency
        try:
            for model_run in constituent_model_runs:
                temp_metadata_dict = format_metadata_dict(dict(model_run.config))
                partition_metadata_dict = {
                    k: v
                    for k, v in temp_metadata_dict.items()
                    if k.lower() in {"calibration", "validation", "forecasting"}
                }
                model_name = temp_metadata_dict.get('name', "N/A")
                if verified_partition_dict is None:
                    verified_partition_dict = partition_metadata_dict
                elif verified_partition_dict != partition_metadata_dict:
                    raise ValueError(
                        f"Partition metadata mismatch between models: Offending model: {model_name}"
                    )
            report_manager.add_heading("Data Partitions", level=2)
            report_manager.add_table(verified_partition_dict)

            # Add ensemble metrics
            report_manager.add_heading("Model Metrics", level=2)
            for metric in metrics:
                full_metric_dataframe = None
                report_manager.add_heading(f"{str(metric).upper()}", level=3)
                
                # Get ensemble metrics
                full_metric_dataframe = filter_metrics_from_dict(
                    evaluation_dict=evaluation_dict,
                    metric=metric,
                    conflict_code=conflict_code,
                    model_name=metadata_dict.get('name', None)
                )
                
                # Get constituent model metrics
                for model_run in constituent_model_runs:
                    temp_evaluation_dict = format_evaluation_dict(dict(model_run.summary))
                    temp_metadata_dict = format_metadata_dict(dict(model_run.config))
                    metric_dataframe = filter_metrics_from_dict(
                        evaluation_dict=temp_evaluation_dict,
                        metric=metric,
                        conflict_code=conflict_code,
                        model_name=temp_metadata_dict.get('name', None)
                    )
                    if full_metric_dataframe is None:
                        full_metric_dataframe = metric_dataframe
                    else:
                        full_metric_dataframe = pd.concat([full_metric_dataframe, metric_dataframe], axis=0)
                
                if full_metric_dataframe is not None:
                    report_manager.add_table(data=full_metric_dataframe)
        except Exception as e:
            logger.error(f"Error generating ensemble report: {e}", exc_info=True)
            raise