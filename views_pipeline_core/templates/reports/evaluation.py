from typing import Dict, List, Optional
from pathlib import Path
import wandb
import pandas as pd
from views_pipeline_core.managers.model import ModelPathManager, ForecastingModelManager
from views_pipeline_core.wandb.utils import get_latest_run, format_evaluation_dict, format_metadata_dict, timestamp_to_date
from views_pipeline_core.reports.utils import ( 
    get_conflict_type_from_feature_name,
    filter_metrics_from_dict,
    search_for_item_name,
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
        markdown_text = (
            f"**Run ID**: [{wandb_run.id}]({wandb_run.url})  \n"
            f"**Owner**: {wandb_run.user.name} ({wandb_run.user.username})  \n"
            f"**Run Date**: {run_date_str}  \n"
        )
        if self.model_path.target == "ensemble":
            markdown_text += f"**Constituent Models**: {metadata_dict.get('models', None)}  \n"
        markdown_text += f"**Pipeline Version**: {PipelineConfig().current_version}"
        report_manager.add_markdown(markdown_text=markdown_text)

        task_definition_md = (
            f"- **Target Variable**: {target}"
            + (f" ({type_of_conflict.title()})" if type_of_conflict else "")
            + "\n"
            f"- **Spatiotemporal Resolution**: {metadata_dict.get('level', 'N/A')}\n"
            f"- **Evaluation Scheme**: `Rolling-Origin Holdout`\n"
            f"    - **Minimum Forecast Horizon**: {metadata_dict.get('steps', [None, None])[0]}\n"
            f"    - **Maximum Forecast Horizon**: {metadata_dict.get('steps', [None, None])[-1]}\n"
            f"    - **Number of Rolling Origins**: {ForecastingModelManager._resolve_evaluation_sequence_number(str(metadata_dict.get('eval_type', 'standard')).lower())}\n"
            f"    - **Context Window Origin**: {metadata_dict.get('calibration', {'train': [None, None], 'test': [None, None]}).get('train')[0]}\n"
            f"    - **Context Window Schedule**: Fixed-origin, Expanding\n"
            f"    - **Target Window Schedule**: Rolling-origin, Fixed-length\n"
            f"    - **Target Window First Origin**: {metadata_dict.get('calibration', {'train': [None, None], 'test': [None, None]}).get('train')[1]}\n"
            f"    - **Training Schedule**: Frozen trained model artifact\n"
        )
        report_manager.add_heading("Task Definition", level=2)
        report_manager.add_markdown(markdown_text=task_definition_md)
        
        # Evaluation scheme description
        eval_scheme_md = (
            "This evaluation scheme uses a **fixed-origin**, **expanding context window** with a **rolling-origin**, **fixed-length target window** strategy.\n"
            "A single **frozen trained model artifact** — trained once on data spanning a **fixed-origin**, **fixed-length training period** — is used throughout.\n"
            f"    - The model is trained once on historical data ending at a defined cutoff date, then saved as a **frozen model artifact** (not retrained).\n"
            f"    - The model generates forecasts for a **fixed forecast horizon** of {len(metadata_dict.get('steps', [*range(1, 36 + 1, 1)]))} temporal units, beginning immediately after the context window ends.\n"
            f"    - At each of the 12 evaluation iterations, the context window is expanded by one additional temporal unit of input data, always starting from the original origin and ending at a new cutoff.\n"
            f"    - The **target window** — a fixed-length, out-of-sample segment of the **target feature** — shifts forward by one temporal unit at each iteration, beginning immediately after the context window ends.\n"
            f"    - The frozen model artifact is re-applied at each iteration using the updated context window, while the model parameters remain unchanged.\n"
            f"    - **Forecast performance** is assessed by comparing the model's predictions to the ground-truth values observed in each corresponding target window of the **holdout set**.\n"
            f"    - This strategy evaluates the **stability and robustness** of a fixed model under operational conditions, where retraining is deferred and forecasts are updated as new input data becomes available.\n"
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
        # try:
        #     partition_metadata = {
        #         k: v
        #         for k, v in metadata_dict.items()
        #         if k.lower() in {"calibration", "validation", "forecasting"}
        #     }
        #     report_manager.add_heading("Data Partitions", level=2)
        #     report_manager.add_table(partition_metadata)
        # except Exception:
        #     logger.warning("Could not find partition metadata in the run summary")

        report_manager.add_heading("Model Metrics", level=2)
        # full_metric_dataframe = None
        for metric in metrics:
            # report_manager.add_heading(f"{str(metric).upper()}", level=3)
            metric_dataframe = filter_metrics_from_dict(
                evaluation_dict=evaluation_dict,
                metrics=[metric, 'mean'],
                conflict_code=conflict_code,
                model_name=metadata_dict.get('name', None)
            )
            report_manager.add_table(data=metric_dataframe)

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
            # report_manager.add_heading("Data Partitions", level=2)
            # report_manager.add_table(verified_partition_dict)

            # Add ensemble metrics
            report_manager.add_heading("Model Metrics", level=2)
            for metric in metrics:
                full_metric_dataframe = None
                report_manager.add_heading(f"{str(metric).upper()}", level=3)
                
                # Get ensemble metrics
                full_metric_dataframe = filter_metrics_from_dict(
                    evaluation_dict=evaluation_dict,
                    metrics=[metric, 'mean'],
                    conflict_code=conflict_code,
                    model_name=metadata_dict.get('name', None)
                )
                
                # Get constituent model metrics
                for model_run in constituent_model_runs:
                    temp_evaluation_dict = format_evaluation_dict(dict(model_run.summary))
                    temp_metadata_dict = format_metadata_dict(dict(model_run.config))
                    metric_dataframe = filter_metrics_from_dict(
                        evaluation_dict=temp_evaluation_dict,
                        metrics=[metric, 'mean'],
                        conflict_code=conflict_code,
                        model_name=temp_metadata_dict.get('name', None)
                    )
                    if full_metric_dataframe is None:
                        full_metric_dataframe = metric_dataframe
                    else:
                        full_metric_dataframe = pd.concat([full_metric_dataframe, metric_dataframe], axis=0)
                
                if full_metric_dataframe is not None:
                    # Sort by metric name
                    target_metric_to_sort = search_for_item_name(searchspace=full_metric_dataframe.columns.tolist(), keywords=[metric, 'mean', 'time', 'series'])
                    full_metric_dataframe = full_metric_dataframe.sort_values(by=target_metric_to_sort, ascending=True)
                    report_manager.add_table(data=full_metric_dataframe)
        except Exception as e:
            logger.error(f"Error generating ensemble report: {e}", exc_info=True)
            raise