from typing import Dict, List
from pathlib import Path
import logging
import wandb
import pandas as pd
from views_pipeline_core.managers.model import ModelPathManager, ForecastingModelManager
from views_pipeline_core.modules.wandb import (
    get_latest_run,
    format_evaluation_dict,
    format_metadata_dict,
    timestamp_to_date,
)
from views_pipeline_core.modules.reports import (
    search_for_item_name,
    filter_metrics_by_eval_type_and_metrics,
)
from views_pipeline_core.files.utils import (
    generate_model_file_name,
)
from views_pipeline_core.modules.reports import ReportModule
from views_pipeline_core.configs.pipeline import PipelineConfig

logger = logging.getLogger(__name__)


class EvaluationReportTemplate:
    def __init__(self, config: Dict, model_path: ModelPathManager, run_type: str):
        """
        Initializes the evaluation report class with model/ensemble configuration, model path manager, and run type.

        Args:
            config (Dict): Configuration dictionary containing evaluation parameters. You will find this in `ModelManager(model_path).config`.
            model_path (ModelPathManager): Manager object for handling model paths.
            run_type (str): Type of run.

        Attributes:
            eval_types (tuple): Types of evaluation supported ('time-series-wise', 'step-wise', 'month-wise').
            baseline_models (list): List of baseline model names used for comparison.
        """
        self.config = config
        self.model_path = model_path
        self.run_type = run_type
        self.eval_types = ["time-series-wise"] # "step-wise", "month-wise"
        self.views_models_url = "https://github.com/views-platform/views-models"

    def generate(self, wandb_run: "wandb.apis.public.runs.Run", target: str) -> Path:
        """
        Generate an evaluation report based on the provided Weights & Biases run and target variable.

        This method compiles metadata, evaluation metrics, and run details into a structured report,
        including task description and summary information. The report is exported as an HTML file
        to a designated path.

        Args:
            wandb_run (wandb.apis.public.runs.Run): The Weights & Biases run object containing summary and config data.
            target (str): The name of the target variable for which the evaluation report is generated.

        Returns:
            Path: The file path to the generated HTML evaluation report.

        Raises:
            ValueError: If the model target type is not 'model' or 'ensemble'.
        """
        """Generate an evaluation report based on the evaluation DataFrame."""
        evaluation_dict = format_evaluation_dict(dict(wandb_run.summary))
        metadata_dict = format_metadata_dict(dict(wandb_run.config))

        # Read metrics directly from the pipeline config (not from the WandB run config).
        metrics = list(dict.fromkeys(
            self.config.get("regression_point_metrics", []) +
            self.config.get("regression_sample_metrics", []) +
            self.config.get("classification_point_metrics", []) +
            self.config.get("classification_sample_metrics", []) +
            self.config.get("regression_metrics", []) +
            self.config.get("classification_metrics", []) +
            self.config.get("metrics", [])
        ))
        if not metrics:
            logger.warning("No metrics found in config. Report metric tables will be empty.")

        report_manager = ReportModule()
        report_manager.add_heading(
            f"Evaluation report for {self.model_path.target} {self.model_path.model_name}",
            level=1,
        )
        _timestamp = dict(wandb_run.summary).get("_timestamp", None)
        run_date_str = f"{timestamp_to_date(_timestamp)}" if _timestamp else "N/A"
        report_manager.add_heading("Run Summary", level=2)
        markdown_text = (
            f"**Run ID**: [{wandb_run.id}]({wandb_run.url}) (links to WandB run) \n"
            f"**Owner**: {wandb_run.user.name} ({wandb_run.user.username})  \n"
            f"**Run Date**: {run_date_str}  \n"
        )
        if self.model_path.target == "ensemble":
            markdown_text += (
                f"**Constituent Models**: {metadata_dict.get('models', None)}  \n"
            )
        markdown_text += f"**Pipeline Version**: {PipelineConfig.current_version}"
        report_manager.add_markdown(markdown_text=markdown_text)

        task_definition_md = (
            f"- **Target Variable**: {target}\n"
            f"- **Spatiotemporal Resolution**: {metadata_dict.get('level', 'N/A')}\n"
            f"- **Evaluation Scheme**: `Rolling-Origin Holdout`\n"
            f"    - **Minimum forecast lead time**: {metadata_dict.get('steps', [None, None])[0]}\n"
            f"    - **Maximum forecast lead time**: {metadata_dict.get('steps', [None, None])[-1]}\n"
            f"    - **Number of Rolling Origins**: {ForecastingModelManager._resolve_evaluation_sequence_number(str(metadata_dict.get('eval_type', 'standard')).lower())}\n"
            f"    - **Context Window Origin**: {metadata_dict.get(self.run_type, {'train': [None, None], 'test': [None, None]}).get('train')[0]}\n"
            f"    - **Context Window Schedule**: Fixed-origin, Expanding\n"
            f"    - **Target Window Schedule**: Rolling-origin, Fixed-length\n"
            f"    - **Target Window First Origin**: {metadata_dict.get(self.run_type, {'train': [None, None], 'test': [None, None]}).get('test')[0]}\n"
            f"    - **Training Schedule**: Frozen trained model artifact\n"
        )
        report_manager.add_heading("Task Description", level=2)
        report_manager.add_markdown(markdown_text=task_definition_md)

        # Model-specific report content
        if self.model_path.target in ("model", "ensemble"):
            self._add_report_content(
                report_manager, metadata_dict, evaluation_dict, target, metrics
            )
        else:
            raise ValueError(
                f"Invalid target type: {self.model_path.target}. Expected 'model' or 'ensemble'."
            )

        # Generate report path
        report_path = (
            self.model_path.reports
            / f"report_{generate_model_file_name(run_type=self.run_type, file_extension='')}_{target}.html"
        )
        report_manager.export_as_html(report_path)
        logger.info(f"Exported report to {report_path}")
        return report_path

    def _add_report_content(
        self,
        report_manager: ReportModule,
        metadata_dict: Dict,
        evaluation_dict: Dict,
        target_identifier: str,
        metrics: List[str],
    ) -> None:
        """
        Adds content to the evaluation report.

        This method populates the evaluation report with metrics and metadata for ensemble and single model runs. It performs the following steps:
        1. Retrieves the list of models involved in the ensemble, including any baseline models.
        2. Gathers the latest calibration run for each constituent model.
        3. Verifies that the partition metadata (e.g., calibration, validation, forecasting) is consistent across all constituent models.
        4. For each evaluation type, collects and combines metrics from both the ensemble and its constituent models.
        5. Sorts the combined metrics by the specified metric and adds them as tables to the report.

        Args:
            report_manager (ReportModule): The report manager instance used to add content to the report.
            metadata_dict (Dict): Metadata dictionary for the ensemble run.
            evaluation_dict (Dict): Evaluation results dictionary for the ensemble run.
            target_identifier (str): Identifier for the target variable.
            metrics (List[str]): List of metric names to include in the report.

        Raises:
            ValueError: If partition metadata is inconsistent across constituent models.
            Exception: If any other error occurs during report generation, it is logged and re-raised.
        """
        models = self.config.get(
            "models", []
        )  # will only be populated for ensemble runs
        # models.append(self.model_path.model_name)

        # Collect baseline model names from all tier-specific keys in the pipeline config.
        baseline_models = list(dict.fromkeys(
            self.config.get("regression_point_baselines", []) +
            self.config.get("regression_sample_baselines", []) +
            self.config.get("classification_point_baselines", []) +
            self.config.get("classification_sample_baselines", [])
        ))
        if not baseline_models:
            logger.warning("No baseline models found in config. Baseline rows will be absent from the report.")
        models = list(set(models).union(baseline_models))
        logger.info(f"Models to search for: {models}")
        verified_partition_dict = None
        verified_level = metadata_dict.get("level", None)

        # Get constituent model runs
        constituent_model_runs = []
        for model in models:
            try:
                latest_run = get_latest_run(
                    entity="views_pipeline", model_name=model, run_type=self.run_type
                )
                if latest_run:
                    constituent_model_runs.append(latest_run)
            except Exception as e:
                logger.warning(
                    f"Error retrieving latest run for model '{model}': {e}. Skipping...",
                    exc_info=False,
                )
                continue

        # Verify partition metadata consistency
        try:
            for model_run in constituent_model_runs:
                temp_metadata_dict = format_metadata_dict(dict(model_run.config))
                partition_metadata_dict = {
                    k: v
                    for k, v in temp_metadata_dict.items()
                    if k.lower() in {"calibration", "validation", "forecasting"}
                }
                if verified_level is None:
                    verified_level = temp_metadata_dict.get("level", None)
                elif verified_level != temp_metadata_dict.get("level", None):
                    raise ValueError(
                        f"LoA metadata mismatch between models: Offending model: {temp_metadata_dict.get('name', 'N/A')}. Expected level: {verified_level}, found: {temp_metadata_dict.get('level', 'N/A')}"
                    )
                model_name = temp_metadata_dict.get("name", "N/A")
                if verified_partition_dict is None:
                    verified_partition_dict = partition_metadata_dict
                elif verified_partition_dict != partition_metadata_dict:
                    logger.error("Partition metadata mismatch: %s vs %s", verified_partition_dict, partition_metadata_dict)
                    raise ValueError(
                        f"Partition metadata mismatch between models: Offending model: {model_name}"
                    )

            # Add ensemble metrics
            report_manager.add_heading("Model Metrics", level=2)
            report_manager.add_markdown(
                markdown_text=f"More information about the following models can be found [here]({self.views_models_url})\n"
            )
            for eval_type in self.eval_types:
                full_metric_dataframe = None
                full_metric_dataframe = filter_metrics_by_eval_type_and_metrics(
                    evaluation_dict=evaluation_dict,
                    eval_type=eval_type,
                    metrics=metrics,
                    target_identifier=target_identifier,
                    model_name=metadata_dict.get("name", None),
                    keywords=["mean"],
                )

                # Get constituent model metrics
                for model_run in constituent_model_runs:
                    temp_evaluation_dict = format_evaluation_dict(
                        dict(model_run.summary)
                    )
                    temp_metadata_dict = format_metadata_dict(dict(model_run.config))
                    metric_dataframe = filter_metrics_by_eval_type_and_metrics(
                        evaluation_dict=temp_evaluation_dict,
                        eval_type=eval_type,
                        metrics=metrics,
                        target_identifier=target_identifier,
                        model_name=temp_metadata_dict.get("name", None),
                        keywords=["mean"],
                    )
                    if full_metric_dataframe is None:
                        full_metric_dataframe = metric_dataframe
                    else:
                        full_metric_dataframe = pd.concat(
                            [full_metric_dataframe, metric_dataframe], axis=0
                        )

                if full_metric_dataframe is not None and not full_metric_dataframe.empty:
                    # Sort by MSLE (point), then CRPS (probabilistic), then first available metric.
                    _cols = full_metric_dataframe.columns.tolist()
                    _sort_candidates = ["MSLE", "CRPS"]
                    target_metric_to_sort = None
                    for _candidate in _sort_candidates:
                        if _candidate in metrics:
                            target_metric_to_sort = search_for_item_name(
                                searchspace=_cols, keywords=[_candidate]
                            )
                        if target_metric_to_sort:
                            break
                    if not target_metric_to_sort and metrics:
                        target_metric_to_sort = search_for_item_name(
                            searchspace=_cols, keywords=[list(metrics)[0]]
                        )
                    full_metric_dataframe = full_metric_dataframe.sort_values(
                        by=target_metric_to_sort, ascending=True
                    )
                    report_manager.add_table(
                        data=full_metric_dataframe,
                        header=f"{eval_type.replace('-', ' ').title()}",
                    )
                else:
                    logger.warning(
                        f"No metrics found for evaluation type '{eval_type}' in the ensemble report. Constituent models may not have metrics for this evaluation type."
                    )
        except Exception as e:
            logger.error(f"Error generating ensemble report: {e}", exc_info=True)
            raise