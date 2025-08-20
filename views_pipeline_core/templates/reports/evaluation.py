from typing import Dict, List
from pathlib import Path
import wandb
import pandas as pd
from ...managers.model import ModelPathManager, ForecastingModelManager
from ...wandb.utils import (
    get_latest_run,
    format_evaluation_dict,
    format_metadata_dict,
    timestamp_to_date,
)
from ...reports.utils import (
    get_conflict_type_from_feature_name,
    # filter_metrics_from_dict,
    search_for_item_name,
    filter_metrics_by_eval_type_and_metrics,
)
from ...files.utils import (
    generate_model_file_name,
)
from ...managers.report import ReportManager
from ...configs.pipeline import PipelineConfig
import logging

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
        self.eval_types = ("time-series-wise") # "step-wise", "month-wise"
        self.cm_baseline_models = ["zero_cmbaseline", "locf_cmbaseline", "average_cmbaseline"]
        self.pgm_baseline_models = ["zero_pgmbaseline", "locf_pgmbaseline", "average_pgmbaseline"]
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
        conflict_code, type_of_conflict = get_conflict_type_from_feature_name(target)
        priority_metrics = ["MSLE", "MSE", "y_hat_bar"]
        metrics = set(metadata_dict.get("metrics", [])).intersection(priority_metrics)

        report_manager = ReportManager()
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
        markdown_text += f"**Pipeline Version**: {PipelineConfig().current_version}"
        report_manager.add_markdown(markdown_text=markdown_text)

        task_definition_md = (
            f"- **Target Variable**: {target}"
            + (f" ({type_of_conflict.title()})" if type_of_conflict else "")
            + "\n"
            f"- **Spatiotemporal Resolution**: {metadata_dict.get('level', 'N/A')}\n"
            f"- **Evaluation Scheme**: `Rolling-Origin Holdout`\n"
            f"    - **Minimum forecast lead time**: {metadata_dict.get('steps', [None, None])[0]}\n"
            f"    - **Maximum forecast lead time**: {metadata_dict.get('steps', [None, None])[-1]}\n"
            f"    - **Number of Rolling Origins**: {ForecastingModelManager._resolve_evaluation_sequence_number(str(metadata_dict.get('eval_type', 'standard')).lower())}\n"
            f"    - **Context Window Origin**: {metadata_dict.get('calibration', {'train': [None, None], 'test': [None, None]}).get('train')[0]}\n"
            f"    - **Context Window Schedule**: Fixed-origin, Expanding\n"
            f"    - **Target Window Schedule**: Rolling-origin, Fixed-length\n"
            f"    - **Target Window First Origin**: {metadata_dict.get('calibration', {'train': [None, None], 'test': [None, None]}).get('train')[1]}\n"
            f"    - **Training Schedule**: Frozen trained model artifact\n"
        )
        report_manager.add_heading("Task Description", level=2)
        report_manager.add_markdown(markdown_text=task_definition_md)

        # Evaluation scheme description
        # eval_scheme_md = (
        #     f"This evaluation scheme uses a **fixed-origin**, **expanding context window** with a **rolling-origin**, **fixed-length target window** strategy.\n"
        #     f"A single **frozen trained model artifact** — trained once on data spanning a **fixed-origin**, **fixed-length training period** — is used throughout.\n"
        #     f"    - The model is trained once on historical data ending at a defined cutoff date, then saved as a **frozen model artifact** (not retrained).\n"
        #     f"    - The model generates forecasts for a **fixed forecast horizon** of {len(metadata_dict.get('steps', [*range(1, 36 + 1, 1)]))} temporal units, beginning immediately after the context window ends.\n"
        #     f"    - At each of the {ForecastingModelManager._resolve_evaluation_sequence_number(str(metadata_dict.get('eval_type', 'standard')).lower())} evaluation iterations, the context window is expanded by one additional temporal unit of input data, always starting from the original origin and ending at a new cutoff.\n"
        #     f"    - The **target window** — a fixed-length, out-of-sample segment of the **target feature** — shifts forward by one temporal unit at each iteration, beginning immediately after the context window ends.\n"
        #     f"    - The frozen model artifact is re-applied at each iteration using the updated context window, while the model parameters remain unchanged.\n"
        #     f"    - **Forecast performance** is assessed by comparing the model's predictions to the ground-truth values observed in each corresponding target window of the **holdout set**.\n"
        #     f"    - This strategy evaluates the **stability and robustness** of a fixed model under operational conditions, where retraining is deferred and forecasts are updated as new input data becomes available.\n"
        # )
        # report_manager.add_heading("Evaluation Scheme Description", level=2)
        # report_manager.add_markdown(markdown_text=eval_scheme_md)

        # Model-specific report content
        if self.model_path.target in ("model", "ensemble"):
            self._add_report_content(
                report_manager, metadata_dict, evaluation_dict, conflict_code, metrics
            )
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
        logger.info(f"Exported report to {report_path}")
        return report_path

    def _add_model_report_content(
        self,
        report_manager: ReportManager,
        metadata_dict: Dict,
        evaluation_dict: Dict,
        conflict_code: str,
        metrics: List[str],
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
        # for metric in metrics:
        #     # report_manager.add_heading(f"{str(metric).upper()}", level=3)
        #     metric_dataframe = filter_metrics_from_dict(
        #         evaluation_dict=evaluation_dict,
        #         metrics=[metric, "mean"],
        #         conflict_code=conflict_code,
        #         model_name=metadata_dict.get("name", None),
        #     )
        #     report_manager.add_table(data=metric_dataframe)
        for eval_type in self.eval_types:
            metric_dataframe = filter_metrics_by_eval_type_and_metrics(
                evaluation_dict=evaluation_dict,
                eval_type=eval_type,
                metrics=metrics,
                conflict_code=conflict_code,
                model_name=metadata_dict.get("name", None),
                keywords=["mean"],
            )
            # if "y_hat_bar" in metric_dataframe.columns:
            #     metric_dataframe.rename(
            #         columns={"y_hat_bar": r"$\bar{\hat{y}}$"},
            #         inplace=True,
            #     )
            report_manager.add_table(
                data=metric_dataframe,
                header=f"{eval_type.replace('-', ' ').title()}",
            )

    def _add_report_content(
        self,
        report_manager: ReportManager,
        metadata_dict: Dict,
        evaluation_dict: Dict,
        conflict_code: str,
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
            report_manager (ReportManager): The report manager instance used to add content to the report.
            metadata_dict (Dict): Metadata dictionary for the ensemble run.
            evaluation_dict (Dict): Evaluation results dictionary for the ensemble run.
            conflict_code (str): Code used to resolve metric conflicts.
            metrics (List[str]): List of metric names to include in the report.

        Raises:
            ValueError: If partition metadata is inconsistent across constituent models.
            Exception: If any other error occurs during report generation, it is logged and re-raised.
        """
        models = self.config.get(
            "models", []
        )  # will only be populated for ensemble runs
        if metadata_dict.get("level", None) == "cm":
            models = set(models).union(self.cm_baseline_models)
        elif metadata_dict.get("level", None) == "pgm":
            models = set(models).union(self.pgm_baseline_models)
        else:
            logger.warning(
                f"Unknown level '{metadata_dict.get('level', None)}'. No baseline models added."
            )
        # models = set(models).union(self.baseline_models)
        verified_partition_dict = None
        verified_level = metadata_dict.get("level", None)

        # Get constituent model runs
        constituent_model_runs = []
        for model in models:
            try:
                latest_run = get_latest_run(
                    entity="views_pipeline", model_name=model, run_type="calibration"
                )
                if latest_run:
                    constituent_model_runs.append(latest_run)
            except Exception as e:
                logger.warning(
                    f"Error retrieving latest run for model '{model}': {e}. Skipping...",
                    exc_info=True,
                )

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
                    raise ValueError(
                        f"Partition metadata mismatch between models: Offending model: {model_name}"
                    )
            # report_manager.add_heading("Data Partitions", level=2)
            # report_manager.add_table(verified_partition_dict)

            # Add ensemble metrics
            report_manager.add_heading("Model Metrics", level=2)
            report_manager.add_markdown(
                markdown_text=f"More information about the following models can be found [here]({self.views_models_url})\n"
            )
            for eval_type in self.eval_types:
                full_metric_dataframe = None
                # report_manager.add_heading(f"{str(eval_type).upper()}", level=3)

                # Get ensemble metrics
                # full_metric_dataframe = filter_metrics_from_dict(
                #     evaluation_dict=evaluation_dict,
                #     metrics=[metric, "mean"],
                #     conflict_code=conflict_code,
                #     model_name=metadata_dict.get("name", None),
                # )
                full_metric_dataframe = filter_metrics_by_eval_type_and_metrics(
                    evaluation_dict=evaluation_dict,
                    eval_type=eval_type,
                    metrics=metrics,
                    conflict_code=conflict_code,
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
                        conflict_code=conflict_code,
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
                    # Sort by metric name
                    target_metric_to_sort = search_for_item_name(
                        searchspace=full_metric_dataframe.columns.tolist(),
                        keywords=["MSLE"] if "MSLE" in metrics else list(metrics)[0],
                    )
                    full_metric_dataframe = full_metric_dataframe.sort_values(
                        by=target_metric_to_sort, ascending=True
                    )
                    # if "y_hat_bar" in full_metric_dataframe.columns:
                    #     full_metric_dataframe.rename(
                    #         columns={"y_hat_bar": r"$\bar{\hat{y}}$"},
                    #         inplace=True,
                    #     )
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
