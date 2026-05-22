"""
EvaluationStage — first implementation of the ADR-045 Stage pattern.

Extracted from ForecastingModelManager._evaluate_prediction_dataframe().
Receives an explicit, frozen EvaluationContext rather than reaching into
a parent class's internals.

Responsibilities:
  - Load actuals from raw viewser data
  - Build EvaluationFrame via EvaluationAdapter (PF or DF path)
  - Compute metrics via NativeEvaluator
  - Publish results to WandB and disk via PredictionIOManager
"""
import gc
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from views_pipeline_core.types import BaseStageContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationContext(BaseStageContext):
    """Immutable context for evaluation. Passed explicitly, not pulled from self.

    Extends BaseStageContext (configs, model_path, run_type) with
    evaluation-specific fields.  Making dependencies explicit documents
    the actual dependency surface of the evaluation stage.
    """
    prediction_format: str
    partition_dict: Dict[str, Any]
    data_loader: Any  # ViewsDataLoader or None
    prepare_actuals_df: Callable  # (pd.DataFrame) -> pd.DataFrame


class EvaluationStage:
    """Evaluation orchestration: load actuals, build EFs, compute metrics, publish.

    First implementation of the Stage pattern (ADR-045). Receives an explicit
    EvaluationContext; does not inherit from or access ForecastingModelManager.

    Collaborators (injected at construction):
      - wandb_module: WandBModule — metrics logging and alerts
      - io_manager: PredictionIOManager — DataFrame persistence
    """

    def __init__(self, wandb_module, io_manager, wandb_notifications: bool = False):
        self._wandb_module = wandb_module
        self._io = io_manager
        self._wandb_notifications = wandb_notifications

    def evaluate(
        self,
        df_predictions,
        context: EvaluationContext,
        ensemble: bool = False,
    ) -> None:
        """Run evaluation: load actuals, build EFs, compute metrics, publish results.

        Args:
            df_predictions: Predictions in one of two formats:
                - List[pd.DataFrame] (DF path)
                - Dict[str, List[PredictionFrame]] (PF path)
            context: Frozen EvaluationContext with all required configuration.
            ensemble: If True, load actuals from models[0] instead of own data_raw.
        """
        from views_evaluation import NativeEvaluator
        from views_pipeline_core.modules.validation.adapter import EvaluationAdapter

        # --- Load actuals ---
        df_actual = self._load_actuals(context, ensemble)
        if df_actual is None:
            return

        # --- Task/target structure ---
        tasks = {
            "regression": context.configs.get("regression_targets", []),
            "classification": context.configs.get("classification_targets", []),
        }

        evaluator = NativeEvaluator(context.configs)

        for task_type, targets in tasks.items():
            if not targets:
                continue

            logger.info(f"Processing {task_type} tasks for evaluation...")

            for target in targets:
                logger.info(f"Calculating {task_type} evaluation metrics for {target}")
                target_identifier = target
                actual_slice = df_actual[[target]]

                # --- Build EvaluationFrame (PF or DF path) ---
                ef = self._build_evaluation_frame(
                    df_predictions, actual_slice, target, context, EvaluationAdapter,
                )
                if ef is None:
                    continue

                # --- Compute metrics ---
                # legacy_compatibility=True preserves step-wise truncation to shortest
                # sequence, matching the deleted EvaluationManager wrapper (C-29).
                report = evaluator.evaluate(ef=ef, legacy_compatibility=True)
                del ef
                gc.collect()

                # --- Publish results ---
                self._publish_results(
                    report, target_identifier, context,
                )

        # --- Summary alert ---
        import wandb
        from views_pipeline_core.managers.prediction.io import PredictionIOManager

        self._wandb_module.send_alert(
            title=f"Metrics for {context.model_path.model_name}",
            text=f"{PredictionIOManager.generate_evaluation_table(wandb.summary._as_dict())}",
            notifications_enabled=self._wandb_notifications,
        )

    def _load_actuals(self, context: EvaluationContext, ensemble: bool):
        """Load and prepare actuals DataFrame from raw data."""
        from views_pipeline_core.files.utils import read_dataframe

        if not ensemble:
            raw_paths = context.model_path._get_raw_data_file_paths(
                run_type=context.run_type
            )
        else:
            from views_pipeline_core.managers.model import ModelPathManager
            mp = ModelPathManager(context.configs["models"][0])
            raw_paths = mp._get_raw_data_file_paths(
                run_type=context.configs["run_type"]
            )

        if not raw_paths:
            model_label = (
                f"ensemble constituent model {context.configs['models'][0]}"
                if ensemble
                else f"model {context.model_path.model_name}"
            )
            logger.error(
                "No raw data file found for %s (run_type=%s)",
                model_label,
                context.run_type,
            )
            return None
        df_path = raw_paths[0]

        df_raw = read_dataframe(df_path)
        logger.info(f"Raw data read from {df_path}")
        df_raw = context.prepare_actuals_df(df_raw)

        all_targets = (
            context.configs.get("regression_targets", [])
            + context.configs.get("classification_targets", [])
        )
        if not all_targets:
            return None

        df_actual = df_raw[all_targets].copy()
        del df_raw
        gc.collect()
        return df_actual

    def _build_evaluation_frame(
        self, df_predictions, actual_slice, target, context, EvaluationAdapter,
    ):
        """Build EvaluationFrame from predictions via the appropriate adapter path."""
        if context.prediction_format == "prediction_frame":
            # PF path: df_predictions is Dict[str, List[PredictionFrame]].
            raw_preds = df_predictions.pop(target, None)
            if raw_preds is None:
                logger.warning(
                    f"PF path: target '{target}' not found in predictions dict "
                    f"(keys: {list(df_predictions.keys())}). Skipping."
                )
                return None
            step_mappings = self._get_evaluation_step_mappings(
                n_sequences=len(raw_preds), context=context,
            )
            ef = EvaluationAdapter.from_prediction_frames(
                actual=actual_slice,
                predictions=raw_preds,
                target=target,
                step_mapping=step_mappings,
            )
            del raw_preds
            gc.collect()
            return ef
        else:
            # DF path: df_predictions is List[pd.DataFrame].
            raw_preds = df_predictions if isinstance(df_predictions, list) else [df_predictions]
            first_df = raw_preds[0]
            if f"pred_{target}" not in first_df.columns:
                logger.warning(
                    f"Column pred_{target} not found in prediction columns. Skipping."
                )
                return None
            pred_slices = [df[[f"pred_{target}"]] for df in raw_preds]
            step_mappings = self._get_evaluation_step_mappings(
                n_sequences=len(pred_slices), context=context,
            )
            return EvaluationAdapter.from_dataframes(
                actual=actual_slice,
                predictions=pred_slices,
                target=target,
                step_mapping=step_mappings,
            )

    def _publish_results(self, report, target_identifier, context):
        """Extract metrics from report and publish to WandB + disk."""
        schemas = report.to_dict()["schemas"]
        step_wise = schemas.get("step", {})
        time_series_wise = schemas.get("time_series", {})
        month_wise = schemas.get("month", {})

        df_step = report.to_dataframe("step")
        df_ts = report.to_dataframe("time_series")
        df_month = report.to_dataframe("month")

        self._wandb_module.log_evaluation_results(
            step_wise, month_wise, time_series_wise, target_identifier,
        )

        if not context.configs.get("sweep", False) and self._io is not None:
            self._io.save_evaluations(
                df_step, df_ts, df_month,
                context.model_path.data_generated,
                target_identifier,
                context.configs.get("run_type", ""),
                context.configs.get("timestamp", ""),
            )
        elif self._io is None:
            logger.info(
                "Skipping evaluation file save — no io_manager configured "
                "(expected for PredictionFrame ensembles)."
            )

    @staticmethod
    def _get_evaluation_step_mappings(
        n_sequences: int, context: EvaluationContext,
    ) -> List[Dict[int, int]]:
        """Build one step mapping per evaluation sequence for rolling-origin evaluation.

        Fulfills ADR-040 (Authority over Inference): the orchestrator is the sole
        authority on lead-times.
        """
        run_type = context.run_type

        if run_type == "forecasting":
            if not context.data_loader:
                raise ValueError(
                    "Forecasting run requires an initialized data loader to determine origin."
                )
            base_origin = context.data_loader.month_last
        else:
            if run_type not in context.partition_dict:
                raise KeyError(
                    f"Partition configuration for run_type '{run_type}' not found. "
                    f"Available keys: {list(context.partition_dict.keys())}"
                )
            base_origin = context.partition_dict[run_type]["test"][0] - 1

        steps = context.configs["steps"]

        mappings = [
            {base_origin + i + s: s for s in steps} for i in range(n_sequences)
        ]

        logger.debug(
            f"Step mappings built for {n_sequences} sequences "
            f"from base_origin {base_origin}: "
            f"seq[0]={mappings[0] if mappings else {}}"
        )
        return mappings
