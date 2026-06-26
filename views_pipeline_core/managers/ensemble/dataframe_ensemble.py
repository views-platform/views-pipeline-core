"""DataFrameEnsembleManager — composition-based ensemble orchestrator.

Proves the ADR-045 stage composition pattern for ensembles without
inheriting from ForecastingModelManager. Fixes C-55 by including
CoreConfigSniffer in execute_single_run.

All ensemble-specific business logic (aggregation, reconciliation,
subprocess delegation) is copied from EnsembleManager (WET-before-DRY).
The architectural difference is HOW infrastructure is accessed:
composition of explicit collaborators vs inheritance chain.
"""
import importlib
import logging
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import tqdm
import wandb

from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.data.handlers import _CDataset, _PGDataset, _ViewsDataset
from views_pipeline_core.domain.reconciliation_port import Reconciler, RECONCILER_NOT_INJECTED_MSG
from views_pipeline_core.exceptions import PipelineException
from views_pipeline_core.files.utils import (
    handle_ensemble_log_creation,
    read_dataframe,
)
from views_pipeline_core.modules.aggregation.aggregator import (
    AggregationModule,
)
from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer
from views_pipeline_core.modules.validation.ensemble import validate_ensemble_model
from views_pipeline_core.types import BaseStageContext

from .ensemble import EnsemblePathManager

logger = logging.getLogger(__name__)

# ADR-034: priogrid_gid → priogrid_id rename for AggregationModule compatibility
_ENTITY_RENAME = {"priogrid_gid": "priogrid_id"}


# ---------------------------------------------------------------------------
# Frozen context — single source of truth during a run
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnsembleContext(BaseStageContext):
    """Immutable execution context for DataFrameEnsembleManager.

    Built once in execute_single_run() after config validation, then
    threaded to every method. Prevents mutable self-state drift.
    """
    project: str
    eval_type: str
    args: ForecastingModelArgs
    models: List[str]
    aggregation: str
    targets: List[str]
    reconciliation: Optional[str]
    reconcile_with: Optional[str]
    use_weights: bool
    weights: Dict[str, float]
    timestamp: str
    deployment_status: str
    prediction_format: str
    partition_dict: Dict[str, Any]


# ---------------------------------------------------------------------------
# DataFrameEnsembleManager
# ---------------------------------------------------------------------------

class DataFrameEnsembleManager:
    """Composition-based ensemble orchestrator for DataFrame predictions.

    Does NOT inherit from ForecastingModelManager or ModelManager.
    Constructs and composes ADR-045 stages directly:
      - EvaluationStage for metric computation
      - PredictionIOManager for persistence
      - ReportingStage for HTML report generation

    Fixes C-55: CoreConfigSniffer.sniff_all() runs in execute_single_run
    before any side effects, matching ForecastingModelManager behaviour.
    """

    def __init__(
        self,
        ensemble_path: EnsemblePathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
        reconciler: Optional[Reconciler] = None,
    ) -> None:
        from views_pipeline_core.managers.configuration import ConfigurationManager
        from views_pipeline_core.managers.evaluation.stage import EvaluationStage
        from views_pipeline_core.managers.prediction.io import PredictionIOManager
        from views_pipeline_core.managers.reporting.stage import ReportingStage
        from views_pipeline_core.modules.logging import LoggingModule
        from views_pipeline_core.modules.wandb import WandBModule

        self._ensemble_path = ensemble_path
        self._wandb_notifications = wandb_notifications
        self._use_prediction_store = use_prediction_store
        self._reconciler = reconciler
        self._entity = "views_pipeline"
        self._args: Optional[ForecastingModelArgs] = None

        self._logger = LoggingModule(model_path=ensemble_path).get_logger()
        self._wandb_module = WandBModule(
            entity=self._entity,
            notifications_enabled=wandb_notifications,
            models_path=ensemble_path.models,
        )

        self._script_paths = ensemble_path.get_scripts()
        self._config_deployment = self._load_config(
            "config_deployment.py", "get_deployment_config"
        )
        self._config_hyperparameters = self._load_config(
            "config_hyperparameters.py", "get_hp_config"
        )
        self._config_meta = self._load_config("config_meta.py", "get_meta_config")
        self._config_modelset = self._load_config("config_modelset.py", "get_modelset_config")
        self._partition_dict = self._load_config("config_partitions.py", "generate")

        effective_meta = dict(self._config_meta or {})
        if self._config_modelset:
            collisions = set(effective_meta) & set(self._config_modelset)
            if collisions:
                logger.warning(
                    "config_modelset overlaps config_meta on keys %s — "
                    "config_modelset values take precedence",
                    collisions,
                )
            effective_meta.update(self._config_modelset)

        self._config_manager = ConfigurationManager(
            config_hyperparameters=self._config_hyperparameters or {},
            config_deployment=self._config_deployment or {},
            config_meta=effective_meta,
            partition_dict=self._partition_dict,
            config_sweep=None,
        )

        self._datastore = None
        self._pred_store_name = None
        if use_prediction_store:
            from views_pipeline_core.configs.prediction_store import (
                PredictionStoreConfig,
            )
            from views_pipeline_core.modules.datastore import DatastoreModule

            pred_store_config = PredictionStoreConfig.from_environment()
            self._pred_store_name = self._get_pred_store_name()
            appwrite_config = pred_store_config.to_appwrite_config(ensemble_path)
            self._datastore = DatastoreModule(
                appwrite_file_manager_config=appwrite_config
            )

        self._io = PredictionIOManager(
            model_path=ensemble_path,
            wandb_module=self._wandb_module,
            wandb_notifications=wandb_notifications,
            use_prediction_store=use_prediction_store,
            datastore=self._datastore,
            pred_store_name=self._pred_store_name,
        )

        self._evaluation_stage = EvaluationStage(
            wandb_module=self._wandb_module,
            io_manager=self._io,
            wandb_notifications=wandb_notifications,
        )

        self._reporting_stage = ReportingStage(
            wandb_module=self._wandb_module,
            wandb_notifications=wandb_notifications,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def configs(self) -> Dict:
        return self._config_manager.get_combined_config()

    @property
    def args(self) -> ForecastingModelArgs:
        if self._args is None:
            raise AttributeError(
                "args not set. Call execute_single_run() first."
            )
        return self._args

    # ------------------------------------------------------------------
    # Config loading (same importlib pattern as ModelManager)
    # ------------------------------------------------------------------

    def _load_config(
        self, script_name: str, config_method: str
    ) -> Union[Dict, None]:
        script_path = self._script_paths.get(script_name)
        if script_path:
            try:
                spec = importlib.util.spec_from_file_location(
                    script_name, script_path
                )
                config_module = importlib.util.module_from_spec(spec)
                sys.modules[script_name] = config_module
                spec.loader.exec_module(config_module)
                if hasattr(config_module, config_method):
                    return getattr(config_module, config_method)()
            except (AttributeError, ImportError) as e:
                logger.error(
                    f"Error loading config from {script_name}: {e}",
                    exc_info=True,
                )
                raise
        return None

    def _get_pred_store_name(self) -> str:
        from datetime import datetime

        from views_pipeline_core.managers.package import PackageManager

        version = PackageManager.get_latest_release_version_from_github(
            repository_name="views-models"
        )
        current_date = datetime.now()
        year = current_date.year
        month = str(current_date.month).zfill(2)

        if version is None:
            version = "0.1.0"
        pred_store_name = (
            "v"
            + "".join(part.zfill(2) for part in version.split("."))
            + f"_{year}_{month}"
        )

        from views_forecasts.extensions import ViewsMetadata

        if pred_store_name not in ViewsMetadata().get_runs().name.tolist():
            logger.warning(
                f"Run {pred_store_name} not found in the database. Creating a new run."
            )
            ViewsMetadata().new_run(
                name=pred_store_name,
                description=f"Development runs for views-models with version {version} in {year}_{month}",
                max_month=999,
                min_month=1,
            )
        return pred_store_name

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute_single_run(self, args: ForecastingModelArgs) -> None:
        if not isinstance(args, ForecastingModelArgs):
            raise ValueError(
                f"args must be an instance of ForecastingModelArgs. "
                f"Got {type(args)} instead."
            )

        self._args = args

        # C-55 fix: CoreConfigSniffer runs before any side effects
        CoreConfigSniffer(
            self.configs, self._partition_dict, target=self._ensemble_path.target
        ).sniff_all(args.run_type)

        self._wandb_module.login()

        self._config_manager.update_for_single_run(
            args, wandb_module=self._wandb_module
        )

        ctx = self._build_context(args)

        try:
            if not args.train:
                validate_ensemble_model(self.configs, saved=args.saved)

            self._execute_model_tasks(ctx)
        except PipelineException:
            raise
        except Exception as e:
            logger.error(
                f"Error during ensemble execution: {e}", exc_info=True
            )
            self._wandb_module.send_alert(
                title="Ensemble Execution Error",
                text=f"An error occurred during ensemble execution: {traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR,
            )
            raise

    def _build_context(self, args: ForecastingModelArgs) -> EnsembleContext:
        c = self.configs
        return EnsembleContext(
            configs=c,
            model_path=self._ensemble_path,
            run_type=args.run_type,
            project=f"{c['name']}_{args.run_type}",
            eval_type=args.eval_type,
            args=args,
            models=c["models"],
            aggregation=c["aggregation"],
            targets=c.get("targets", c.get("regression_targets", [])),
            reconciliation=c.get("reconciliation"),
            reconcile_with=c.get("reconcile_with"),
            use_weights=c.get("use_weights", False),
            weights=c.get("weights", {}),
            timestamp=c.get("timestamp", ""),
            deployment_status=c.get("deployment_status", "shadow"),
            prediction_format=c.get("prediction_format", "dataframe"),
            partition_dict=self._partition_dict or {},
        )

    # ------------------------------------------------------------------
    # Execution methods (receive EnsembleContext)
    # ------------------------------------------------------------------

    def _execute_model_tasks(self, ctx: EnsembleContext) -> None:
        start_t = time.time()

        if ctx.args.train:
            self._execute_model_training(ctx)
        if ctx.args.evaluate:
            self._execute_model_evaluation(ctx)
        if ctx.args.forecast:
            self._execute_model_forecasting(ctx)
        if ctx.args.report and ctx.args.forecast:
            self._execute_forecast_reporting(ctx)
        if ctx.args.report and ctx.args.evaluate:
            self._execute_evaluation_reporting(ctx)

        end_t = time.time()
        minutes = (end_t - start_t) / 60
        logger.info(f"Done. Runtime: {minutes:.3f} minutes.\n")

    def _execute_model_training(self, ctx: EnsembleContext) -> None:
        with self._wandb_module.initialize_run(
            project=ctx.project, config=ctx.configs, job_type="train"
        ):
            try:
                logger.info(f"Training ensemble {ctx.configs['name']}...")
                self._train_ensemble(ctx)

                self._wandb_module.send_alert(
                    title=f"Training for ensemble {ctx.configs['name']} completed successfully.",
                )
            except PipelineException:
                raise
            except Exception:
                logger.error(
                    f"Ensemble training failed: {traceback.format_exc()}"
                )
                raise PipelineException(
                    f"Training failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_evaluation(self, ctx: EnsembleContext) -> None:
        with self._wandb_module.initialize_run(
            project=ctx.project, config=ctx.configs, job_type="evaluate"
        ):
            try:
                logger.info(f"Evaluating ensemble {ctx.configs['name']}...")
                df_predictions = self._evaluate_ensemble(ctx)

                handle_ensemble_log_creation(
                    model_path=self._ensemble_path, config=ctx.configs
                )

                for i, df in enumerate(df_predictions):
                    self._io.save_predictions(
                        df,
                        self._ensemble_path.data_generated,
                        run_type=ctx.run_type,
                        timestamp=ctx.timestamp,
                        sequence_number=i,
                    )

                self._evaluate_predictions(df_predictions, ctx)

                self._wandb_module.send_alert(
                    title=f"Evaluation for ensemble {ctx.configs['name']} completed successfully.",
                )
            except PipelineException:
                raise
            except Exception:
                logger.error(
                    f"Ensemble evaluation failed: {traceback.format_exc()}"
                )
                raise PipelineException(
                    f"Evaluation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_model_forecasting(self, ctx: EnsembleContext) -> None:
        with self._wandb_module.initialize_run(
            project=ctx.project, config=ctx.configs, job_type="forecast"
        ):
            try:
                logger.info(f"Forecasting ensemble {ctx.configs['name']}...")
                df_prediction = self._forecast_ensemble(ctx)

                self._wandb_module.send_alert(
                    title=f"Forecasting for ensemble {ctx.configs['name']} completed successfully.",
                )

                handle_ensemble_log_creation(
                    model_path=self._ensemble_path, config=ctx.configs
                )

                self._io.save_predictions(
                    df_prediction,
                    self._ensemble_path.data_generated,
                    run_type=ctx.run_type,
                    timestamp=ctx.timestamp,
                )
            except PipelineException:
                raise
            except Exception:
                logger.error(
                    f"Ensemble forecasting failed: {traceback.format_exc()}"
                )
                raise PipelineException(
                    f"Forecasting failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_forecast_reporting(self, ctx: EnsembleContext) -> None:
        from views_pipeline_core.managers.reporting.stage import ReportingContext

        with self._wandb_module.initialize_run(
            project=ctx.project, config=ctx.configs, job_type="report"
        ):
            try:
                reporting_ctx = ReportingContext(
                    configs=ctx.configs,
                    model_path=self._ensemble_path,
                    run_type=ctx.run_type,
                    entity=self._entity,
                    prediction_format=ctx.prediction_format,
                )
                self._reporting_stage.generate_forecast_report(reporting_ctx)
            except PipelineException:
                raise
            except Exception:
                logger.error(
                    f"Forecast report generation failed: {traceback.format_exc()}"
                )
                raise PipelineException(
                    f"Forecast report generation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _execute_evaluation_reporting(self, ctx: EnsembleContext) -> None:
        from views_pipeline_core.managers.reporting.stage import ReportingContext

        with self._wandb_module.initialize_run(
            project=ctx.project, config=ctx.configs, job_type="report"
        ):
            try:
                reporting_ctx = ReportingContext(
                    configs=ctx.configs,
                    model_path=self._ensemble_path,
                    run_type=ctx.run_type,
                    entity=self._entity,
                    prediction_format=ctx.prediction_format,
                )
                self._reporting_stage.generate_evaluation_report(reporting_ctx)
            except PipelineException:
                raise
            except Exception:
                logger.error(
                    f"Evaluation report generation failed: {traceback.format_exc()}"
                )
                raise PipelineException(
                    f"Evaluation report generation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    # ------------------------------------------------------------------
    # Evaluation delegation (composes EvaluationStage directly)
    # ------------------------------------------------------------------

    def _evaluate_predictions(
        self, df_predictions: List[pd.DataFrame], ctx: EnsembleContext
    ) -> None:
        from views_pipeline_core.managers.evaluation.stage import EvaluationContext

        eval_ctx = EvaluationContext(
            configs=ctx.configs,
            model_path=self._ensemble_path,
            prediction_format="dataframe",
            partition_dict=ctx.partition_dict,
            run_type=ctx.run_type,
            data_loader=None,
            prepare_actuals_df=lambda df: df,
        )
        self._evaluation_stage.evaluate(
            df_predictions, eval_ctx, ensemble=True
        )

    # ------------------------------------------------------------------
    # Ensemble orchestration
    # ------------------------------------------------------------------

    def _train_ensemble(self, ctx: EnsembleContext) -> None:
        for model_name in tqdm.tqdm(ctx.models, desc="Training ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            self._train_model_artifact(model_name, ctx)

    def _evaluate_ensemble(self, ctx: EnsembleContext) -> List[pd.DataFrame]:
        eval_results: Dict[str, List[pd.DataFrame]] = {}

        for model_name in tqdm.tqdm(ctx.models, desc="Evaluating ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            eval_results[model_name] = self._evaluate_model_artifact(
                model_name, ctx
            )

        n_outputs = len(next(iter(eval_results.values())))
        aggregated_outputs: List[pd.DataFrame] = []

        tqdm.tqdm.write("Aggregating metrics...")
        for i in range(n_outputs):
            model_dfs_i = {}
            for model_name, dfs in eval_results.items():
                if i >= len(dfs):
                    raise ValueError(
                        f"Model '{model_name}' returned only {len(dfs)} outputs, "
                        f"but at least {i + 1} are required for ensemble aggregation. "
                        f"All models must return the same number of outputs."
                    )
                model_dfs_i[model_name] = dfs[i]

            df_agg = self._get_aggregated_df(
                df_to_aggregate=model_dfs_i,
                aggregation=ctx.aggregation,
                ctx=ctx,
            )
            aggregated_outputs.append(df_agg)

        return aggregated_outputs

    def _forecast_ensemble(self, ctx: EnsembleContext) -> pd.DataFrame:
        model_dfs: Dict[str, pd.DataFrame] = {}

        for model_name in tqdm.tqdm(ctx.models, desc="Forecasting ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            df = self._forecast_model_artifact(model_name, ctx)
            model_dfs[model_name] = df

        df_prediction = self._get_aggregated_df(
            df_to_aggregate=model_dfs,
            aggregation=ctx.aggregation,
            ctx=ctx,
        )
        df_prediction = _ViewsDataset(source=df_prediction).dataframe

        if ctx.reconciliation:
            df_prediction = self._apply_reconciliation(df_prediction, ctx)

        if not isinstance(df_prediction, pd.DataFrame):
            raise TypeError(
                f"Expected predictions to be a DataFrame, got {type(df_prediction)} instead."
            )

        return df_prediction

    # ------------------------------------------------------------------
    # Model artifact execution (subprocess delegation)
    # ------------------------------------------------------------------

    def _train_model_artifact(
        self, model_name: str, ctx: EnsembleContext
    ) -> None:
        from views_pipeline_core.data.model_path import ModelPathManager

        logger.info(f"Training single model {model_name}...")
        model_path = ModelPathManager(model_name)
        model_args = self._create_model_args(ctx, train=True)
        self._execute_shell_script(model_path, model_name, model_args)

    def _evaluate_model_artifact(
        self, model_name: str, ctx: EnsembleContext
    ) -> List[pd.DataFrame]:
        from views_pipeline_core.data.model_path import ModelPathManager
        from views_pipeline_core.managers.model import ForecastingModelManager

        logger.info(f"Evaluating single model {model_name}...")
        model_path = ModelPathManager(model_name)
        run_type = ctx.run_type
        path_generated = model_path.data_generated
        path_artifact = model_path.resolve_artifact_path(
            run_type=run_type
        )

        ts = path_artifact.stem[-15:]
        preds = []

        for sequence_number in range(
            ForecastingModelManager._resolve_evaluation_sequence_number(
                ctx.eval_type
            )
        ):
            name = (
                f"{model_name}_predictions_{run_type}_{ts}_"
                f"{str(sequence_number).zfill(2)}"
            )
            pred = self._load_or_generate_prediction(
                model_path=model_path,
                model_name=model_name,
                name=name,
                path_generated=path_generated,
                run_type=run_type,
                ts=ts,
                ctx=ctx,
                sequence_number=sequence_number,
                evaluate=True,
            )
            preds.append(pred)

        return preds

    def _forecast_model_artifact(
        self, model_name: str, ctx: EnsembleContext
    ) -> pd.DataFrame:
        from views_pipeline_core.data.model_path import ModelPathManager

        logger.info(f"Forecasting single model {model_name}...")
        model_path = ModelPathManager(model_name)
        run_type = ctx.run_type
        path_generated = model_path.data_generated
        path_artifact = model_path.resolve_artifact_path(
            run_type=run_type
        )

        ts = path_artifact.stem[-15:]
        name = f"{model_name}_predictions_{run_type}_{ts}"

        return self._load_or_generate_prediction(
            model_path=model_path,
            model_name=model_name,
            name=name,
            path_generated=path_generated,
            run_type=run_type,
            ts=ts,
            ctx=ctx,
            forecast=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_model_args(
        self,
        ctx: EnsembleContext,
        train: bool = False,
        evaluate: bool = False,
        forecast: bool = False,
    ) -> ForecastingModelArgs:
        saved = ctx.args.saved if train else True
        use_prediction_store = (
            True if forecast and self._use_prediction_store else False
        )
        return ForecastingModelArgs(
            run_type=ctx.args.run_type,
            train=train,
            evaluate=evaluate,
            forecast=forecast,
            saved=saved,
            eval_type=ctx.args.eval_type,
            update_viewser=ctx.args.update_viewser,
            prediction_store=use_prediction_store,
            wandb_notifications=self._wandb_notifications,
            override_timestep=ctx.args.override_timestep,
        )

    def _execute_shell_script(
        self,
        model_path,
        model_name: str,
        model_args: ForecastingModelArgs,
    ) -> None:
        try:
            shell_command = model_args.to_shell_command(model_path)
            logger.info(f"Executing shell command: {' '.join(shell_command)}")
            subprocess.run(shell_command, check=True, timeout=7200)
        except subprocess.TimeoutExpired:
            logger.error(
                f"Shell command timed out for model {model_name} after 7200s",
            )
            raise PipelineException(
                f"Shell command timed out for model {model_name} after 7200s. "
                "Consider increasing the timeout or investigating the model script.",
                wandb_module=self._wandb_module,
            )
        except Exception as e:
            logger.error(
                f"Error during shell command execution for model {model_name}: {e}",
                exc_info=True,
            )
            raise PipelineException(
                f"Error during shell command execution for model {model_name}: {e}",
                wandb_module=self._wandb_module,
            )

    def _load_or_generate_prediction(
        self,
        model_path,
        model_name: str,
        name: str,
        path_generated: Path,
        run_type: str,
        ts: str,
        ctx: EnsembleContext,
        sequence_number: Optional[int] = None,
        evaluate: bool = False,
        forecast: bool = False,
    ) -> pd.DataFrame:
        if self._use_prediction_store:
            try:
                pred = pd.DataFrame.forecasts.read_store(
                    run=self._pred_store_name, name=name
                )
                logger.info(
                    f"Loading existing prediction {name} from prediction store"
                )
                return pred
            except (ImportError, KeyError, IndexError, ValueError, AttributeError, OSError) as e:
                logger.info(
                    f"No existing {run_type} predictions found ({type(e).__name__}). "
                    f"Generating new predictions..."
                )
        else:
            seq_suffix = (
                f"_{str(sequence_number).zfill(2)}"
                if sequence_number is not None
                else ""
            )
            file_path = (
                path_generated
                / f"predictions_{run_type}_{ts}{seq_suffix}{PipelineConfig.dataframe_format}"
            )
            if file_path.exists():
                pred = read_dataframe(file_path)
                logger.info(
                    f"Loading existing prediction {name} from local file"
                )
                return pred
            else:
                logger.info(
                    f"No existing {run_type} predictions found. Generating new predictions..."
                )

        model_args = self._create_model_args(
            ctx, evaluate=evaluate, forecast=forecast
        )
        self._execute_shell_script(model_path, model_name, model_args)

        if self._use_prediction_store:
            return pd.DataFrame.forecasts.read_store(
                run=self._pred_store_name, name=name
            )
        else:
            prediction_files = (
                model_path._get_generated_predictions_data_file_paths(run_type)
            )
            if not prediction_files:
                raise PipelineException(
                    f"No prediction files found for {model_name} after generation",
                    wandb_module=self._wandb_module,
                )
            if sequence_number is not None:
                seq_suffix = f"_{str(sequence_number).zfill(2)}"
                matching = [f for f in prediction_files if f.stem.endswith(seq_suffix)]
                if not matching:
                    raise PipelineException(
                        f"No prediction file for sequence {sequence_number} found for {model_name} after generation",
                        wandb_module=self._wandb_module,
                    )
                latest_prediction_file = matching[0]
            else:
                latest_prediction_file = prediction_files[0]
            logger.info(
                f"Loading newly generated prediction from {latest_prediction_file}"
            )
            return read_dataframe(latest_prediction_file)

    def _get_aggregated_df(
        self,
        df_to_aggregate: Dict[str, pd.DataFrame],
        aggregation: str,
        ctx: EnsembleContext,
    ) -> pd.DataFrame:
        if not df_to_aggregate:
            raise ValueError(
                "df_to_aggregate must contain at least one DataFrame."
            )

        first_df = next(iter(df_to_aggregate.values()))
        index_cols = [
            _ENTITY_RENAME.get(c, c) for c in first_df.index.names
        ]
        target_cols = ["pred_" + col for col in ctx.targets]

        manager = AggregationModule(
            index_cols=index_cols,
            target_cols=target_cols,
        )

        for model_name, df in df_to_aggregate.items():
            weight = ctx.weights.get(model_name)
            manager.add_model(data=df, weight=weight, name=model_name)

        pred_type = manager.prediction_type
        if pred_type is None:
            raise RuntimeError(
                "AggregationModule.prediction_type is None. "
                "Make sure at least one model was added with `add_model`."
            )

        aggregated_pl = manager.aggregate(
            method=aggregation,
            use_weights=ctx.use_weights,
        )

        aggregated_pdf = aggregated_pl.to_pandas()
        aggregated_pdf = aggregated_pdf.set_index(index_cols).sort_index()
        return aggregated_pdf

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def _apply_reconciliation(
        self, df_prediction: pd.DataFrame, ctx: EnsembleContext
    ) -> pd.DataFrame:
        reconciliation_type = ctx.reconciliation

        if reconciliation_type == "pgm_cm_point":
            reconciled_pg = self._reconcile_pg_with_c(
                pg_dataframe=df_prediction, ctx=ctx
            )

            if reconciled_pg is not None:
                logger.info(
                    "Reconciliation complete. Predictions reconciled with C dataset."
                )
                self._wandb_module.send_alert(
                    title="Ensemble reconciliation complete",
                    level=wandb.AlertLevel.INFO,
                )
                return reconciled_pg
            else:
                logger.error(
                    "Reconciliation configured (pgm_cm_point) but failed. "
                    "Refusing to publish unreconciled predictions as reconciled."
                )
                raise PipelineException(
                    "Reconciliation configured but failed. C dataset could not be loaded "
                    "or reconciliation computation returned None. Cannot publish "
                    "unreconciled predictions as reconciled.",
                    wandb_module=self._wandb_module,
                )
        else:
            logger.info(
                "No valid reconciliation type specified. "
                "Returning predictions without reconciliation."
            )

        return df_prediction

    def _reconcile_pg_with_c(
        self,
        pg_dataframe: pd.DataFrame,
        ctx: EnsembleContext,
        c_dataframe: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        cm_model = ctx.reconcile_with
        if cm_model is None:
            logger.info(
                "No reconciliation model specified. Skipping reconciliation."
            )
            return None

        # Fail loud (no silent-off): configured but no Reconciler injected (#194/#195).
        if self._reconciler is None:
            raise PipelineException(
                RECONCILER_NOT_INJECTED_MSG,
                wandb_module=self._wandb_module,
            )

        latest_c_dataset = self._load_c_dataset(cm_model, c_dataframe, ctx)
        if latest_c_dataset is None:
            return None

        latest_pg_dataset = _PGDataset(source=pg_dataframe)

        if latest_pg_dataset is None:
            logger.error(
                "Could not create PG dataset. Reconciliation cannot proceed."
            )
            return None

        # Reconcile via the injected port + the dataset↔frame adapter (no
        # views_reporting / views_postprocessing import here — ADP, no cycle).
        from views_pipeline_core.modules.reconciliation.adapter import reconcile_datasets

        return reconcile_datasets(self._reconciler, latest_c_dataset, latest_pg_dataset)

    def _load_c_dataset(
        self,
        cm_model: str,
        c_dataframe: Optional[pd.DataFrame],
        ctx: EnsembleContext,
    ) -> Optional[_CDataset]:
        if c_dataframe is not None:
            logger.info(f"Using provided C dataset for model {cm_model}")
            return _CDataset(source=c_dataframe)

        if self._use_prediction_store:
            try:
                from views_forecasts.extensions import ViewsMetadata

                logger.info(
                    f"Fetching latest C dataset for {cm_model} from prediction store"
                )
                run_id = ViewsMetadata().get_run_id_from_name(
                    self._pred_store_name
                )
                all_runs = (
                    ViewsMetadata().with_name(cm_model).fetch()["name"].to_list()
                )

                reconcile_with_forecasts = [
                    fc
                    for fc in all_runs
                    if cm_model in fc and "forecasting" in fc
                ]
                reconcile_with_forecasts.sort()
                reconcile_with_forecast = reconcile_with_forecasts[-1]

                return _CDataset(
                    source=pd.DataFrame.forecasts.read_store(
                        run=run_id, name=reconcile_with_forecast
                    )
                )
            except (ImportError, KeyError, IndexError, ValueError, OSError) as e:
                logger.warning(
                    f"Could not find latest C dataset for {cm_model} "
                    f"in prediction store: {e}"
                )

        try:
            logger.info(
                f"Fetching latest C dataset for {cm_model} from local path"
            )
            return _CDataset(
                source=EnsemblePathManager(
                    cm_model
                )._get_generated_predictions_data_file_paths(
                    run_type=ctx.run_type
                )[0]
            )
        except (IndexError, OSError) as e:
            logger.warning(
                f"Could not find latest C dataset for {cm_model} locally: {e}"
            )
            return None

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        attrs = [
            f"ensemble_name='{self._ensemble_path.model_name}'",
            f"wandb_notifications={self._wandb_notifications}",
            f"use_prediction_store={self._use_prediction_store}",
        ]
        if self._args is not None:
            attrs.append(f"run_type='{self._args.run_type}'")
        return (
            f"{self.__class__.__name__}(\n    "
            + "\n    ".join(attrs)
            + "\n)"
        )

    def __str__(self) -> str:
        base = f"{self.__class__.__name__} for ensemble '{self._ensemble_path.model_name}'"
        if self._args is not None:
            base += f" ({self._args.run_type})"
        return base
