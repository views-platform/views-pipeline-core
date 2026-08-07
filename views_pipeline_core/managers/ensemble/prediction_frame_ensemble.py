"""PredictionFrameEnsembleManager — composition-based ensemble orchestrator for sample-based predictions.

ADR-051 Phase 2: handles PredictionFrame (numpy) predictions throughout,
never converting to parquet. Fixes C-66 by using numpy-native aggregation
instead of DataFrame-based pooling.

All ensemble-specific business logic is copied from DataFrameEnsembleManager
(WET-before-DRY). The architectural difference is the data format:
PredictionFrame (N, S) numpy arrays instead of pd.DataFrame.
"""
import importlib
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import tqdm
import wandb

from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.modules.reconciliation import (
    RECONCILER_NOT_INJECTED_MSG,
    Reconciler,
)
from views_pipeline_core.exceptions import PipelineException
from views_pipeline_core.modules.frames.prediction_frame_io import load_pf, save_pf
from views_pipeline_core.files.utils import handle_ensemble_log_creation
from views_pipeline_core.modules.reconciliation.reconcile_frames import reconcile_frames
from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer
from views_pipeline_core.modules.validation.ensemble import validate_ensemble_model

from .cm_forecast_loader import load_cm_frame
from .dataframe_ensemble import EnsembleContext
from .ensemble import EnsemblePathManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_PF_AGGREGATION_METHODS = frozenset({"concat", "arithmetic_mean"})


# ---------------------------------------------------------------------------
# Pure aggregation function
# ---------------------------------------------------------------------------

def _aggregate_prediction_frames(
    frames: List[PredictionFrame],
    method: str,
) -> PredictionFrame:
    """Aggregate multiple PredictionFrame outputs into one.

    Args:
        frames: Non-empty list of PredictionFrames with identical n_rows,
                identical identifiers (time/unit arrays), and identical
                sample_count (#160 — no silent unbalanced pooling).
        method: Aggregation method — "concat" or "arithmetic_mean".

    Returns:
        New PredictionFrame with aggregated y_pred and shared identifiers.

    Raises:
        ValueError: If frames is empty, rows/identifiers/sample_count mismatch,
                    or method is unsupported.
    """
    if not frames:
        raise ValueError(
            "_aggregate_prediction_frames requires at least one PredictionFrame."
        )

    if method not in SUPPORTED_PF_AGGREGATION_METHODS:
        raise ValueError(
            f"Unsupported aggregation method: '{method}'. "
            f"Supported: {sorted(SUPPORTED_PF_AGGREGATION_METHODS)}"
        )

    reference = frames[0]
    for i, pf in enumerate(frames[1:], start=1):
        if pf.n_rows != reference.n_rows:
            raise ValueError(
                f"n_rows mismatch: frame[0] has {reference.n_rows} rows, "
                f"frame[{i}] has {pf.n_rows} rows. "
                f"All PredictionFrames must have the same number of rows."
            )
        if pf.sample_count != reference.sample_count:
            # #160: np.concatenate(axis=1) accepts differing widths, so without this
            # guard `concat` would pool constituents into a silently UNBALANCED draw
            # pool, biasing downstream quantiles/metrics with no error signal
            # (register C-205). Fail loud; deliberate weighted pooling is a future
            # feature, never a silent default.
            raise ValueError(
                f"sample_count mismatch: frame[0] has {reference.sample_count} "
                f"samples, frame[{i}] has {pf.sample_count}. All constituent "
                f"PredictionFrames must carry the same number of samples — a mixed "
                f"pool would be silently unbalanced (issue #160)."
            )
        for key in reference.identifiers:
            if key not in pf.identifiers:
                raise ValueError(
                    f"Missing identifier '{key}' in frame[{i}]. "
                    f"All PredictionFrames must have matching identifiers."
                )
            if not np.array_equal(reference.identifiers[key], pf.identifiers[key]):
                raise ValueError(
                    f"Identifier '{key}' values differ between frame[0] and frame[{i}]. "
                    f"All PredictionFrames must have matching identifiers."
                )

    if method == "concat":
        y_agg = np.concatenate([pf.values for pf in frames], axis=1)
    else:
        y_agg = np.mean(np.stack([pf.values for pf in frames]), axis=0)

    # Reuse the reference frame's index — all frames share matching identifiers
    # (validated above) and the aggregate keeps the same rows, so its (time, unit,
    # level) index is identical. The leaf index is immutable, so sharing is safe.
    return PredictionFrame(y_agg, reference.index)


def _assert_expected_sample_count(
    model_results: Dict[str, Dict[str, PredictionFrame]],
    expected_samples_per_model: Optional[int],
    models: List[str],
    targets: List[str],
) -> Dict[str, int]:
    """Fail loud when a constituent's PRODUCED sample count is not what the config
    expects, and return the produced counts for logging.

    The #160 balance guard (`_aggregate_prediction_frames`) rejects constituents
    whose sample counts differ FROM EACH OTHER. It cannot see the case where every
    constituent reloaded a stale cached forecast at the same wrong count
    (all-equal-but-wrong) — the failure that silently discarded a sample-count
    config change during the 2026-07-20 FAO delivery (register C-85). This compares
    the produced `pf.sample_count` against the ensemble config's declared
    `expected_samples_per_model`; when that is undeclared the check is a no-op and
    only reports the produced counts.
    """
    produced: Dict[str, int] = {}
    for model_name in models:
        target_pfs = model_results.get(model_name, {})
        pf = next(
            (target_pfs[t] for t in targets if target_pfs.get(t) is not None), None
        )
        if pf is None:
            continue
        produced[model_name] = pf.sample_count
        if (
            expected_samples_per_model is not None
            and pf.sample_count != expected_samples_per_model
        ):
            raise ValueError(
                f"Constituent '{model_name}' produced {pf.sample_count} samples but "
                f"the ensemble config expects {expected_samples_per_model} "
                f"(expected_samples_per_model). A stale or mismatched cached forecast "
                f"is being reused: clear models/{model_name}/data/generated/"
                f"predictions_* and re-run, or fix the constituent's sample-count "
                f"config. (This is the all-equal-but-wrong case the #160 balance "
                f"guard cannot detect — register C-85.)"
            )
    return produced


def _cached_sample_count(pf_dir: Path) -> Optional[int]:
    """The sample width (axis 1) of a cached ``y_pred.npy`` without loading it,
    or None if the file is absent or not a 2-D (N, S) array.

    Reads the array's own shape header via mmap — cheap, and it reports what the
    cache ACTUALLY contains rather than a self-reported metadata value that could
    itself drift. Used to detect a sample-count-stale cache (register C-85).
    """
    y_pred_path = Path(pf_dir) / "y_pred.npy"
    if not y_pred_path.exists():
        return None
    try:
        arr = np.load(y_pred_path, mmap_mode="r")
    except Exception:  # noqa: BLE001 — an unreadable peek must never break the
        # load path; returning None defers to load_pf (and S1 still guards the
        # produced count downstream).
        return None
    return int(arr.shape[1]) if arr.ndim == 2 else None


# ---------------------------------------------------------------------------
# PredictionFrameEnsembleManager
# ---------------------------------------------------------------------------

class PredictionFrameEnsembleManager:
    """Composition-based ensemble orchestrator for PredictionFrame predictions.

    Does NOT inherit from ForecastingModelManager or ModelManager.
    Composes ADR-045 stages directly. PredictionFrame data stays in numpy
    format throughout — no conversion to DataFrame or parquet.
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
        from views_pipeline_core.managers.reporting.stage import ReportingStage
        from views_pipeline_core.modules.logging import LoggingModule
        from views_pipeline_core.modules.wandb import WandBModule

        self._ensemble_path = ensemble_path
        self._wandb_notifications = wandb_notifications
        self._use_prediction_store = use_prediction_store
        self._datastore = None  # built lazily by _build_datastore (#269)
        # Accepted for uniform composition-root injection; PFE has no point
        # reconciliation path yet (probabilistic PFE reconciliation is #200).
        self._reconciler = reconciler
        self._entity = "views_pipeline"
        self._args: Optional[ForecastingModelArgs] = None

        self._logger = LoggingModule(model_path=ensemble_path).get_logger()
        self._wandb_module = WandBModule(
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

        self._evaluation_stage = EvaluationStage(
            wandb_module=self._wandb_module,
            io_manager=None,
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
        # Reconciliation is read from config and applied in `_forecast_ensemble` via the
        # injected `Reconciler` port (#236, epic #233). A configured reconciliation with no
        # injected reconciler fails loud at apply time (RECONCILER_NOT_INJECTED_MSG) — no
        # silent-off.
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
            prediction_format="prediction_frame",
            partition_dict=self._partition_dict or {},
            expected_samples_per_model=c.get("expected_samples_per_model"),
        )

    # ------------------------------------------------------------------
    # Execution methods
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
                self._evaluate_ensemble(ctx)

                handle_ensemble_log_creation(
                    model_path=self._ensemble_path, config=ctx.configs
                )

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
                self._forecast_ensemble(ctx)

                self._wandb_module.send_alert(
                    title=f"Forecasting for ensemble {ctx.configs['name']} completed successfully.",
                )

                handle_ensemble_log_creation(
                    model_path=self._ensemble_path, config=ctx.configs
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
    # Evaluation delegation
    # ------------------------------------------------------------------

    def _evaluate_predictions(
        self,
        pf_predictions: Dict[str, List[PredictionFrame]],
        ctx: EnsembleContext,
    ) -> None:
        from views_pipeline_core.managers.evaluation.stage import EvaluationContext

        eval_ctx = EvaluationContext(
            configs=ctx.configs,
            model_path=self._ensemble_path,
            prediction_format="prediction_frame",
            partition_dict=ctx.partition_dict,
            run_type=ctx.run_type,
            data_loader=None,
            prepare_actuals_df=lambda df: df,
        )
        self._evaluation_stage.evaluate(
            pf_predictions, eval_ctx, ensemble=True
        )

    # ------------------------------------------------------------------
    # Ensemble orchestration
    # ------------------------------------------------------------------

    def _train_ensemble(self, ctx: EnsembleContext) -> None:
        for model_name in tqdm.tqdm(ctx.models, desc="Training ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            self._train_model_artifact(model_name, ctx)

    def _evaluate_ensemble(
        self, ctx: EnsembleContext
    ) -> Dict[str, List[PredictionFrame]]:
        from views_pipeline_core.managers.model import ForecastingModelManager

        n_sequences = ForecastingModelManager._resolve_evaluation_sequence_number(
            ctx.eval_type
        )

        model_results: Dict[str, Dict[str, List[PredictionFrame]]] = {}
        for model_name in tqdm.tqdm(ctx.models, desc="Evaluating ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            model_results[model_name] = self._evaluate_model_artifact(
                model_name, ctx
            )

        aggregated: Dict[str, List[PredictionFrame]] = {}
        for target in ctx.targets:
            aggregated[target] = []
            for seq_idx in range(n_sequences):
                frames_for_seq = []
                for model_name in ctx.models:
                    target_pfs = model_results[model_name].get(target, [])
                    if seq_idx >= len(target_pfs):
                        raise ValueError(
                            f"Model '{model_name}' returned only {len(target_pfs)} "
                            f"outputs for target '{target}', but at least "
                            f"{seq_idx + 1} are required."
                        )
                    frames_for_seq.append(target_pfs[seq_idx])

                agg_pf = _aggregate_prediction_frames(
                    frames_for_seq, method=ctx.aggregation
                )
                save_dir = (
                    self._ensemble_path.data_generated
                    / f"predictions_{ctx.run_type}_{ctx.timestamp}"
                    / f"origin_{seq_idx}"
                    / target
                )
                save_pf(agg_pf, save_dir)
                aggregated[target].append(agg_pf)

        tqdm.tqdm.write("Aggregation complete.")

        self._evaluate_predictions(aggregated, ctx)

        return aggregated

    def _forecast_ensemble(
        self, ctx: EnsembleContext
    ) -> Dict[str, PredictionFrame]:
        model_results: Dict[str, Dict[str, PredictionFrame]] = {}

        for model_name in tqdm.tqdm(ctx.models, desc="Forecasting ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            model_results[model_name] = self._forecast_model_artifact(
                model_name, ctx
            )

        # Fail loud if a constituent's PRODUCED sample count is not what the config
        # expects, and record the produced counts (register C-85): the only visible
        # signal that a stale cached forecast was reloaded used to be progress-bar
        # duration. The #160 balance guard cannot catch all-equal-but-wrong.
        produced_counts = _assert_expected_sample_count(
            model_results,
            ctx.expected_samples_per_model,
            ctx.models,
            ctx.targets,
        )
        logger.info(
            "Ensemble '%s' forecast sample counts per constituent: %s "
            "(expected_samples_per_model=%s)",
            ctx.configs["name"],
            produced_counts,
            ctx.expected_samples_per_model,
        )

        forecasts: Dict[str, PredictionFrame] = {}
        for target in ctx.targets:
            frames = []
            for model_name in ctx.models:
                pf = model_results[model_name].get(target)
                if pf is None:
                    raise ValueError(
                        f"Model '{model_name}' did not produce a forecast "
                        f"for target '{target}'."
                    )
                frames.append(pf)

            agg_pf = _aggregate_prediction_frames(
                frames, method=ctx.aggregation
            )
            if ctx.reconciliation:
                agg_pf = self._reconcile_forecast(agg_pf, target, ctx)
            save_dir = (
                self._ensemble_path.data_generated
                / f"predictions_{ctx.run_type}_{ctx.timestamp}"
                / target
            )
            save_pf(agg_pf, save_dir)
            forecasts[target] = agg_pf
            if self._use_prediction_store:
                # #269 / ADR-013 §3: the Hop-A publish leg — Track A archives +
                # manifest-last, per (run, target). Runs only under the flag; child
                # runs never reach here with it set (prediction_store=False forcing).
                self._publish_sampled_forecast(agg_pf, target, ctx)

        return forecasts

    def _publish_sampled_forecast(
        self, agg_pf: PredictionFrame, target: str, ctx: EnsembleContext
    ) -> None:
        """Publish one (run, target) to the prediction store (ADR-013 §3, #269)."""
        from views_pipeline_core.managers.ensemble.sampled_forecast_publisher import (
            publish_sampled_forecast,
        )

        publish_sampled_forecast(
            self._build_datastore(),
            agg_pf,
            ensemble_name=ctx.configs["name"],
            internal_target=target,
            run_id=f"{ctx.configs['name']}_{ctx.run_type}_{ctx.timestamp}",
            level=ctx.configs["level"],
            reconciled=bool(ctx.reconciliation),
        )

    def _build_datastore(self):
        """Lazily construct the store uploader (same pattern as model.py's publish path:
        `PredictionStoreConfig.from_environment()` fails loud on missing credentials)."""
        if self._datastore is None:
            from views_pipeline_core.configs.prediction_store import PredictionStoreConfig
            from views_pipeline_core.modules.datastore import DatastoreModule

            config = PredictionStoreConfig.from_environment()
            self._datastore = DatastoreModule(
                appwrite_file_manager_config=config.to_appwrite_config(
                    self._ensemble_path
                )
            )
        return self._datastore

    def _reconcile_forecast(
        self, agg_pf: PredictionFrame, target: str, ctx: EnsembleContext
    ) -> PredictionFrame:
        """Reconcile one aggregated grid forecast to its country (cm) totals.

        Fail loud (no silent-off) if reconciliation is configured but no concrete
        `Reconciler` was injected at the composition root. Geography is held by the
        injected reconciler (views-models) — pipeline-core builds no map here.
        """
        if self._reconciler is None:
            raise PipelineException(
                RECONCILER_NOT_INJECTED_MSG, wandb_module=self._wandb_module
            )
        cm_frame = load_cm_frame(ctx.reconcile_with, target, ctx.run_type)
        reconciled = reconcile_frames(self._reconciler, cm_frame, agg_pf)
        logger.info(
            "Reconciled '%s' to country totals from '%s' (%s).",
            target,
            ctx.reconcile_with,
            ctx.reconciliation,
        )
        return reconciled

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
    ) -> Dict[str, List[PredictionFrame]]:
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
        n_sequences = ForecastingModelManager._resolve_evaluation_sequence_number(
            ctx.eval_type
        )

        result: Dict[str, List[PredictionFrame]] = {}
        for target in ctx.targets:
            pfs = []
            for seq_idx in range(n_sequences):
                pf_dir = (
                    path_generated
                    / f"predictions_{run_type}_{ts}"
                    / f"origin_{seq_idx}"
                    / target
                )
                pf = self._load_or_generate_pf(
                    model_path=model_path,
                    model_name=model_name,
                    pf_dir=pf_dir,
                    ctx=ctx,
                    mmap=True,
                    evaluate=True,
                )
                pfs.append(pf)
            result[target] = pfs

        return result

    def _forecast_model_artifact(
        self, model_name: str, ctx: EnsembleContext
    ) -> Dict[str, PredictionFrame]:
        from views_pipeline_core.data.model_path import ModelPathManager

        logger.info(f"Forecasting single model {model_name}...")
        model_path = ModelPathManager(model_name)
        run_type = ctx.run_type
        path_generated = model_path.data_generated
        path_artifact = model_path.resolve_artifact_path(
            run_type=run_type
        )

        ts = path_artifact.stem[-15:]

        result: Dict[str, PredictionFrame] = {}
        for target in ctx.targets:
            pf_dir = (
                path_generated
                / f"predictions_{run_type}_{ts}"
                / target
            )
            pf = self._load_or_generate_pf(
                model_path=model_path,
                model_name=model_name,
                pf_dir=pf_dir,
                ctx=ctx,
                mmap=False,
                forecast=True,
            )
            result[target] = pf

        return result

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
        return ForecastingModelArgs(
            run_type=ctx.args.run_type,
            train=train,
            evaluate=evaluate,
            forecast=forecast,
            saved=saved,
            eval_type=ctx.args.eval_type,
            update_viewser=ctx.args.update_viewser,
            prediction_store=False,
            wandb_notifications=self._wandb_notifications,
            override_timestep=ctx.args.override_timestep,
        )

    def _execute_shell_script(
        self,
        model_path,
        model_name: str,
        model_args: ForecastingModelArgs,
    ) -> None:
        """Delegate to :func:`execute_model_subprocess` (C-2 audit: shared helper)."""
        from views_pipeline_core.modules.ensemble.subprocess_runner import (
            execute_model_subprocess,
        )
        execute_model_subprocess(
            model_path=model_path,
            model_name=model_name,
            model_args=model_args,
            wandb_module=self._wandb_module,
        )

    def _load_or_generate_pf(
        self,
        model_path,
        model_name: str,
        pf_dir: Path,
        ctx: EnsembleContext,
        mmap: bool = False,
        evaluate: bool = False,
        forecast: bool = False,
    ) -> PredictionFrame:
        y_pred_path = pf_dir / "y_pred.npy"
        if y_pred_path.exists():
            # Config-aware cache (register C-85): the cache dir is keyed on the
            # sample-agnostic model-artifact timestamp, so a sample-count config
            # change leaves the stale y_pred.npy in place and it would be reused
            # silently. Peek the cached array's ACTUAL sample width (it cannot lie
            # about itself) and regenerate when it no longer matches the ensemble's
            # expected count, instead of blindly reloading.
            cached = _cached_sample_count(pf_dir)
            expected = ctx.expected_samples_per_model
            if expected is not None and cached is not None and cached != expected:
                logger.warning(
                    "Stale cached forecast at %s: cached sample_count=%s but the "
                    "ensemble config expects %s — regenerating instead of reusing "
                    "(register C-85).",
                    pf_dir,
                    cached,
                    expected,
                )
            else:
                logger.info(f"Loading existing PredictionFrame from {pf_dir}")
                return load_pf(pf_dir, ctx.configs["level"], mmap=mmap)
        else:
            logger.info(
                f"No existing PredictionFrame at {pf_dir}. "
                f"Generating new predictions..."
            )
        model_args = self._create_model_args(
            ctx, evaluate=evaluate, forecast=forecast
        )
        self._execute_shell_script(model_path, model_name, model_args)

        try:
            return load_pf(pf_dir, ctx.configs["level"], mmap=mmap)
        except FileNotFoundError:
            logger.error(
                f"No PredictionFrame output found at {pf_dir} for "
                f"{model_name} after generation."
            )
            raise PipelineException(
                f"No PredictionFrame output found at {pf_dir} for "
                f"{model_name} after generation.",
                wandb_module=self._wandb_module,
            )

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