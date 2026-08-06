"""EvaluationMixin — extracted from ForecastingModelManager (C-1 audit decision).

This mixin contains the evaluate concern methods. It is mixed into
ForecastingModelManager via multiple inheritance; all methods read/write
``self._*`` attributes that are set on the combined instance by
ModelManager.__init__ and ForecastingModelManager.__init__.

Backward compatibility: every method keeps its exact name and signature.
r2darts2's DartsForecastingModelManager (which subclasses
ForecastingModelManager) continues to work unchanged.
"""
from __future__ import annotations

# Imports are kept minimal — each mixin imports only what its methods use.
# Heavy imports (pandas, pyarrow) are deferred to runtime inside method bodies
# to preserve import purity (the base manager must remain pandas-free at
# module scope; see _lazy.py and tests/test_import_purity.py).

import logging
import gc
import traceback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:  # annotation-only; never imported at runtime
    import pandas as pd

from views_pipeline_core.exceptions import (
    DataFetchException,
    ModelEvaluationException,
    ModelTrainingException,
    PipelineException,
)
from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.managers.configuration.configuration import ConfigurationManager, combined_targets
from views_pipeline_core.modules.dataloaders.datafactory_contract import (
    DATA_FORMAT_DATAFRAME,
    DATA_FORMAT_FEATURE_FRAME,
    declared_data_format,
)
from views_pipeline_core.modules.frames.prediction_frame_io import load_pf, save_pf

logger = logging.getLogger(__name__)


class EvaluationMixin:
    """Mixin providing evaluate methods for ForecastingModelManager."""

    def _execute_model_evaluation(self) -> None:
        """
        Evaluate model on test data.
        
        Generates predictions, validates structure, calculates metrics,
        and saves evaluation results. Supports multi-sequence evaluation.
        
        Pipeline Stage:
            evaluate
        
        Side Effects:
            - Creates WandB run (job_type="evaluate")
            - Generates predictions for each sequence
            - Validates prediction DataFrames
            - Calculates and saves metrics
            - Logs to WandB
            - Sends completion notification
        
        Raises:
            ModelEvaluationException: If evaluation fails
        
        Example:
            >>> # Internal usage
            >>> self._execute_model_evaluation()
            INFO: Evaluating purple_alien...
            INFO: Validating 12 prediction sequences...
            INFO: Evaluation completed.
        
        Note:
            - Uses threadpool for parallel validation
            - Metrics calculated only if specified in config
        """
        import traceback
        from views_pipeline_core.modules.validation.core_prediction_sniffer import CorePredictionSniffer
        from views_pipeline_core.files.utils import handle_single_log_creation

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="evaluate",
        ):
            try:
                logger.info(
                    f"Evaluating {self._model_path.target} {self.configs['name']}..."
                )

                # Layer 2: log declared temporal window before expensive inference.
                # This makes the expected outcome visible in the run log so any
                # mismatch with actual model output can be diagnosed from logs alone.
                if self.args.run_type != "forecasting":
                    _steps = self.configs["steps"]
                    _base_origin = self._partition_dict[self.args.run_type]['test'][0] - 1
                    logger.info(
                        f"Declared temporal window: base_origin={_base_origin}, "
                        f"step 1 → month {_base_origin + 1}, "
                        f"step {max(_steps)} → month {_base_origin + max(_steps)} "
                        f"({len(_steps)} steps total). Model inference starting."
                    )

                import gc
                import shutil
                import concurrent.futures

                if self._prediction_format == "prediction_frame":
                    # ── PF path — streaming evaluation ───────────────────────────────
                    # Process one origin at a time so at most one origin's PredictionFrames
                    # are alive simultaneously.  Each origin writes:
                    #   Track A  staging/_pf_staging/origin_i/target/ — compact .npy,
                    #            used by the metrics reload below (mmap-safe)
                    #   Track A+ data_generated/predictions_{run_type}_{ts}/origin_i/target/
                    #            — permanent numpy for PF ensemble consumption
                    #   Track B  data_generated/predictions_*.parquet — list-in-cell,
                    #            controlled by mandatory skip_predictions_delivery key
                    _skip_delivery = self.configs["skip_predictions_delivery"]
                    if not _skip_delivery:
                        from views_pipeline_core.modules.frames.prediction_frame_converter import (
                            PredictionFrameConverter,
                        )
                        converter = PredictionFrameConverter()
                    staging_path = self._model_path.data_generated / "_pf_staging"
                    _run_type = self.args.run_type
                    _ts = self._model_path.resolve_artifact_path(
                        self.args.run_type, self.args.artifact_name
                    ).stem[-15:]
                    all_targets: List[str] = []
                    n_sequences = 0

                    def _origin_sink(
                        origin_idx: int, pf_dict: Dict[str, PredictionFrame]
                    ) -> None:
                        nonlocal n_sequences
                        if not all_targets:
                            all_targets.extend(pf_dict.keys())
                        else:
                            missing = set(all_targets) - set(pf_dict.keys())
                            if missing:
                                logger.warning(
                                    "Origin %d is missing targets %s present "
                                    "in origin 0. These targets will not be "
                                    "saved for this origin, and mmap reload "
                                    "will fail at metric evaluation time.",
                                    origin_idx, sorted(missing),
                                )
                        for target in list(pf_dict.keys()):
                            pf = pf_dict.pop(target)  # remove from dict → refcount drops
                            # Track A — compact numpy (metrics mmap reload)
                            save_pf(pf, staging_path / f"origin_{origin_idx}" / target)
                            # Track A+ — permanent numpy for ensemble consumption
                            save_pf(
                                pf,
                                self._model_path.data_generated
                                / f"predictions_{_run_type}_{_ts}"
                                / f"origin_{origin_idx}"
                                / target
                            )
                            # Track B — list-in-cell parquet (delivery)
                            # Controlled by mandatory skip_predictions_delivery key.
                            # PF ensembles consume Track A+ numpy, not Track B.
                            if not _skip_delivery:
                                table = converter.to_arrow_table(
                                    pf, target, level=self.configs["level"]
                                )
                                self._save_predictions(
                                    table, self._model_path.data_generated, origin_idx,
                                    send_alert=False,
                                    target_identifier=target,
                                )
                                del table
                            del pf
                            gc.collect()  # return ~1.6 GB to OS per target
                        del pf_dict  # now empty — trivial
                        gc.collect()
                        n_sequences += 1

                    self._evaluate_model_artifact_streaming(
                        self._eval_type, self.args.artifact_name, origin_sink=_origin_sink
                    )
                else:
                    # ── DF path (legacy DataFrame format) ────────────────────────────
                    raw_preds = self._evaluate_model_artifact(
                        self._eval_type, self.args.artifact_name
                    )
                    # Type enforcement guard (ADR-042, fail-loud).
                    if isinstance(raw_preds, dict):
                        raise ValueError(
                            "prediction_format='dataframe' declared but "
                            "_evaluate_model_artifact() returned a dict, expected "
                            "List[pd.DataFrame]. Model contract violation."
                        )
                    self._assert_predictions_in_step_window(raw_preds)
                    # Validate (sniff) and save each prediction DataFrame.
                    n_sequences = len(raw_preds)

                    def validate_and_save(
                        df, idx, configs, model_path, save_predictions_func
                    ):
                        logger.info(
                            f"Validating evaluation dataframe of sequence {idx+1}/{n_sequences}"
                        )
                        CorePredictionSniffer(level=configs["level"]).sniff_predictions(
                            df, targets=combined_targets(configs)
                        )
                        save_predictions_func(df, model_path.data_generated, idx, send_alert=False)

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [
                            executor.submit(
                                validate_and_save,
                                df,
                                i,
                                self.configs,
                                self._model_path,
                                self._save_predictions,
                            )
                            for i, df in enumerate(raw_preds)
                        ]
                        concurrent.futures.wait(futures)

                self._wandb_module.send_alert(
                    title="Evaluation Predictions Saved",
                    text=f"Validated and saved {n_sequences} prediction sequences at {self._model_path.data_generated.relative_to(self._model_path.root)}.",
                    notifications_enabled=self._wandb_notifications,
                )

                handle_single_log_creation(
                    model_path=self._model_path,
                    config=self.configs,
                    train=False,
                )

                has_metrics = self._has_evaluation_metrics()

                if has_metrics:
                    if self.configs.get("skip_evaluation_metrics", False):
                        logger.warning(
                            "skip_evaluation_metrics=True — skipping metric evaluation "
                            "to avoid peak y_pred_out allocation at high sample counts."
                        )
                    elif self._prediction_format == "prediction_frame":
                        # Reload PFs from Track A staging files via mmap.
                        # Only accessed pages enter RAM — peak memory is bounded
                        # by the EvaluationAdapter's sequential access pattern,
                        # not by M × T × PF_size simultaneously.
                        raw_preds_for_metrics = {
                            target: [
                                load_pf(
                                    staging_path / f"origin_{i}" / target,
                                    self.configs["level"], mmap=True,
                                )
                                for i in range(n_sequences)
                            ]
                            for target in all_targets
                        }
                        self._evaluate_prediction_dataframe(
                            raw_preds_for_metrics, self._eval_type
                        )
                        del raw_preds_for_metrics
                        gc.collect()
                    else:
                        self._evaluate_prediction_dataframe(raw_preds, self._eval_type)
                else:
                    logger.warning("No metrics specified in config")

                if self._prediction_format == "prediction_frame":
                    shutil.rmtree(staging_path, ignore_errors=True)

                self._wandb_module.send_alert(
                    title=f"Evaluation for {self._model_path.target} {self.configs['name']} completed successfully.",
                    notifications_enabled=self._wandb_notifications,
                )

            except Exception as e:
                logger.error(
                    f"{self._model_path.target.title()} evaluating model: {e}",
                    exc_info=True,
                )
                raise ModelEvaluationException(
                    f"Evaluation failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _evaluate_prediction_dataframe(
        self, df_predictions, eval_type, ensemble=False
    ) -> None:
        """
        Calculate evaluation metrics from predictions.
        
        Computes metrics at multiple aggregation levels (step, time-series,
        month) and logs to WandB. Saves results to disk.
        
        Internal Use:
            Called by evaluation and sweep methods.
        
        Args:
            df_predictions: List of prediction DataFrames or single DataFrame
            eval_type: Evaluation type
            ensemble: Whether predictions from ensemble model
        
        Side Effects:
            - Calculates metrics using NativeEvaluator
            - Logs metrics to WandB
            - Saves evaluation files
            - Sends summary notification
        
        Note:
            - Loads actual values from viewser data
            - Processes each task type separately (regression/classification)
            - Groups metrics by conflict type
            - Enforces scalar predictions for point metrics
        """
        from views_pipeline_core.managers.evaluation.stage import EvaluationContext

        context = EvaluationContext(
            configs=self.configs,
            model_path=self._model_path,
            prediction_format=self._prediction_format,
            partition_dict=self._partition_dict,
            run_type=self.args.run_type,
            data_loader=getattr(self, '_data_loader', None),
            prepare_actuals_df=self.prepare_actuals_df,
            # #302: frame-fed models evaluate against the frame cache; defaults
            # keep every dataframe-path and ensemble construction unchanged.
            data_format=getattr(self, "_data_format", DATA_FORMAT_DATAFRAME),
            frame_cache_path=getattr(self, "_cached_frame_path", None),
        )
        self._evaluation_stage.evaluate(df_predictions, context, ensemble=ensemble)

    def _has_evaluation_metrics(self) -> bool:
        """Return True if any metric keys are specified in config."""
        return any([
            self.configs.get("metrics"),
            self.configs.get("regression_metrics"),
            self.configs.get("classification_metrics"),
            self.configs.get("regression_point_metrics"),
            self.configs.get("regression_sample_metrics"),
            self.configs.get("classification_point_metrics"),
            self.configs.get("classification_sample_metrics"),
        ])

    def _get_evaluation_step_mappings(self, n_sequences: int) -> List[Dict[int, int]]:
        """
        Build one step mapping per evaluation sequence for rolling-origin evaluation.

        Fulfills ADR-031 (Authority over Inference): the orchestrator is the sole
        authority on lead-times. Each sequence i is anchored at (base_origin + i),
        shifting the origin by one month per sequence as in standard rolling-origin
        cross-validation.

        Args:
            n_sequences: Number of prediction sequences (len of df_predictions list).

        Returns:
            List of dicts, one per sequence: [{base_origin+i+s: s for s in steps} ...]
        """
        run_type = self.args.run_type

        # 1. Resolve Base Origin from Authority (DNA)
        if run_type == "forecasting":
            # Forecasting origin is dynamic based on current data state (explicit override)
            if not (hasattr(self, '_data_loader') and self._data_loader):
                # Should be impossible if initialization succeeded, but rigorous check
                raise ValueError("Forecasting run requires an initialized data loader to determine origin.")
            base_origin = self._data_loader.month_last
        else:
            # Calibration/Validation origin is static from partition config
            # Structure: self._partition_dict[run_type]['train'] -> (start, end)
            
            if run_type not in self._partition_dict:
                raise KeyError(
                    f"Partition configuration for run_type '{run_type}' not found. "
                    f"Available keys: {list(self._partition_dict.keys())}"
                )
            # base_origin = test[0] - 1 is definitionally correct.
            # The forecast origin is "the last month of observed data before the
            # evaluation period begins", which is test[0] - 1 by definition.
            # Using train[1] was an implicit assumption that the partition is
            # gap-free (train[1] + 1 == test[0]). If any gap exists between
            # train end and test start, train[1] != test[0] - 1 and the old
            # formula would produce a shifted window that excludes the model's
            # last prediction month. test[0] - 1 is correct in all cases.
            base_origin = self._partition_dict[run_type]['test'][0] - 1

        steps = self.configs["steps"]

        mappings = [
            {base_origin + i + s: s for s in steps}
            for i in range(n_sequences)
        ]

        logger.debug(
            f"Step mappings built for {n_sequences} sequences "
            f"from base_origin {base_origin}: "
            f"seq[0]={mappings[0] if mappings else {}}"
        )
        return mappings

    def _evaluate_model_artifact_streaming(
        self,
        eval_type: str,
        artifact_name: str,
        origin_sink: Callable[[int, Dict[str, PredictionFrame]], None],
    ) -> None:
        """
        Call origin_sink(origin_idx, pf_dict) once per rolling origin.

        origin_sink receives a dict mapping each target name to the single
        PredictionFrame for that origin. The sink is responsible for saving
        the PF to disk and freeing it before returning.

        Subclasses should override this method to emit one origin at a time
        without accumulating all origins in memory first. Overriding is the
        primary way to eliminate the M×T×PF_size memory spike.

        Default behaviour
        -----------------
        Wraps the existing batch ``_evaluate_model_artifact()`` for backward
        compatibility with models that have not yet adopted streaming. The full
        batch dict is loaded once and then emitted origin by origin — memory
        footprint is unchanged relative to the old code path, but the sink
        interface is honoured so callers written for streaming still work.
        """
        raw_preds = self._evaluate_model_artifact(eval_type, artifact_name)
        if not isinstance(raw_preds, dict):
            err_msg = (
                f"prediction_format='prediction_frame' declared but "
                f"_evaluate_model_artifact() returned {type(raw_preds).__name__}, "
                f"expected Dict[str, List[PredictionFrame]]. "
                f"Model contract violation."
            )
            logger.error(err_msg)
            raise ModelEvaluationException(err_msg)
        n_origins = len(next(iter(raw_preds.values())))
        for i in range(n_origins):
            pf_dict = {target: pf_list[i] for target, pf_list in raw_preds.items()}
            origin_sink(i, pf_dict)

