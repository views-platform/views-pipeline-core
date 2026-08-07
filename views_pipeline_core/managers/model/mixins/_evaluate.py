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
        """Evaluate model on test data using the unified PF track.

        Unified flow (DF and PF tracks converge):
          1. Call ``_evaluate_model_artifact()`` (subclass-specific).
          2. If the result is ``List[pd.DataFrame]``, convert to
             ``Dict[str, List[PredictionFrame]]`` via
             ``PredictionFrameConverter.from_legacy_dfs``.
          3. Now always have ``Dict[str, List[PredictionFrame]]``.
          4. Stream one origin at a time via ``_evaluate_model_artifact_streaming``.
             For each origin:
             - Track A: staging numpy (temp, for mmap metrics reload)
             - Track A+: permanent numpy (for PF ensemble)
             - Combined parquet per sequence (for DF ensemble, cm_forecast_loader, reporting)
          5. If metrics are configured, reload from staging and evaluate.
        """
        import traceback
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

                # ── Unified streaming evaluation ──────────────────────────────
                _skip_delivery = self.configs.get("skip_predictions_delivery", False)
                staging_path = self._model_path.data_generated / "_pf_staging"
                _run_type = self.args.run_type
                _ts = self._model_path.resolve_artifact_path(
                    self.args.run_type, self.args.artifact_name
                ).stem[-15:]
                _level = self.configs["level"]
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
                                "in origin 0.",
                                origin_idx, sorted(missing),
                            )
                    for target in list(pf_dict.keys()):
                        pf = pf_dict.pop(target)
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
                        del pf
                        gc.collect()
                    del pf_dict
                    gc.collect()
                    n_sequences += 1

                self._evaluate_model_artifact_streaming(
                    self._eval_type, self.args.artifact_name, origin_sink=_origin_sink
                )

                # ── Save combined parquet per sequence ────────────────────────
                # After streaming, reload from staging to build combined parquets.
                # This writes the same combined multi-target layout as the legacy
                # DF path: predictions_{run_type}_{ts}_{i:02d}.parquet
                if not _skip_delivery and n_sequences > 0:
                    self._save_combined_eval_parquets(
                        staging_path, all_targets, _level, _run_type, _ts, n_sequences
                    )

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

                # ── Metrics ───────────────────────────────────────────────────
                has_metrics = self._has_evaluation_metrics()

                if has_metrics:
                    if self.configs.get("skip_evaluation_metrics", False):
                        logger.warning(
                            "skip_evaluation_metrics=True — skipping metric evaluation."
                        )
                    else:
                        raw_preds_for_metrics = {
                            target: [
                                load_pf(
                                    staging_path / f"origin_{i}" / target,
                                    _level, mmap=True,
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
                    logger.warning("No metrics specified in config")

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

    def _save_combined_eval_parquets(
        self,
        staging_path,
        all_targets: List[str],
        level: str,
        run_type: str,
        ts: str,
        n_sequences: int,
    ) -> None:
        """Save combined multi-target parquet per evaluation sequence.

        Reads the staging numpy files (Track A), converts to a combined
        Arrow table via ``to_combined_arrow_table``, and writes one
        parquet file per sequence: ``predictions_{run_type}_{ts}_{i:02d}.parquet``.

        This produces the same on-disk layout as the legacy DF evaluation
        path, so every downstream reader works unchanged.
        """
        import pyarrow.parquet as pq
        from views_pipeline_core.configs.pipeline import PipelineConfig
        from views_pipeline_core.modules.frames.prediction_frame_converter import (
            PredictionFrameConverter,
        )

        converter = PredictionFrameConverter()
        data_generated = self._model_path.data_generated
        data_generated.mkdir(parents=True, exist_ok=True)

        for i in range(n_sequences):
            pf_dict = {}
            for target in all_targets:
                pf = load_pf(
                    staging_path / f"origin_{i}" / target,
                    level, mmap=False,
                )
                pf_dict[target] = pf

            combined_table = converter.to_combined_arrow_table(pf_dict, level)
            combined_name = (
                f"predictions_{run_type}_{ts}_{i:02d}"
                f"{PipelineConfig.dataframe_format}"
            )
            pq.write_table(combined_table, data_generated / combined_name)
            logger.info(f"Saved combined eval parquet: {combined_name}")

            # Upload to prediction store if enabled
            if self._use_prediction_store:
                try:
                    combined_df = combined_table.to_pandas()
                    store_name = f"{self._model_path.model_name}_predictions_{run_type}_{ts}_{i:02d}"
                    combined_df.forecasts.set_run(self._pred_store_name)
                    combined_df.forecasts.to_store(name=store_name, overwrite=True)
                except Exception as e:
                    logger.error(f"Prediction store upload failed for seq {i}: {e}")

            # Upload to Appwrite if enabled
            if self._datastore is not None:
                try:
                    self._datastore.upload_data(
                        file=data_generated / combined_name,
                        filename=combined_name,
                        loa=level,
                        name=self._model_path.model_name,
                        targets=list(pf_dict.keys()),
                        category="forecast",
                        description="",
                        type=self._model_path.target,
                    )
                except Exception as e:
                    logger.error(f"Appwrite upload failed for seq {i}: {e}")

            del pf_dict, combined_table
            gc.collect()

    def _evaluate_prediction_dataframe(
        self, df_predictions, eval_type, ensemble=False
    ) -> None:
        """Calculate evaluation metrics from predictions."""
        from views_pipeline_core.managers.evaluation.stage import EvaluationContext

        context = EvaluationContext(
            configs=self.configs,
            model_path=self._model_path,
            prediction_format=self._prediction_format,
            partition_dict=self._partition_dict,
            run_type=self.args.run_type,
            data_loader=getattr(self, '_data_loader', None),
            prepare_actuals_df=self.prepare_actuals_df,
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
        """Build one step mapping per evaluation sequence for rolling-origin evaluation."""
        run_type = self.args.run_type

        if run_type == "forecasting":
            if not (hasattr(self, '_data_loader') and self._data_loader):
                raise ValueError("Forecasting run requires an initialized data loader to determine origin.")
            base_origin = self._data_loader.month_last
        else:
            if run_type not in self._partition_dict:
                raise KeyError(
                    f"Partition configuration for run_type '{run_type}' not found. "
                    f"Available keys: {list(self._partition_dict.keys())}"
                )
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
        """Call origin_sink(origin_idx, pf_dict) once per rolling origin.

        Default behaviour: wraps the existing batch
        ``_evaluate_model_artifact()``. If the model returns
        ``List[pd.DataFrame]`` (DF path), converts each DataFrame to
        ``Dict[str, PredictionFrame]`` via ``from_legacy_dfs`` before
        emitting — so the sink always receives PredictionFrames.

        Subclasses can override this for true streaming (one origin at
        a time without accumulating all origins in memory).
        """
        raw_preds = self._evaluate_model_artifact(eval_type, artifact_name)

        # ── Unified: convert DF → PF at the boundary ──────────────────
        if isinstance(raw_preds, list):
            # DF path: List[pd.DataFrame] → Dict[str, List[PredictionFrame]]
            import pandas as pd
            from views_pipeline_core.modules.frames.prediction_frame_converter import (
                PredictionFrameConverter,
            )
            level = self.configs["level"]
            targets = combined_targets(self.configs)
            if isinstance(targets, str):
                targets = [targets]
            converter = PredictionFrameConverter()
            # Convert: for each target, extract its column from each DF
            # and build a PredictionFrame
            pf_preds: Dict[str, List[PredictionFrame]] = {}
            for target in targets:
                pf_preds[target] = converter.from_legacy_dfs(raw_preds, target, level)
            raw_preds = pf_preds
            logger.info(
                f"Converted {len(raw_preds)} target(s) from DataFrame to "
                f"PredictionFrame for unified evaluation."
            )

        if not isinstance(raw_preds, dict):
            err_msg = (
                f"_evaluate_model_artifact() returned {type(raw_preds).__name__}, "
                f"expected List[pd.DataFrame] or Dict[str, List[PredictionFrame]]. "
                f"Model contract violation."
            )
            logger.error(err_msg)
            raise ModelEvaluationException(err_msg)

        n_origins = len(next(iter(raw_preds.values())))
        for i in range(n_origins):
            pf_dict = {target: pf_list[i] for target, pf_list in raw_preds.items()}
            origin_sink(i, pf_dict)
