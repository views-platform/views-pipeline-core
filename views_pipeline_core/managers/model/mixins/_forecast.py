"""ForecastingMixin — extracted from ForecastingModelManager (C-1 audit decision).

This mixin contains the forecast concern methods. It is mixed into
ForecastingModelManager via multiple inheritance; all methods read/write
``self._*`` attributes that are set on the combined instance by
ModelManager.__init__ and ForecastingModelManager.__init__.

Backward compatibility: every method keeps its exact name and signature.
r2darts2's DartsForecastingModelManager (which subclasses
ForecastingModelManager) continues to work unchanged.
"""
from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from views_pipeline_core.exceptions import ModelForecastingException, PipelineException
from views_pipeline_core.modules.frames.prediction_frame_io import load_pf, save_pf
from views_pipeline_core.managers.forecasting.stage import ForecastingStage, ForecastingContext

logger = logging.getLogger(__name__)


class ForecastingMixin:
    """Mixin providing forecast methods for ForecastingModelManager."""

    def _execute_model_forecasting(self) -> None:
        """Generate future predictions and persist them.

        Unified flow (DF and PF tracks converge):
          1. Call ``_forecast_model_artifact()`` (subclass-specific).
          2. If the result is a DataFrame, convert it to
             ``Dict[str, PredictionFrame]`` via
             ``PredictionFrameConverter.from_prediction_df``.
          3. Now always have ``Dict[str, PredictionFrame]``.
          4. Save Track A+ numpy (per-target directories) for
             views-reporting + PF ensemble.
          5. Save combined multi-target parquet for DF ensemble,
             ``load_cm_frame``, reporting, and all other consumers.
          6. Upload to prediction store + Appwrite if enabled.
        """
        from views_pipeline_core.managers.forecasting.stage import ForecastingContext

        with self._wandb_module.initialize_run(
            project=self._project,
            config=self.configs,
            job_type="forecast",
        ):
            try:
                predictions = self._forecast_model_artifact(self.args.artifact_name)

                # --- Unify: convert DF → Dict[str, PredictionFrame] ---
                if not isinstance(predictions, dict):
                    import pandas as pd
                    from views_pipeline_core.modules.frames.prediction_frame_converter import (
                        PredictionFrameConverter,
                    )
                    level = self.configs["level"]
                    targets = self.configs.get("regression_targets") or self.configs.get("targets") or []
                    if isinstance(targets, str):
                        targets = [targets]
                    converter = PredictionFrameConverter()
                    predictions = {
                        t: converter.from_prediction_df(
                            predictions[[f"pred_{t}"]], t, level
                        )
                        for t in targets
                    }
                    logger.info(
                        f"Converted DataFrame to {len(predictions)} PredictionFrames "
                        f"for unified persistence."
                    )

                # --- Step 4: Track A+ numpy (per-target directories) ---
                _ts = self._model_path.resolve_artifact_path(
                    self.args.run_type, self.args.artifact_name
                ).stem[-15:]
                for target, pf in predictions.items():
                    save_pf(
                        pf,
                        self._model_path.data_generated
                        / f"predictions_{self.args.run_type}_{_ts}"
                        / target
                    )

                # --- Step 5+6: Combined parquet + uploads ---
                self._save_combined_forecast(predictions, _ts)

            except Exception as e:
                logger.error(
                    f"Error forecasting {self._model_path.target}: {e}", exc_info=True
                )
                raise ModelForecastingException(
                    f"Forecasting failed: {traceback.format_exc()}",
                    wandb_module=self._wandb_module,
                )
            finally:
                self._wandb_module.finish_run()

    def _save_combined_forecast(self, predictions: Dict[str, Any], ts: str) -> None:
        """Save a combined multi-target parquet and upload to stores.

        Writes one file: ``predictions_{run_type}_{ts}.parquet`` containing
        all targets' ``pred_{target}`` columns. This is the same layout
        the legacy DF path produced, so every downstream reader works
        unchanged.

        Also uploads to the prediction store (views-forecasts) and Appwrite
        if those are enabled.
        """
        import pyarrow.parquet as pq
        from views_pipeline_core.configs.pipeline import PipelineConfig
        from views_pipeline_core.modules.frames.prediction_frame_converter import (
            PredictionFrameConverter,
        )
        from views_pipeline_core.files.utils import handle_single_log_creation

        level = self.configs["level"]
        run_type = self.args.run_type
        model_name = self._model_path.model_name
        data_generated = self._model_path.data_generated

        # Build the combined Arrow table from all PredictionFrames
        converter = PredictionFrameConverter()
        combined_table = converter.to_combined_arrow_table(predictions, level)

        # Write the combined parquet file
        combined_name = f"predictions_{run_type}_{ts}{PipelineConfig.dataframe_format}"
        combined_path = data_generated / combined_name
        data_generated.mkdir(parents=True, exist_ok=True)
        pq.write_table(combined_table, combined_path)
        logger.info(f"Saved combined forecast: {combined_path}")

        # Upload to prediction store (views-forecasts) if enabled
        if self._use_prediction_store:
            try:
                combined_df = combined_table.to_pandas()
                store_name = f"{model_name}_predictions_{run_type}_{ts}"
                combined_df.forecasts.set_run(self._pred_store_name)
                combined_df.forecasts.to_store(name=store_name, overwrite=True)
                logger.info(f"Uploaded to prediction store: {store_name}")
            except Exception as e:
                logger.error(f"Prediction store upload failed: {e}", exc_info=True)

        # Upload to Appwrite if enabled
        if self._datastore is not None:
            try:
                targets = list(predictions.keys())
                self._datastore.upload_data(
                    file=combined_path,
                    filename=combined_name,
                    loa=level,
                    name=model_name,
                    targets=targets,
                    category="forecast",
                    description="",
                    type=self._model_path.target,
                )
                logger.info(f"Uploaded to Appwrite: {combined_name}")
            except Exception as e:
                logger.error(f"Appwrite upload failed: {e}", exc_info=True)

        # Log creation
        handle_single_log_creation(
            model_path=self._model_path,
            config=self.configs,
            train=False,
        )

        # Completion alert
        self._wandb_module.send_alert(
            title=(
                f"Forecasting for {self._model_path.target} "
                f"{model_name} completed successfully."
            ),
            notifications_enabled=self._wandb_notifications,
        )
