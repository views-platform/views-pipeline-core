"""ConstituentMixin — model artifact execution for ensemble members.

Extracted from EnsembleManager. Handles training, evaluating, and
forecasting individual constituent models via shell script delegation.
"""
from __future__ import annotations

import logging
from typing import List, Dict
import pandas as pd
import tqdm

from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.cli.args import ForecastingModelArgs

logger = logging.getLogger(__name__)


class ConstituentMixin:
    """Mixin providing constituent model execution methods for EnsembleManager."""

    def _train_ensemble(self) -> None:
        """Train all models in the ensemble."""

        for model_name in tqdm.tqdm(self.configs["models"], desc="Training ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            self._train_model_artifact(model_name)

    def _train_model_artifact(self, model_name: str) -> None:
        """Train a single constituent model via shell script."""
        logger.info(f"Training single model {model_name}...")
        model_path = ModelPathManager(model_name)
        model_args = self._create_model_args(train=True)
        self._execute_shell_script(model_path, model_name, model_args)

    def _evaluate_model_artifact(self, model_name: str) -> List[pd.DataFrame]:
        """Load or generate evaluation predictions for a constituent model."""
        from views_pipeline_core.managers.model import ForecastingModelManager

        logger.info(f"Evaluating single model {model_name}...")
        model_path = ModelPathManager(model_name)
        run_type = self.configs["run_type"]
        path_generated = model_path.data_generated
        path_artifact = model_path.resolve_artifact_path(run_type=run_type)
        ts = path_artifact.stem[-15:]
        preds = []

        for sequence_number in range(
            ForecastingModelManager._resolve_evaluation_sequence_number(self._eval_type)
        ):
            name = f"{model_name}_predictions_{run_type}_{ts}_{str(sequence_number).zfill(2)}"
            pred = self._load_or_generate_prediction(
                model_path, model_name, name, path_generated,
                run_type, ts, sequence_number, evaluate=True,
            )
            months = pred.index.get_level_values("month_id")
            logger.info(
                f"LOADED PRED | model={model_name} | seq={sequence_number:02d} | "
                f"month_id=[{months.min()}, {months.max()}] | n={len(pred)}"
            )
            preds.append(pred)
        return preds

    def _forecast_model_artifact(self, model_name: str) -> pd.DataFrame:
        """Load or generate forecast predictions for a constituent model."""
        logger.info(f"Forecasting single model {model_name}...")
        model_path = ModelPathManager(model_name)
        run_type = self.configs["run_type"]
        path_generated = model_path.data_generated
        path_artifact = model_path.resolve_artifact_path(run_type=run_type)
        ts = path_artifact.stem[-15:]
        name = f"{model_name}_predictions_{run_type}_{ts}"
        return self._load_or_generate_prediction(
            model_path, model_name, name, path_generated, run_type, ts, forecast=True
        )

    def _create_model_args(
        self, train: bool = False, evaluate: bool = False, forecast: bool = False
    ) -> ForecastingModelArgs:
        """Create ForecastingModelArgs for a constituent model run."""
        saved = self.args.saved if train else True
        use_prediction_store = (
            True if forecast and self._use_prediction_store else False
        )
        return ForecastingModelArgs(
            run_type=self.args.run_type,
            train=train,
            evaluate=evaluate,
            forecast=forecast,
            saved=saved,
            eval_type=self.args.eval_type,
            update_viewser=self.args.update_viewser,
            prediction_store=use_prediction_store,
            wandb_notifications=self._wandb_notifications,
            override_timestep=self.args.override_timestep,
        )

    def _execute_shell_script(self, model_path, model_name, model_args) -> None:
        """Delegate to the shared subprocess runner."""
        from views_pipeline_core.modules.ensemble.subprocess_runner import (
            execute_model_subprocess,
        )
        execute_model_subprocess(
            model_path=model_path,
            model_name=model_name,
            model_args=model_args,
            wandb_module=self._wandb_module,
        )
