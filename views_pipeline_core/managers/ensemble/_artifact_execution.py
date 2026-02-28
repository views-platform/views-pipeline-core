"""
Ensemble artifact-execution mixin.

Contains the ensemble orchestration loops (train / evaluate / forecast each
constituent model) and the lower-level helpers that execute individual model
artifacts via shell scripts.
"""

import logging
import subprocess
from typing import List, Dict
from pathlib import Path

import pandas as pd
import polars as pl
import tqdm

from views_pipeline_core.managers.model import (
    ModelPathManager,
    ForecastingModelManager,
)
from views_pipeline_core.cli.args import ForecastingModelArgs
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.exceptions import PipelineException
from views_pipeline_core.modules.dataset.core import SpatioTemporalDataset

logger = logging.getLogger(__name__)


class EnsembleArtifactMixin:
    """Ensemble orchestration loops and individual model-artifact execution."""

    # ============================================================
    # ENSEMBLE ORCHESTRATION
    # ============================================================

    def _train_ensemble(self) -> None:
        """
        Trains all models in the ensemble.
        Uses ``self.args`` and ``self.configs``.
        """
        for model_name in tqdm.tqdm(self.configs["models"], desc="Training ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            self._train_model_artifact(model_name)

    def _evaluate_ensemble(self) -> List[SpatioTemporalDataset]:
        """
        Evaluates the ensemble of models and returns aggregated predictions.
        Uses ``self.args`` and ``self.configs``.

        Returns:
            List[SpatioTemporalDataset]: Aggregated evaluation prediction datasets.
        """
        eval_results: Dict[str, List[SpatioTemporalDataset]] = {}

        for model_name in tqdm.tqdm(self.configs["models"], desc="Evaluating ensemble"):
            tqdm.tqdm.write(f"Current model: {model_name}")
            eval_results[model_name] = self._evaluate_model_artifact(model_name)

        n_outputs = len(next(iter(eval_results.values())))
        aggregated_outputs: List[SpatioTemporalDataset] = []

        tqdm.tqdm.write("Aggregating metrics...")
        for i in range(n_outputs):
            model_dfs_i = {}
            for model_name, dfs in eval_results.items():
                if i >= len(dfs):
                    raise ValueError(
                        f"Model '{model_name}' returned only {len(dfs)} outputs, "
                        f"but at least {i+1} are required for ensemble aggregation. "
                        f"All models must return the same number of outputs."
                    )
                model_dfs_i[model_name] = dfs[i]

            ds_agg = self._get_aggregated_df(
                df_to_aggregate=model_dfs_i,
                aggregation=self.configs["aggregation"],
            )

            aggregated_outputs.append(ds_agg)

        return aggregated_outputs

    def _forecast_ensemble(self) -> SpatioTemporalDataset:
        """
        Generates ensemble forecasts, aggregates results, and optionally
        reconciles predictions.
        Uses ``self.args`` and ``self.configs``.

        Returns:
            SpatioTemporalDataset: The aggregated (and possibly reconciled)
            forecast dataset.
        """
        model_datasets: Dict[str, SpatioTemporalDataset] = {}

        for model_name in tqdm.tqdm(
            self.configs["models"], desc="Forecasting ensemble"
        ):
            tqdm.tqdm.write(f"Current model: {model_name}")
            ds = self._forecast_model_artifact(model_name)
            model_datasets[model_name] = ds

        ds_prediction = self._get_aggregated_df(
            df_to_aggregate=model_datasets,
            aggregation=self.configs["aggregation"],
        )

        # Apply reconciliation if configured
        if self._activate_reconciliation:
            ds_prediction = self._apply_reconciliation(ds_prediction)

        if not isinstance(ds_prediction, SpatioTemporalDataset):
            raise TypeError(
                f"Expected predictions to be a SpatioTemporalDataset, "
                f"got {type(ds_prediction)} instead."
            )

        return ds_prediction

    # ============================================================
    # MODEL ARTIFACT EXECUTION
    # ============================================================

    def _train_model_artifact(self, model_name: str) -> None:
        """
        Trains a single model artifact.

        Args:
            model_name (str): The name of the model to train.
        """
        logger.info(f"Training single model {model_name}...")

        model_path = ModelPathManager(model_name)
        model_args = self._create_model_args(train=True)

        self._execute_shell_script(model_path, model_name, model_args)

    def _evaluate_model_artifact(self, model_name: str) -> List[SpatioTemporalDataset]:
        """
        Evaluate a model artifact by loading or generating predictions.

        Verifies that the constituent model's data and trained artifact
        belong together before loading predictions.

        Args:
            model_name (str): The name of the model to evaluate.

        Returns:
            List[SpatioTemporalDataset]: Prediction datasets for each
            evaluation sequence.
        """
        logger.info(f"Evaluating single model {model_name}...")

        model_path = ModelPathManager(model_name)
        run_type = self.configs["run_type"]
        path_generated = model_path.data_generated
        path_artifact = model_path.get_latest_model_artifact_path(run_type=run_type)

        # Verify constituent model's data/model consistency via its registry
        from views_pipeline_core.modules.artifacts import ArtifactRegistry
        model_registry = ArtifactRegistry(model_path.model_dir)
        if model_registry.count > 0:
            if not model_registry.validate_data_model_match(run_type=run_type):
                raise RuntimeError(
                    f"Data/model artifact mismatch for constituent model "
                    f"{model_name!r} (run_type={run_type!r}). "
                    f"The model was not trained on the current data."
                )
            logger.info(
                f"Constituent model {model_name!r}: data/model match "
                f"verified (run_type={run_type!r}, "
                f"registry entries={model_registry.count})"
            )
        else:
            raise RuntimeError(
                f"Artifact registry for constituent model {model_name!r} "
                f"is empty — cannot verify data/model consistency"
            )

        ts = path_artifact.stem[-15:]
        preds = []

        for sequence_number in range(
            ForecastingModelManager._resolve_evaluation_sequence_number(self._eval_type)
        ):
            name = f"{model_name}_predictions_{run_type}_{ts}_{str(sequence_number).zfill(2)}"
            pred = self._load_or_generate_prediction(
                model_path,
                model_name,
                name,
                path_generated,
                run_type,
                ts,
                sequence_number,
                evaluate=True,
            )
            preds.append(pred)

        return preds

    def _forecast_model_artifact(self, model_name: str) -> SpatioTemporalDataset:
        """
        Forecasts a model artifact and returns the predictions.

        Verifies that the constituent model's data and trained artifact
        belong together before generating forecasts.

        Args:
            model_name (str): The name of the model to forecast.

        Returns:
            SpatioTemporalDataset: Dataset containing the forecasted predictions.
        """
        logger.info(f"Forecasting single model {model_name}...")

        model_path = ModelPathManager(model_name)
        run_type = self.configs["run_type"]
        path_generated = model_path.data_generated
        path_artifact = model_path.get_latest_model_artifact_path(run_type=run_type)

        # Verify constituent model's data/model consistency via its registry
        from views_pipeline_core.modules.artifacts import ArtifactRegistry
        model_registry = ArtifactRegistry(model_path.model_dir)
        if model_registry.count > 0:
            if not model_registry.validate_data_model_match(run_type=run_type):
                raise RuntimeError(
                    f"Data/model artifact mismatch for constituent model "
                    f"{model_name!r} (run_type={run_type!r}). "
                    f"The model was not trained on the current data."
                )
            logger.info(
                f"Constituent model {model_name!r}: data/model match "
                f"verified (run_type={run_type!r}, "
                f"registry entries={model_registry.count})"
            )
        else:
            raise RuntimeError(
                f"Artifact registry for constituent model {model_name!r} "
                f"is empty — cannot verify data/model consistency"
            )

        ts = path_artifact.stem[-15:]
        name = f"{model_name}_predictions_{run_type}_{ts}"

        return self._load_or_generate_prediction(
            model_path, model_name, name, path_generated, run_type, ts, forecast=True
        )

    # ============================================================
    # HELPERS
    # ============================================================

    def _create_model_args(
        self, train: bool = False, evaluate: bool = False, forecast: bool = False
    ) -> ForecastingModelArgs:
        """
        Create a ForecastingModelArgs instance with current settings.

        Args:
            train (bool): Whether to train.
            evaluate (bool): Whether to evaluate.
            forecast (bool): Whether to forecast.

        Returns:
            ForecastingModelArgs: Configured args instance.

        Note:
            If train, the saved flag is set to the value of the saved flag in
            the args.  Check cli validation "if --train or --sweep is not set,
            you should use --saved flag".
        """
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

    def _execute_shell_script(
        self,
        model_path: ModelPathManager,
        model_name: str,
        model_args: ForecastingModelArgs,
    ) -> None:
        """
        Executes a shell script for a model artifact using ForecastingModelArgs.

        Args:
            model_path (ModelPathManager): The path manager for the model.
            model_name (str): The name of the model.
            model_args (ForecastingModelArgs): The arguments for the model
                execution.
        """
        try:
            shell_command = model_args.to_shell_command(model_path)
            logger.info(f"Executing shell command: {' '.join(shell_command)}")
            subprocess.run(shell_command, check=True)
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
        model_path: ModelPathManager,
        model_name: str,
        name: str,
        path_generated: Path,
        run_type: str,
        ts: str,
        sequence_number: int = None,
        evaluate: bool = False,
        forecast: bool = False,
    ) -> SpatioTemporalDataset:
        """
        Load existing prediction or generate new one if not found.

        Returns a SpatioTemporalDataset wrapping the prediction data.
        File-based loading uses lazy scanning; prediction-store loading
        converts from pandas at the boundary.

        Args:
            model_path (ModelPathManager): Path manager for the model.
            model_name (str): Name of the model.
            name (str): Prediction name.
            path_generated (Path): Path to generated data.
            run_type (str): Run type.
            ts (str): Timestamp.
            sequence_number (int, optional): Sequence number for evaluation.
            evaluate (bool): Whether this is for evaluation.
            forecast (bool): Whether this is for forecasting.

        Returns:
            SpatioTemporalDataset: The prediction dataset.
        """
        if self._use_prediction_store:
            try:
                pred = pd.DataFrame.forecasts.read_store(
                    run=self._pred_store_name, name=name
                )
                logger.info(f"Loading existing prediction {name} from prediction store")
                # Prediction store returns pandas — convert at boundary
                if isinstance(pred.index, pd.MultiIndex) or pred.index.name is not None:
                    pred = pred.reset_index()
                return self._coerce_to_dataset(pl.from_pandas(pred))
            except Exception:
                logger.info(
                    f"No existing {run_type} predictions found. Generating new predictions..."
                )
        else:
            seq_suffix = (
                f"_{str(sequence_number).zfill(2)}"
                if sequence_number is not None
                else ""
            )
            file_path = (
                path_generated
                / f"predictions_{run_type}_{ts}{seq_suffix}{PipelineConfig().dataframe_format}"
            )
            if file_path.exists():
                logger.info(f"Loading existing prediction {name} from {file_path}")
                return self._coerce_to_dataset(file_path)
            else:
                logger.info(
                    f"No existing {run_type} predictions found. Generating new predictions..."
                )

        # Generate new predictions
        model_args = self._create_model_args(evaluate=evaluate, forecast=forecast)
        self._execute_shell_script(model_path, model_name, model_args)

        # Load the newly generated prediction
        if self._use_prediction_store:
            pred = pd.DataFrame.forecasts.read_store(
                run=self._pred_store_name, name=name
            )
            # Prediction store returns pandas — convert at boundary
            if isinstance(pred.index, pd.MultiIndex) or pred.index.name is not None:
                pred = pred.reset_index()
            return self._coerce_to_dataset(pl.from_pandas(pred))
        else:
            # Get the latest prediction file (shell script generates with new timestamp)
            prediction_files = model_path._get_generated_predictions_data_file_paths(run_type)
            if not prediction_files:
                raise PipelineException(
                    f"No prediction files found for {model_name} after generation",
                    wandb_module=self._wandb_module
                )
            latest_prediction_file = prediction_files[0]
            logger.info(f"Loading newly generated prediction from {latest_prediction_file}")
            return self._coerce_to_dataset(latest_prediction_file)
