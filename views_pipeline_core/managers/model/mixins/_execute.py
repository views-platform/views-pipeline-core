"""ExecutionMixin — extracted from ForecastingModelManager (C-1 audit decision).

This mixin contains the execute concern methods. It is mixed into
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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from views_pipeline_core.exceptions import (
    DataFetchException,
    ModelEvaluationException,
    ModelTrainingException,
    PipelineException,
)
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.modules.validation.core_config_sniffer import CoreConfigSniffer

logger = logging.getLogger(__name__)


class ExecutionMixin:
    """Mixin providing execute methods for ForecastingModelManager."""

    def execute_single_run(self, args: ForecastingModelArgs) -> None:
        """
        Execute single pipeline run with given arguments.
        
        Main entry point for model pipeline operations. Orchestrates
        data fetching, training, evaluation, forecasting, and reporting
        based on command line arguments.
        
        Execution Flow:
            1. Validate and store arguments
            2. Initialize WandB session
            3. Update configuration
            4. Fetch/load data
            5. Execute requested stages (train/evaluate/forecast/report)
        
        Args:
            args: Validated command line arguments.
                Must be ForecastingModelArgs instance.
        
        Raises:
            ValueError: If args not ForecastingModelArgs instance
            PipelineException: If pipeline execution fails
            ModelTrainingException: If training fails
            ModelEvaluationException: If evaluation fails
            ModelForecastingException: If forecasting fails
        
        Side Effects:
            - Sets self._args
            - Initializes WandB session
            - Creates artifacts/predictions/reports
            - Sends WandB notifications
        
        Example:
            >>> manager = ForecastingModelManager(model_path)
            >>> args = ForecastingModelArgs.parse_args()
            >>> manager.execute_single_run(args)
        
        Note:
            - Typical runtime: Minutes to hours
            - GPU recommended for large models
        """
        if not isinstance(args, ForecastingModelArgs):
            raise ValueError(
                f"args must be an instance of ForecastingModelArgs. Got {type(args)} instead."
            )

        # Store args FIRST before using them
        self._args = args

        # Layer 1: structural pre-condition — fail immediately if partition config
        # is inaccessible, before any side effects (WandB login, data fetching, etc.)
        self._assert_partition_config_accessible(args.run_type)
        CoreConfigSniffer(self.configs, self._partition_dict, target=self._model_path.target).sniff_all(args.run_type)

        # Construct ViewsDataLoader now that config is validated
        self._initialize_data_loader()

        self._wandb_module.login()

        # Now we can use self.args in config_manager
        self._config_manager.update_for_single_run(
            self.args,
            wandb_module=self._wandb_module,
        )

        self._project = f"{self.configs['name']}_{self.args.run_type}"
        self._eval_type = self.args.eval_type
        self._config_manager.add_config({"eval_type": self._eval_type})

        # Fetch data
        self._execute_data_fetching()

        # Execute model tasks
        self._execute_model_tasks()

    def execute_sweep_run(self, args: ForecastingModelArgs) -> None:
        """
        Execute hyperparameter sweep with WandB.
        
        Runs WandB sweep agent for hyperparameter optimization.
        Trains and evaluates models with different configurations.
        
        Args:
            args: Command line arguments.
                Must have sweep=True.
        
        Raises:
            ValueError: If args not ForecastingModelArgs instance
        
        Side Effects:
            - Creates WandB sweep
            - Initializes sweep agent
            - Runs multiple training iterations
        
        Example:
            >>> args = ForecastingModelArgs(
            ...     run_type='calibration',
            ...     sweep=True
            ... )
            >>> manager.execute_sweep_run(args)
        
        Note:
            - Fetches data once, reuses for all iterations
            - Sweep config must be defined in config_sweep.py
        """
        if not isinstance(args, ForecastingModelArgs):
            raise ValueError(
                f"args must be an instance of ForecastingModelArgs. Got {type(args)} instead."
            )
        import wandb

        # Store args FIRST before using them
        self._args = args

        # Layer 1: structural pre-condition — match execute_single_run() contract
        self._assert_partition_config_accessible(args.run_type)
        CoreConfigSniffer(self.configs, self._partition_dict, target=self._model_path.target).sniff_all(args.run_type)

        # Construct ViewsDataLoader now that config is validated
        self._initialize_data_loader()

        self._wandb_module.login()

        self._project = f"{self._config_manager.config_sweep['name']}_sweep"
        self._eval_type = self.args.eval_type
        self._sweep = True

        # Fetch data
        self._execute_data_fetching()

        # Execute sweep
        sweep_id = wandb.sweep(
            self._config_manager.config_sweep,
            project=self._project,
            entity=self._entity,
        )
        wandb.agent(sweep_id, self._execute_model_tasks, entity=self._entity)

    def _execute_model_tasks(self) -> None:
        """
        Execute requested pipeline stages.
        
        Orchestrates training, evaluation, forecasting, and reporting
        based on arguments. Handles both single runs and sweeps.
        
        Internal Use:
            Called by execute_single_run() and execute_sweep_run().
        
        Execution Flow:
            If sweep:
                - Execute sweep training and evaluation
            
            If single run:
                - Train model (if args.train)
                - Evaluate model (if args.evaluate)
                - Generate forecasts (if args.forecast)
                - Create reports (if args.report)
        
        Side Effects:
            - Executes pipeline stages
            - Creates artifacts/predictions
            - Logs to WandB
            - Sends notifications
        
        Note:
            - Logs total runtime at completion
            - All exceptions handled by stage methods
        """
        import time

        start_t = time.time()

        if self._sweep:
            self._execute_model_sweeping()
        else:
            if self.args.train:
                self._execute_model_training()
            if self.args.evaluate:
                self._execute_model_evaluation()
            if self.args.forecast:
                self._execute_model_forecasting()
            if self.args.report and self.args.forecast:
                self._execute_forecast_reporting()
            if self.args.report and self.args.evaluate:
                self._execute_evaluation_reporting()

        end_t = time.time()
        minutes = (end_t - start_t) / 60
        logger.info(f"Done. Runtime: {minutes:.3f} minutes.\n")

