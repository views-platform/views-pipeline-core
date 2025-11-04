from typing import Dict, Optional
import logging
from datetime import datetime
from views_pipeline_core.models.check import validate_config
from views_pipeline_core.exceptions import ConfigurationException
from views_pipeline_core.cli.args import ForecastingModelArgs

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Manages configuration loading, validation, and updates.
    Centralizes all configuration-related logic.
    """

    def __init__(
        self,
        config_hyperparameters: Dict,
        config_deployment: Dict,
        config_meta: Dict,
        partition_dict: Optional[Dict] = None,
        config_sweep: Optional[Dict] = None,
    ):
        self.config_hyperparameters = config_hyperparameters or {}
        self.config_deployment = config_deployment or {}
        self.config_meta = config_meta or {}
        self.partition_dict = partition_dict or {}
        self.config_sweep = config_sweep
        self._runtime_config = {}
        
        # Add timestamp at initialization
        self._runtime_config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_combined_config(self) -> Dict:
        """Get the combined configuration from all sources."""
        config = {}
        
        # Merge configurations in order of priority
        if self.partition_dict:
            config.update(self.partition_dict)
        if self.config_hyperparameters:
            config.update(self.config_hyperparameters)
        if self.config_deployment:
            config.update(self.config_deployment)
        if self.config_meta:
            config.update(self.config_meta)
        if self._runtime_config:
            config.update(self._runtime_config)
        
        return config
    
    def add_config(self, config: Dict) -> None:
        """Add runtime configuration."""
        self._runtime_config.update(config)
    
    def update_for_single_run(
        self,
        args: ForecastingModelArgs,
        wandb_manager: Optional['WandBModule'] = None,
    ) -> None:
        """
        Update configuration for a single run with validated arguments.
        
        Args:
            args (ForecastingModelArgs): Validated pipeline arguments
            wandb_manager (Optional[WandBModule]): WandB manager for error reporting
            
        Raises:
            ConfigurationException: If configuration validation fails
        """
        # Add run-specific parameters from args to runtime config
        self._runtime_config["run_type"] = args.run_type
        self._runtime_config["eval_type"] = args.eval_type
        self._runtime_config["sweep"] = args.sweep

        # Handle override timestep
        if args.override_timestep is not None:
            self._apply_timestep_override(args)

        # Validate configuration
        try:
            validate_config(self.get_combined_config())
        except Exception as e:
            raise ConfigurationException(
                f"Configuration validation failed: {e}",
                wandb_manager=wandb_manager,
            )

    def _apply_timestep_override(self, args: ForecastingModelArgs) -> None:
        """
        Apply timestep override to partition configuration.
        
        Args:
            args (ForecastingModelArgs): Arguments containing override timestep
        """
        config = self.get_combined_config()
        
        if "steps" not in config:
            logger.warning("No 'steps' found in config. Skipping timestep override.")
            return
            
        self._runtime_config["forecasting"] = {
            "train": (121, args.override_timestep),
            "test": (
                args.override_timestep + 1,
                args.override_timestep + 1 + len(config["steps"])
            ),
        }
        
        logger.info(
            f"Applied timestep override: train=(121, {args.override_timestep}), "
            f"test=({args.override_timestep + 1}, {args.override_timestep + 1 + len(config['steps'])})"
        )

    def update_for_sweep_run(
        self,
        wandb_config: Dict,
        args: ForecastingModelArgs,
        wandb_manager: Optional['WandBModule'] = None,
    ) -> None:
        """
        Update configuration for a sweep run.
        
        Args:
            wandb_config (Dict): Configuration from WandB sweep
            args (ForecastingModelArgs): Validated pipeline arguments
            wandb_manager (Optional[WandBModule]): WandB manager for error reporting
            
        Raises:
            ConfigurationException: If configuration validation fails
        """
        # Override with wandb sweep parameters
        self._runtime_config.update(wandb_config)
        
        # Add run-specific parameters from args
        self._runtime_config["run_type"] = args.run_type
        self._runtime_config["eval_type"] = args.eval_type
        self._runtime_config["sweep"] = args.sweep

        # Validate configuration
        try:
            validate_config(self.get_combined_config())
        except Exception as e:
            raise ConfigurationException(
                f"Sweep configuration validation failed: {e}",
                wandb_manager=wandb_manager,
            )