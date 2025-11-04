import wandb
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PipelineException(Exception):
    """Base exception for all pipeline errors with auto WandB alerting."""
    
    def __init__(
        self,
        message: str,
        wandb_manager: Optional['WandBModule'] = None,
        alert_level: wandb.AlertLevel = wandb.AlertLevel.ERROR,
    ):
        super().__init__(message)
        self.message = message
        
        # Auto-send WandB alert
        if wandb_manager:
            wandb_manager.send_alert(
                title=self.__class__.__name__,
                text=message,
                level=alert_level,
            )


class ModelTrainingException(PipelineException):
    """Raised when model training fails."""
    pass


class ModelEvaluationException(PipelineException):
    """Raised when model evaluation fails."""
    pass


class ModelForecastingException(PipelineException):
    """Raised when model forecasting fails."""
    pass


class ConfigurationException(PipelineException):
    """Raised when configuration is invalid."""
    pass


class DataFetchException(PipelineException):
    """Raised when data fetching fails."""
    pass


class ValidationException(PipelineException):
    """Raised when validation fails."""
    pass