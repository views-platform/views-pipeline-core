from typing import Dict, Optional, Any
from pathlib import Path
import wandb
import logging
from dataclasses import asdict

logger = logging.getLogger(__name__)


class WandBModule:
    """
    Centralized manager for all Weights & Biases operations.
    Handles initialization, logging, alerts, and artifact management.
    """

    def __init__(
        self,
        entity: str,
        notifications_enabled: bool = False,
        models_path: Optional[Path] = None,
    ):
        self.entity = entity
        self.notifications_enabled = notifications_enabled
        self.models_path = models_path
        self._active_run = None

    def initialize_run(
        self,
        project: str,
        config: Dict,
        job_type: str,
        name: Optional[str] = None,
    ) -> wandb.sdk.wandb_run.Run:
        """Initialize a WandB run with proper configuration."""
        self._active_run = wandb.init(
            project=project,
            entity=self.entity,
            config=config,
            job_type=job_type,
            name=name,
        )
        self._add_custom_metrics()
        return self._active_run

    def _add_custom_metrics(self) -> None:
        """Define custom WandB metrics for structured logging."""
        wandb.define_metric("step-wise/step")
        wandb.define_metric("step-wise/*", step_metric="step-wise/step")
        wandb.define_metric("month-wise/month")
        wandb.define_metric("month-wise/*", step_metric="month-wise/month")
        wandb.define_metric("time-series-wise/time-series")
        wandb.define_metric("time-series-wise/*", step_metric="time-series-wise/time-series")

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to WandB."""
        if self._active_run:
            wandb.log(metrics)

    def log_evaluation_results(
        self,
        step_wise: Dict,
        month_wise: Dict,
        time_series_wise: Dict,
        conflict_type: str,
    ) -> None:
        """Log structured evaluation results."""
        from views_pipeline_core.modules.wandb import log_wandb_log_dict
        
        log_wandb_log_dict(
            step_wise,
            time_series_wise,
            month_wise,
            conflict_type,
        )

    @staticmethod
    def send_alert(
        title: str,
        text: str = "",
        level: wandb.AlertLevel = wandb.AlertLevel.INFO,
        models_path: Optional[Path] = None,
        notifications_enabled: bool = False,
    ) -> None:
        """Send a WandB alert with path redaction."""
        if not notifications_enabled or not wandb.run:
            return

        try:
            # Redact sensitive paths
            if models_path:
                text = str(text).replace(str(models_path), "[REDACTED]")
            
            wandb.alert(title=title, text=text, level=level)
            logger.info(f"WandB alert sent: {title}")
        except Exception as e:
            logger.error(f"Failed to send WandB alert: {e}")

    def log_artifact(
        self,
        artifact_path: Path,
        artifact_name: str,
        artifact_type: str,
        description: str = "",
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log an artifact to WandB."""
        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                description=description,
                metadata=metadata or {},
            )
            artifact.add_file(str(artifact_path))
            wandb.run.log_artifact(artifact)
            logger.info(f"Artifact '{artifact_name}' logged successfully")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
            raise

    def finish_run(self) -> None:
        """Finish the current WandB run."""
        if self._active_run:
            wandb.finish()
            self._active_run = None

    def save(self, path: str) -> None:
        """Save a file to WandB."""
        wandb.save(path)

    def log(self, data: Any) -> None:
        """Log data to WandB."""
        wandb.log(data)

    @staticmethod
    def login() -> None:
        """Login to WandB."""
        wandb.login()