import wandb
import pandas as pd
import joblib
import tempfile
from pathlib import Path
from typing import Optional
import sys
import traceback
import logging

logger = logging.getLogger(__name__)

class WandbManager:
    """Handles all Weights & Biases (wandb) integration and operations."""
    
    def __init__(self, entity: str, project: str, config: dict, job_type: str, 
                 notifications: bool = True, model_path = None):
        """
        Initialize WandbManager.
        
        Args:
            entity: Wandb entity name
            project: Project name
            config: Configuration dictionary
            job_type: Type of job (train, evaluate, etc.)
            notifications: Enable wandb notifications
            model_path: ModelPathManager instance
        """
        self.entity = entity
        self.project = project
        self.config = config
        self.job_type = job_type
        self.notifications = notifications
        self.model_path = model_path
        self.run = None
        self._active_run = False
        self._global_handler = None
        
    def __enter__(self):
        """Context manager entry - initialize wandb run."""
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=self.config,
            job_type=self.job_type
        )
        self.add_standard_metrics()
        self._active_run = True
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - finish wandb run and handle errors."""
        if exc_type:
            self.send_alert(
                title=f"{self.job_type.capitalize()} Error",
                text=f"Error during {self.job_type}: {exc_value}\n{traceback.format_exc()}",
                level=wandb.AlertLevel.ERROR
            )
        wandb.finish()
        self._active_run = False
        return False
    
    def register_global_exception_handler(self):
        """Register a global exception handler to catch unhandled exceptions."""
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                return  # Don't handle KeyboardInterrupt
                
            self.send_alert(
                title=f"Unhandled {exc_type.__name__}",
                text=f"Unhandled exception occurred:\n\n{exc_value}\n\nTraceback:\n{''.join(traceback.format_tb(exc_traceback))}",
                level=wandb.AlertLevel.ERROR
            )
            # Call default excepthook
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            
        self._global_handler = handle_exception
        sys.excepthook = handle_exception
        
    def unregister_global_exception_handler(self):
        """Restore default exception handling."""
        if self._global_handler:
            sys.excepthook = sys.__excepthook__
    
    def ensure_active(self):
        """Ensure we have an active run, create one if needed."""
        if not self._active_run:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                config=self.config,
                job_type=self.job_type
            )
            self.add_standard_metrics()
            self._active_run = True
            self.register_global_exception_handler()
        
    def add_standard_metrics(self):
        """Define standard metrics for wandb tracking."""
        wandb.define_metric("step-wise/step")
        wandb.define_metric("step-wise/*", step_metric="step-wise/step")

        wandb.define_metric("month-wise/month")
        wandb.define_metric("month-wise/*", step_metric="month-wise/month")

        wandb.define_metric("time-series-wise/time-series")
        wandb.define_metric(
            "time-series-wise/*", step_metric="time-series-wise/time-series"
        )
            
    def send_alert(self, title: str, text: str, level: wandb.AlertLevel = None):
        """Send alert through wandb if notifications are enabled."""
        if self.notifications:
            level = level or wandb.AlertLevel.INFO
            try:
                self.ensure_active()
                text = str(text).replace(str(self.model_path.models), "[REDACTED]")
                wandb.alert(title=title, text=text, level=level)
            except Exception as e:
                logger.error(f"Failed to send Wandb alert: {e}")
            
    def log_artifact(self, name: str, type: str, file_path: Path, 
                    description: str = "", metadata: dict = None):
        """Log a file as wandb artifact."""
        self.ensure_active()
        artifact = wandb.Artifact(
            name=name,
            type=type,
            description=description,
            metadata=metadata or {}
        )
        artifact.add_file(str(file_path))
        self.run.log_artifact(artifact)
        
    def log_dataframe(self, name: str, df: pd.DataFrame, description: str = ""):
        """Log a pandas DataFrame as wandb artifact."""
        self.ensure_active()
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name)
            self.log_artifact(
                name=name,
                type="data",
                file_path=Path(tmp.name),
                description=description
            )
            
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics to wandb."""
        self.ensure_active()
        self.run.log(metrics, step=step)
        
    def log_summary(self, summary: dict):
        """Update wandb summary metrics."""
        self.ensure_active()
        for key, value in summary.items():
            self.run.summary[key] = value
            
    def log_table(self, table_name: str, dataframe: pd.DataFrame):
        """Log a pandas DataFrame as wandb table."""
        self.ensure_active()
        table = wandb.Table(dataframe=dataframe)
        self.log_metrics({table_name: table})
        
    def log_evaluation_metrics(self, step_metrics: pd.DataFrame, 
                             ts_metrics: pd.DataFrame, 
                             month_metrics: pd.DataFrame):
        """Log evaluation metrics in standardized format."""
        self.ensure_active()
        self.log_table("evaluation_metrics_step", step_metrics)
        self.log_table("evaluation_metrics_ts", ts_metrics)
        self.log_table("evaluation_metrics_month", month_metrics)
        
    def log_model(self, model: any, name: str, metadata: dict = None):
        """Log model as wandb artifact."""
        self.ensure_active()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            joblib.dump(model, tmp.name)
            self.log_artifact(
                name=name,
                type="model",
                file_path=Path(tmp.name),
                metadata=metadata or {},
                description=f"{self.config['name']} model artifact"
            )
            
    def log_config(self, config: dict):
        """Log configuration dictionary to wandb."""
        self.ensure_active()
        self.run.config.update(config)
        
    def save_file(self, file_path: Path):
        """Save file to wandb."""
        self.ensure_active()
        wandb.save(str(file_path))
        
    @classmethod
    def create_sweep(cls, sweep_config: dict, project: str, entity: str) -> str:
        """Create a new wandb sweep."""
        return wandb.sweep(sweep_config, project=project, entity=entity)
    
    @classmethod
    def run_agent(cls, sweep_id: str, function: callable, entity: str, count: int = None):
        """Run wandb agent for a sweep."""
        wandb.agent(sweep_id, function=function, entity=entity, count=count)