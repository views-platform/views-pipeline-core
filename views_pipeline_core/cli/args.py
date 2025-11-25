from dataclasses import dataclass
from typing import Optional, List, Dict, Type, TypeVar
from pathlib import Path
from abc import ABC, abstractmethod
import sys
import argparse
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='ModelArgs')


@dataclass
class ModelArgs(ABC):
    """
    Base class for model pipeline arguments.
    
    Provides common functionality for argument parsing, validation,
    and conversion to various formats.
    """
    
    def __post_init__(self):
        """Validate arguments after initialization."""
        self._validate()
    
    @classmethod
    def parse_args(cls: Type[T]) -> T:
        """
        Parse command line arguments and create ModelArgs instance.
        
        Returns:
            ModelArgs: Validated arguments dataclass
        """
        parser = cls._create_parser()
        args = parser.parse_args()
        return cls.from_namespace(args)
    
    @classmethod
    @abstractmethod
    def _create_parser(cls) -> argparse.ArgumentParser:
        """
        Create argument parser with model-specific arguments.
        
        Must be implemented by subclasses to define their specific arguments.
        
        Returns:
            argparse.ArgumentParser: Configured argument parser
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_namespace(cls: Type[T], args: argparse.Namespace) -> T:
        """
        Create ModelArgs instance from argparse Namespace.
        
        Args:
            args: argparse Namespace object
            
        Returns:
            ModelArgs: Validated arguments instance
        """
        pass
    
    @abstractmethod
    def _validate(self) -> None:
        """
        Validate argument combinations and constraints.
        
        Must be implemented by subclasses to define their validation logic.
        Should raise errors for invalid argument combinations.
        """
        pass
    
    @abstractmethod
    def to_shell_command(
        self,
        model_path,
        script_name: str = "run.sh"
    ) -> List[str]:
        """
        Generate shell command from arguments.
        
        Args:
            model_path: Path manager for the model
            script_name (str): Name of the script to execute
            
        Returns:
            List[str]: Shell command as list of strings
        """
        pass
    
    @abstractmethod
    def get_dict(self) -> Dict:
        """
        Get arguments as dictionary.
        
        Returns:
            Dict: Arguments as key-value pairs
        """
        pass
    
    @staticmethod
    def _exit_with_error(*messages) -> None:
        """
        Print error messages and exit.
        
        Args:
            *messages: Error messages to print
        """
        for msg in messages:
            print(msg)
        sys.exit(1)
    
    def __str__(self) -> str:
        """String representation of arguments."""
        return f"{self.__class__.__name__}({self.get_dict()})"
    
    def __repr__(self) -> str:
        """Detailed string representation of arguments."""
        args_str = ", ".join(f"{k}={v!r}" for k, v in self.get_dict().items())
        return f"{self.__class__.__name__}({args_str})"


@dataclass
class ForecastingModelArgs(ModelArgs):
    """
    Dataclass for storing and validating forecasting model pipeline arguments.
    
    Attributes:
        run_type (str): Type of run (calibration, validation, forecasting)
        sweep (bool): Whether to run as part of a sweep
        train (bool): Whether to train the model
        evaluate (bool): Whether to evaluate the model
        forecast (bool): Whether to generate forecasts
        prediction_store (bool): Whether to use prediction store
        artifact_name (Optional[str]): Name of artifact to use
        saved (bool): Whether to use locally stored data
        override_timestep (Optional[int]): Override timestep value
        drift_self_test (bool): Enable drift detection self-test
        eval_type (str): Type of evaluation
        report (bool): Whether to generate report
        update_viewser (bool): Whether to update viewser dataframe
        wandb_notifications (bool): Whether to enable W&B notifications
        monthly (bool): Shorthand for monthly production runs
    """
    
    run_type: str = "calibration"
    sweep: bool = False
    train: bool = False
    evaluate: bool = False
    forecast: bool = False
    prediction_store: bool = False
    artifact_name: Optional[str] = None
    saved: bool = False
    override_timestep: Optional[int] = None
    drift_self_test: bool = False
    eval_type: str = "standard"
    report: bool = False
    update_viewser: bool = False
    wandb_notifications: bool = False
    monthly: bool = False
    
    @classmethod
    def _create_parser(cls) -> argparse.ArgumentParser:
        """
        Create argument parser for forecasting model arguments.
        
        Returns:
            argparse.ArgumentParser: Configured argument parser
        """
        parser = argparse.ArgumentParser(
            description="Run forecasting model pipeline with specified run type."
        )

        parser.add_argument(
            "-r",
            "--run_type",
            choices=["calibration", "validation", "forecasting"],
            type=str,
            default="calibration",
            help="Choose the run type for the model: calibration, validation, or forecasting. Default is calibration. "
            "Note: If --sweep is flagged, --run_type must be calibration.",
        )

        parser.add_argument(
            "-s",
            "--sweep",
            action="store_true",
            help="Set flag to run the model pipeline as part of a sweep. No explicit flag means no sweep. "
            "Note: If --sweep is flagged, --run_type must be calibration, and both training and evaluation is automatically implied.",
        )

        parser.add_argument(
            "-t",
            "--train",
            action="store_true",
            help="Flag to indicate if a new model should be trained. "
            "Note: If --sweep is flagged, --train will also automatically be flagged.",
        )

        parser.add_argument(
            "-e",
            "--evaluate",
            action="store_true",
            help="Flag to indicate if the model should be evaluated. "
            "Note: If --sweep is specified, --evaluate will also automatically be flagged. "
            "Cannot be used with --run_type forecasting.",
        )

        parser.add_argument(
            "-f",
            "--forecast",
            action="store_true",
            help="Flag to indicate if the model should produce predictions. "
            "Note: If --sweep is specified, --forecast will also automatically be flagged. "
            "Can only be used with --run_type forecasting.",
        )

        parser.add_argument(
            "-p",
            "--prediction_store",
            action="store_true",
            help="Flag to indicate if the model should use the prediction store.",
        )

        parser.add_argument(
            "-a",
            "--artifact_name",
            type=str,
            help="Specify the name of the model artifact to be used for evaluation. "
            "The file extension will be added in the main and fit with the specific model algorithm. "
            "The artifact name should be in the format: <run_type>_model_<timestamp>.pt. "
            "where <run_type> is calibration, validation, or forecasting, and <timestamp> is in the format YMD_HMS. "
            "If not provided, the latest artifact will be used by default.",
        )

        parser.add_argument(
            "-sa", "--saved", action="store_true", help="Use locally stored data"
        )

        parser.add_argument(
            "-o",
            "--override_timestep",
            help="Override use of current time (year/month/week/day depending on your LoA)",
            type=int,
        )

        parser.add_argument(
            "-dd",
            "--drift_self_test",
            action="store_true",
            default=False,
            help="Enable drift-detection self_test at data-fetch",
        )

        parser.add_argument(
            "-et",
            "--eval_type",
            type=str,
            default="standard",
            help="Type of evaluation to be performed",
        )

        parser.add_argument(
            "-re", "--report", action="store_true", help="Generate evaluation or forecast report."
        )

        parser.add_argument(
            "-u",
            "--update_viewser",
            action="store_true",
            help="Update the viewser dataframe for a set of months where viewser returns only zeros.",
        )

        parser.add_argument(
            "-wn",
            "--wandb_notifications",
            action="store_true",
            help="Enable Weights & Biases notifications.",
        )

        parser.add_argument(
            "-m",
            "--monthly",
            action="store_true",
            help="Shorthand flag for monthly production runs. "
            "Automatically sets: --run_type forecasting, --train, --forecast, --report, --prediction_store, --wandb_notifications.",
        )

        return parser
        
    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> 'ForecastingModelArgs':
        """
        Create ForecastingModelArgs from argparse Namespace.
        
        Args:
            args: argparse Namespace object
            
        Returns:
            ForecastingModelArgs: Validated arguments dataclass
        """
        return cls(
            run_type=args.run_type,
            sweep=args.sweep,
            train=args.train,
            evaluate=args.evaluate,
            forecast=args.forecast,
            prediction_store=args.prediction_store,
            artifact_name=args.artifact_name,
            saved=args.saved,
            override_timestep=args.override_timestep,
            drift_self_test=args.drift_self_test,
            eval_type=args.eval_type,
            report=args.report,
            update_viewser=args.update_viewser,
            wandb_notifications=args.wandb_notifications,
            monthly=args.monthly,
        )
    
    def _validate(self) -> None:
        """Validate forecasting model argument combinations."""
        if self.monthly:
            if self.sweep:
                self._exit_with_error(
                    "Error: --monthly flag cannot be used with --sweep flag.",
                    "To fix: Remove --sweep flag when using --monthly."
                )
            
            if self.evaluate:
                self._exit_with_error(
                    "Error: --monthly flag cannot be used with --evaluate flag "
                    "(monthly runs do forecasting, not evaluation).",
                    "To fix: Remove --evaluate flag when using --monthly."
                )
            
            # Auto-set flags for monthly runs
            self.run_type = "forecasting"
            self.train = True
            self.forecast = True
            self.report = True
            self.prediction_store = True
            self.wandb_notifications = True

        # Allow report with calibration if evaluate is set
        if self.report and self.run_type == "calibration" and not self.evaluate:
            self._exit_with_error(
                "Error: --report with --run_type calibration requires --evaluate to be set.",
                "To fix: Add --evaluate flag when using --report with calibration runs."
            )

        if self.report and not (self.evaluate or self.forecast):
            self._exit_with_error(
                "Error: --report requires either --evaluate or --forecast to be set.",
                "To fix: Add --evaluate or --forecast flag when using --report."
            )

        if self.sweep and self.run_type != "calibration":
            self._exit_with_error(
                "Error: Sweep runs must have --run_type set to 'calibration'.",
                "To fix: Use --run_type calibration when --sweep is flagged."
            )

        if self.sweep and (self.train or self.evaluate):
            self._exit_with_error(
                "Error: Sweep runs cannot have --train or --evaluate flags set. "
                "Sweep does training and evaluation by default.",
                "To fix: Remove --train, or --evaluate flags when --sweep is flagged."
            )

        if self.sweep and self.forecast:
            self._exit_with_error(
                "Error: Sweep runs cannot have --forecast flag set because sweep doesn't do forecasting.",
                "To fix: Remove --forecast flag when --sweep is flagged."
            )

        if self.evaluate and self.run_type == "forecasting":
            self._exit_with_error(
                "Error: Forecasting runs cannot evaluate.",
                "To fix: Remove --evaluate flag when --run_type is 'forecasting'."
            )

        if (
            self.run_type in ["calibration", "validation", "forecasting"]
            and not self.train
            and not self.evaluate
            and not self.forecast
            and not self.sweep
            and not self.report
        ):
            self._exit_with_error(
                f"Error: Run type is {self.run_type} but neither --train, --evaluate, "
                "nor --sweep flag is set. Nothing to do...",
                "To fix: Add --train and/or --evaluate flag. Or use --sweep to run both "
                "training and evaluation in a WandB sweep loop."
            )

        if self.train and self.artifact_name:
            self._exit_with_error(
                "Error: Both --train and --artifact_name flags are set.",
                "To fix: Remove --artifact_name if --train is set, or vice versa."
            )

        if self.forecast and self.run_type != "forecasting":
            self._exit_with_error(
                "Error: --forecast flag can only be used with --run_type forecasting.",
                "To fix: Set --run_type to forecasting if --forecast is flagged."
            )

        if (not self.train and not self.sweep) and not self.saved:
            self._exit_with_error(
                "Error: if --train or --sweep is not set, you should only use --saved flag.",
                "To fix: Add --train or --sweep or --saved flag."
            )

        if self.eval_type not in ["standard", "long", "complete", "live"]:
            self._exit_with_error(
                "Error: --eval_type should be one of 'standard', 'long', 'complete', or 'live'.",
                "To fix: Set --eval_type to one of the above options."
            )

        if self.prediction_store and not self.forecast:
            self._exit_with_error(
                "Error: --prediction_store flag can only be used with --forecast flag.",
                "To fix: Set --forecast flag if --prediction_store is flagged."
            )
    
    def to_shell_command(
        self,
        model_path,
        script_name: str = "run.sh"
    ) -> List[str]:
        """
        Generate shell command from forecasting model arguments.
        
        Args:
            model_path: Path manager for the model
            script_name (str): Name of the script to execute
            
        Returns:
            List[str]: Shell command as list of strings
        """
        shell_command = [str(model_path.model_dir / script_name)]
        
        # Add run type
        shell_command.extend(["--run_type", self.run_type])
        
        # Add boolean flags
        if self.train:
            shell_command.append("--train")
        if self.evaluate:
            shell_command.append("--evaluate")
        if self.forecast:
            shell_command.append("--forecast")
        if self.saved:
            shell_command.append("--saved")
        if self.update_viewser:
            shell_command.append("--update_viewser")
        if self.prediction_store:
            shell_command.append("--prediction_store")
        if self.wandb_notifications:
            shell_command.append("--wandb_notifications")
        if self.sweep:
            shell_command.append("--sweep")
        if self.drift_self_test:
            shell_command.append("--drift_self_test")
        if self.report:
            shell_command.append("--report")
        if self.monthly:
            shell_command.append("--monthly")
        
        # Add evaluation type
        shell_command.extend(["--eval_type", self.eval_type])
        
        # Add optional parameters
        if self.override_timestep is not None:
            shell_command.extend(["--override_timestep", str(self.override_timestep)])
        
        if self.artifact_name is not None:
            shell_command.extend(["--artifact_name", self.artifact_name])
        
        return shell_command
    
    def get_dict(self) -> Dict:
        """
        Get forecasting model arguments as dictionary.
        
        Returns:
            Dict: Arguments as key-value pairs
        """
        return {
            "run_type": self.run_type,
            "sweep": self.sweep,
            "train": self.train,
            "evaluate": self.evaluate,
            "forecast": self.forecast,
            "prediction_store": self.prediction_store,
            "artifact_name": self.artifact_name,
            "saved": self.saved,
            "override_timestep": self.override_timestep,
            "drift_self_test": self.drift_self_test,
            "eval_type": self.eval_type,
            "report": self.report,
            "update_viewser": self.update_viewser,
            "wandb_notifications": self.wandb_notifications,
            "monthly": self.monthly,
        }


# @dataclass
# class PreprocessorModelArgs(ModelArgs):
#     """
#     Dataclass for storing and validating preprocessor pipeline arguments.
    
#     Attributes:
#         process (bool): Whether to run preprocessing
#         saved (bool): Whether to use locally stored data
#         output_format (str): Format for output data (parquet, csv, etc.)
#         validate_output (bool): Whether to validate preprocessed output
#         wandb_notifications (bool): Whether to enable W&B notifications
#     """
    
#     process: bool = False
#     saved: bool = False
#     output_format: str = "parquet"
#     validate_output: bool = True
#     wandb_notifications: bool = False
    
#     @classmethod
#     def _create_parser(cls) -> argparse.ArgumentParser:
#         """
#         Create argument parser for preprocessor arguments.
        
#         Returns:
#             argparse.ArgumentParser: Configured argument parser
#         """
#         parser = argparse.ArgumentParser(
#             description="Run preprocessor pipeline."
#         )

#         parser.add_argument(
#             "-p",
#             "--process",
#             action="store_true",
#             help="Flag to run the preprocessing pipeline.",
#         )

#         parser.add_argument(
#             "-sa",
#             "--saved",
#             action="store_true",
#             help="Use locally stored data",
#         )

#         parser.add_argument(
#             "-of",
#             "--output_format",
#             type=str,
#             default="parquet",
#             choices=["parquet", "csv", "pickle"],
#             help="Output format for preprocessed data. Default is parquet.",
#         )

#         parser.add_argument(
#             "-vo",
#             "--validate_output",
#             action="store_true",
#             default=True,
#             help="Validate preprocessed output data.",
#         )

#         parser.add_argument(
#             "-wn",
#             "--wandb_notifications",
#             action="store_true",
#             help="Enable Weights & Biases notifications.",
#         )

#         return parser
    
#     @classmethod
#     def from_namespace(cls, args: argparse.Namespace) -> 'PreprocessorModelArgs':
#         """
#         Create PreprocessorModelArgs from argparse Namespace.
        
#         Args:
#             args: argparse Namespace object
            
#         Returns:
#             PreprocessorModelArgs: Validated arguments dataclass
#         """
#         return cls(
#             process=args.process,
#             saved=args.saved,
#             output_format=args.output_format,
#             validate_output=args.validate_output,
#             wandb_notifications=args.wandb_notifications,
#         )
    
#     def _validate(self) -> None:
#         """Validate preprocessor argument combinations."""
#         if not self.process and not self.saved:
#             self._exit_with_error(
#                 "Error: Neither --process nor --saved flag is set. Nothing to do...",
#                 "To fix: Add --process to run preprocessing or --saved to use existing data."
#             )
        
#         if self.output_format not in ["parquet", "csv", "pickle"]:
#             self._exit_with_error(
#                 "Error: --output_format must be one of 'parquet', 'csv', or 'pickle'.",
#                 "To fix: Set --output_format to one of the valid options."
#             )
    
#     def to_shell_command(
#         self,
#         model_path,
#         script_name: str = "run.sh"
#     ) -> List[str]:
#         """
#         Generate shell command from preprocessor arguments.
        
#         Args:
#             model_path: Path manager for the model
#             script_name (str): Name of the script to execute
            
#         Returns:
#             List[str]: Shell command as list of strings
#         """
#         shell_command = [str(model_path.model_dir / script_name)]
        
#         # Add boolean flags
#         if self.process:
#             shell_command.append("--process")
#         if self.saved:
#             shell_command.append("--saved")
#         if self.validate_output:
#             shell_command.append("--validate_output")
#         if self.wandb_notifications:
#             shell_command.append("--wandb_notifications")
        
#         # Add output format
#         shell_command.extend(["--output_format", self.output_format])
        
#         return shell_command
    
#     def get_dict(self) -> Dict:
#         """
#         Get preprocessor arguments as dictionary.
        
#         Returns:
#             Dict: Arguments as key-value pairs
#         """
#         return {
#             "process": self.process,
#             "saved": self.saved,
#             "output_format": self.output_format,
#             "validate_output": self.validate_output,
#             "wandb_notifications": self.wandb_notifications,
#         }