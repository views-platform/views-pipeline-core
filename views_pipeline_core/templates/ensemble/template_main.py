from typing import Dict
from views_pipeline_core.templates.utils import save_script
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate(script_dir: Path) -> bool:
    """
    Generates a script that sets up the project paths, parses command-line arguments,
    sets up logging, and executes a single model run.

    Parameters:
        script_dir (Path):
            The directory where the generated script will be saved.
            This should be a valid writable path.

    Returns:
        bool:
            True if the script was written and compiled successfully, False otherwise.
    """

    code = """import wandb
    import warnings
    from pathlib import Path
    from views_pipeline_core.cli.utils import parse_args, validate_arguments
    from views_pipeline_core.managers.log import LoggingManager
    from views_pipeline_core.managers.ensemble import EnsemblePathManager, EnsembleManager

    warnings.filterwarnings("ignore")

    try:
        ensemble_path = EnsemblePathManager(Path(__file__))
        logger = LoggingManager(ensemble_path).get_logger()
    except FileNotFoundError as fnf_error:
        raise RuntimeError(
            f"File not found: {fnf_error}. Check the file path and try again."
        )
    except PermissionError as perm_error:
        raise RuntimeError(
            f"Permission denied: {perm_error}. Check your permissions and try again."
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")

    if __name__ == "__main__":
    wandb.login()
    args = parse_args()
    validate_arguments(args)

    EnsembleManager(ensemble_path=ensemble_path).execute_single_run(args)
    """
    return save_script(script_dir, code)
