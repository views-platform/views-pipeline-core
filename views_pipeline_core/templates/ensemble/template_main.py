from views_pipeline_core.templates.utils import save_python_script
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate(script_path: Path) -> bool:
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
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers.ensemble import EnsemblePathManager, EnsembleManager
from views_pipeline_core.managers.ensemble import PredictionFrameEnsembleManager

# Narrowed from a bare `warnings.filterwarnings("ignore")` (#366). The blanket form
# silenced EVERY category in the process that consumes the data — including
# views-datafactory's coverage `UserWarning`, which under their ADR-047 is the only signal
# distinguishing an observed zero from a zero-filled gap. Their own words: a filled month
# is "structurally indistinguishable from months where the source observed zero events".
#
# The noise this was hiding is real, so it is silenced BY NAME rather than removed. What is
# no longer silenced is the category nobody considered — which is the one that mattered.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    ensemble_path = EnsemblePathManager(Path(__file__))
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
    args = ForecastingModelArgs.parse_args()

    # For PredictionFrame ensembles (numpy end-to-end), use:
    #   PredictionFrameEnsembleManager
    # For DataFrame ensembles (parquet-based), use:
    #   EnsembleManager
    manager = EnsembleManager(
        ensemble_path=ensemble_path,
        wandb_notifications=args.wandb_notifications,
        use_prediction_store=args.prediction_store,
    )

    manager.execute_single_run(args)

"""
    return save_python_script(script_path, code)
