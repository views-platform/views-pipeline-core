from views_pipeline_core.templates.utils import save_python_script
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def generate(script_path: Path, model_name: str, model_algorithm: str) -> bool:
    """
    Generates a script that defines the `get_meta_config` function for model metadata.

    Parameters:
        script_path (Path):
            The path where the generated deployment configuration script will be saved.
            This should be a valid writable path.

        model_name (str):
            The name of the model. This will be included in the metadata configuration.

        model_algorithm (str):
            The algorithm of the model. This string will also be included in the metadata configuration.

    Returns:
        bool:
            True if the script was written and compiled successfully, False otherwise.
    """
    code = f"""def get_meta_config():
    \"""
    Contains the meta data for the model (model algorithm, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    \"""
    
    meta_config = {{
        "name": "{model_name}",
        "algorithm": "{model_algorithm}",
        # Uncomment and fill in. Those marked REQUIRED are enforced by
        # CoreConfigSniffer and a model will not run without them; the rest are optional.
        # "queryset": "escwa001_cflong",
        # "level": "pgm",
        # "creator": "Your name here",
        # "prediction_format": "dataframe", # or "prediction_frame" — REQUIRED for models
        # "rolling_origin_stride": 1, # REQUIRED for models
        # "evaluation_sequencing": "rolling_origin", # or "horizon_chunks" (ADR-060); omit for the default
        # "output_scale": "log", # "log" for log-scale predictions, "natural" for engines that undo transforms (HydraNet, R2DARTS2, SHURF)
        "regression_point_metrics": ["RMSLE", "MSE", "MSLE", "y_hat_bar"],
        # "metrics" was retired in views-evaluation 0.4.0 and its validator
        # now raises on it (#380). CRPS is a SAMPLE metric — declare it under
        # "regression_sample_metrics" if the model emits distributions.
    }}
    return meta_config
"""
    return save_python_script(script_path, code)
