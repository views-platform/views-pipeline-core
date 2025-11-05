from datetime import datetime
import logging
from pathlib import Path
from views_pipeline_core.files.utils import read_log_file

logger = logging.getLogger(__name__)


def validate_model_conditions(path_generated, run_type):
    """
    Validate temporal requirements for model training and data freshness.

    Checks that the model was trained in the current training cycle (after July)
    and that both generated features and raw data were fetched in the current month.

    Args:
        path_generated: Path to model's data_generated directory containing log files
        run_type: Type of run to validate: 'calibration' | 'forecasting' | 'validation'
            Determines which log file to read ({run_type}_log.txt)

    Returns:
        True if all temporal conditions are met, False otherwise

    Raises:
        Exception: If log file cannot be read (logged but not raised)

    Example:
        >>> from pathlib import Path
        >>> path = Path('models/conflict_model_v1/data/generated')
        >>> is_valid = validate_model_conditions(path, 'forecasting')
        >>> if not is_valid:
        ...     print("Model does not meet temporal requirements")

    Note:
        - Training cycle: July-June (models trained after July of previous year)
        - Current month requirement ensures data freshness for production
        - Logs detailed error messages before returning False
    """
    
    log_file_path = Path(path_generated) / f"{run_type}_log.txt"
    try:
        log_data = read_log_file(log_file_path)
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return False

    current_time = datetime.now()
    current_year = current_time.year
    current_month = current_time.month

    # Extract from log data
    model_name = log_data["Single Model Name"]
    model_timestamp = datetime.strptime(log_data["Single Model Timestamp"], "%Y%m%d_%H%M%S")
    data_generation_timestamp = None if log_data["Data Generation Timestamp"] == "None" else (
        datetime.strptime(log_data["Data Generation Timestamp"], "%Y%m%d_%H%M%S"))

    data_fetch_timestamp = None if log_data["Data Fetch Timestamp"] == "None" else (
        datetime.strptime(log_data["Data Fetch Timestamp"], "%Y%m%d_%H%M%S"))

    # Condition 1: Model trained in the current year after July
    if current_month >= 7:
        if not (model_timestamp.year == current_year and model_timestamp.month >= 7):
            logger.error(f"Model {model_name} was trained in {model_timestamp.year}_{model_timestamp.month}. "
                         f"Please use the latest model that is trained after {current_year}_07. Exiting.")
            return False
    elif current_month < 7:
        if not (
                (model_timestamp.year == current_year - 1 and model_timestamp.month >= 7) or
                (model_timestamp.year == current_year and model_timestamp.month < 7)
        ):
            logger.error(f"Model {model_name} was trained in {model_timestamp.year}_{model_timestamp.month}. "
                         f"Please use the latest model that is trained after {current_year - 1}_07. Exiting.")
            return False

    # Condition 2: Data generated in the current month
    if data_generation_timestamp and not (
            data_generation_timestamp.year == current_year and data_generation_timestamp.month == current_month):
        logger.error(f"Data for model {model_name} was not generated in the current month. Exiting.")
        return False

    # Condition 3: Raw data fetched in the current month
    if data_fetch_timestamp and not (
            data_fetch_timestamp.year == current_year and data_fetch_timestamp.month == current_month):
        logger.error(f"Raw data for model {model_name} was not fetched in the current month. Exiting.")
        return False

    return True


def validate_ensemble_model_deployment_status(path_generated, run_type, ensemble_deployment_status):
    """
    Validate deployment status compatibility between ensemble and constituent models.

    Ensures that ensemble deployment status matches constituent model requirements
    and that deprecated models are not used in production ensembles.

    Args:
        path_generated: Path to model's data_generated directory containing log files
        run_type: Type of run to validate: 'calibration' | 'forecasting' | 'validation'
        ensemble_deployment_status: Deployment status of ensemble:
            'production' | 'shadow' | 'deprecated'

    Returns:
        True if deployment status conditions are met, False otherwise

    Raises:
        Exception: If log file cannot be read (logged but not raised)

    Example:
        >>> path = Path('models/rf_model/data/generated')
        >>> is_valid = validate_ensemble_model_deployment_status(
        ...     path, 'forecasting', 'production'
        ... )
        >>> if not is_valid:
        ...     print("Deployment status mismatch")

    Note:
        - Deprecated ensembles cannot be used
        - Deprecated constituent models cannot be used
        - production models can only be in deployed ensembles
        - Prevents accidental use of outdated models
    """

    log_file_path = Path(path_generated) / f"{run_type}_log.txt"
    try:
        log_data = read_log_file(log_file_path)
    except Exception as e:
        logger.error(f"Error reading log file: {e}. Exiting.")
        return False

    model_name = log_data["Single Model Name"]
    single_model_dp_status = log_data["Deployment Status"]

    # More check conditions can be added here
    if ensemble_deployment_status == 'Deprecated':
        logger.error(f"Deployment status is deprecated. Exiting.")
        return False
    
    if single_model_dp_status == 'Deprecated':
        logger.error(f"Model {model_name} deployment status is deprecated. Exiting.")
        return False

    if single_model_dp_status == "production" and ensemble_deployment_status != "production":
        logger.error(f"Model {model_name} deployment status is deployed "
                     f"but the ensemble is not. Exiting.")
        return False

    return True


def validate_partition_config(ensemble_manager, model_manager, run_type):
    """
    Validates the partition configuration for the ensemble model.
    """
    ensemble_partition_config = ensemble_manager._partition_dict[run_type]
    model_partition_config = model_manager._partition_dict[run_type]
    if ensemble_partition_config != model_partition_config:
        logger.error(f"Ensemble partition config {ensemble_partition_config} does not match model partition config {model_partition_config}. Exiting.")
        return False
    return True

def validate_ensemble_model(config):
    """
    Validate data partition compatibility between ensemble and constituent model.

    Ensures that train/test splits match between the ensemble and individual
    models to maintain evaluation integrity.

    Args:
        ensemble_manager: EnsembleManager instance containing partition configuration
        model_manager: ModelManager instance containing partition configuration
        run_type: Type of run to validate partition for:
            'calibration' | 'forecasting' | 'validation'

    Returns:
        True if partition configurations match, False otherwise

    Example:
        >>> from views_pipeline_core.managers.ensemble import EnsembleManager
        >>> from views_pipeline_core.managers.model import ModelManager
        >>> ensemble = EnsembleManager(ensemble_path)
        >>> model = ModelManager(model_path)
        >>> is_valid = validate_partition_config(ensemble, model, 'calibration')
        >>> if not is_valid:
        ...     print("Partition mismatch detected")

    Note:
        - Critical for fair ensemble evaluation
        - Prevents data leakage between train/test sets
        - Logs error with both partition configs on mismatch
    """
    from views_pipeline_core.managers.model import ModelManager, ModelPathManager
    from views_pipeline_core.managers.ensemble import EnsembleManager, EnsemblePathManager

    ensemble_manager = EnsembleManager(EnsemblePathManager(config["name"]))
    for model_name in config["models"]:
        model_path_manager = ModelPathManager(model_name)
        model_manager = ModelManager(model_path_manager)
        path_generated = model_path_manager.data_generated

        if (
                (not validate_model_conditions(path_generated, config["run_type"])) or
                (not validate_ensemble_model_deployment_status(path_generated, config["run_type"], config["deployment_status"])) or
                (not validate_partition_config(ensemble_manager, model_manager, config["run_type"]))
        ):
            exit(1)  # Shut down if conditions are not met
    logger.info(f"Model {config['name']} meets the required conditions.")