import logging
import pandas as pd
from pathlib import Path
from typing import Union
from datetime import datetime

logger = logging.getLogger(__name__)


def read_log_file(log_file_path):
    """
    Reads the log file and returns a dictionary with the relevant information.

    Args:

    - log_file_path (str or Path): The path to the log file.

    Returns:
    - dict: A dictionary containing the model name, model timestamp, data generation timestamp, and data fetch timestamp.
    """

    log_data = {}
    with open(log_file_path, "r") as file:
        for line in file:
            line = line.strip()

            # Skip blank lines
            if not line:
                continue
            else:
                key, value = line.split(": ", 1)
                # There are duplicated keys for ensemble models, but it's not a problem bc these keys are not used
                log_data[key] = value

    return log_data
    

def create_data_fetch_log_file(path_raw, 
                              run_type,
                              model_name,
                              data_fetch_timestamp):
    """
    Creates a log file in the specified single model folder with details about the data fetch.

    Args:
    - path_raw (Path): The path to the folder where the log file will be created.
    - run_type (str): The type of run.
    - model_name (str): The name of the model.
    - data_fetch_timestamp (str): The timestamp when the raw data used was fetched from VIEWS.
    """
    
    data_fetch_log_file_path = f"{path_raw}/{run_type}_data_fetch_log.txt"

    with open(data_fetch_log_file_path, "w") as log_file:
        log_file.write(f"Single Model Name: {model_name}\n")
        log_file.write(f"Data Fetch Timestamp: {data_fetch_timestamp}\n\n")

    logger.info(f"Data fetch log file created at {data_fetch_log_file_path}")


def create_specific_log_file(path_generated,
                    run_type,
                    model_name,
                    deployment_status,
                    model_timestamp,
                    data_generation_timestamp,
                    data_fetch_timestamp,
                    model_type="single",
                    mode="w",):
    """
    Creates a log file in the specified model folder with details about the generated data.

    Args:
    - path_generated (Path): The path to the folder where the log file will be created.
    - run_type (str): The type of run
    - model_name (str): The name of the model.
    - deployment_status (str): The status of the deployment.
    - model_timestamp (str): The timestamp when the model was trained.
    - data_generation_timestamp (str): The timestamp when the data was generated.
    - data_fetch_timestamp (str, optional): The timestamp when the raw data used was fetched from VIEWS.
    - model_type (str, optional): The type of model. Default is "single".
    - mode (str, optional): The mode in which the file will be opened. Default is "w".
    """

    log_file_path = f"{path_generated}/{run_type}_log.txt"

    # Capitalize the first letter of the model type
    model_type = model_type[0].upper() + model_type[1:]

    with open(log_file_path, mode) as log_file:
        log_file.write(f"{model_type} Model Name: {model_name}\n")
        log_file.write(f"{model_type} Model Timestamp: {model_timestamp}\n")
        log_file.write(f"Data Generation Timestamp: {data_generation_timestamp}\n")
        log_file.write(f"Data Fetch Timestamp: {data_fetch_timestamp}\n")
        log_file.write(f"Deployment Status: {deployment_status}\n\n")


def create_log_file(path_generated,
                    model_config,
                    model_timestamp,
                    data_generation_timestamp,
                    data_fetch_timestamp,
                    model_type="single",
                    models=None):
    """
    Creates a log file for the given model configuration and timestamps.
    Parameters:
    path_generated (str): The path where the log file will be generated.
    model_config (dict): Configuration dictionary for the model containing keys:
        - "run_type" (str): The type of run (e.g., "calibration", "evaluation").
        - "name" (str): The name of the model.
        - "deployment_status" (str): The deployment status of the model.
    model_timestamp (str): Timestamp for the model.
    data_generation_timestamp (str): Timestamp for data generation.
    data_fetch_timestamp (str): Timestamp for data fetching.
    model_type (str, optional): The type of model, default is "single".
    models (list, optional): List of model names for which logs should be created, default is None.
    
    Returns:
    None
    """
    
    run_type = model_config["run_type"]
    model_name = model_config["name"]
    deployment_status = model_config["deployment_status"]
    
    create_specific_log_file(path_generated, run_type, model_name, deployment_status,
                    model_timestamp, data_generation_timestamp, data_fetch_timestamp, model_type)
    if models:
        from views_pipeline_core.managers.model import ModelPathManager
        for m_name in models:
            model_path = ModelPathManager(m_name)
            model_path_generated = model_path.data_generated
            log_data = read_log_file(model_path_generated / f"{run_type}_log.txt")
            create_specific_log_file(path_generated, run_type, m_name, log_data["Deployment Status"], 
                                     log_data["Single Model Timestamp"], log_data["Data Generation Timestamp"], log_data["Data Fetch Timestamp"], mode="a")
    
    logger.info(f"Log file created at {path_generated}/{run_type}_log.txt")


def handle_single_log_creation(model_path, config: dict, train: bool) -> None:
    """
    Handles the creation of log files for different stages of the model pipeline.

    Depending on the flags provided, this method creates log files for training,
    evaluation, or forecasting stages of the model pipeline. It reads the data
    fetch timestamp from an existing log file and includes it in the new log files.

    Args:
        model_path (ModelPathManager): The path manager for the model.
        config (dict): The configuration dictionary for the model.
        train (bool): Flag indicating whether the log is for the training stage.

    Returns:
        None
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_fetch_timestamp = read_log_file(
        model_path.data_raw / f"{config['run_type']}_data_fetch_log.txt"
    ).get("Data Fetch Timestamp", None)

    create_log_file(
        path_generated=model_path.data_generated,
        model_config=config,
        model_timestamp=timestamp if train else config["timestamp"],
        data_generation_timestamp=None if train else timestamp,
        data_fetch_timestamp=data_fetch_timestamp,
    )


def handle_ensemble_log_creation(model_path, config: dict) -> None:
    """
    This method generates a timestamp for data generation, updates the configuration
    with this timestamp, and creates a log file for the ensemble model using the
    provided configuration and paths.

    Returns:
        None
    """
    path_generated_e = model_path.data_generated
    data_generation_timestamp = datetime.now().strftime("%Y%m%d")
    config["timestamp"] = data_generation_timestamp

    # How to define an ensemble model timestamp? Currently set as data_generation_timestamp.
    create_log_file(
        path_generated_e,
        config,
        data_generation_timestamp,
        data_generation_timestamp,
        data_fetch_timestamp=None,
        model_type="ensemble",
        models=config["models"],
    )

        
def save_dataframe(dataframe: pd.DataFrame, save_path: Union[str, Path]):
    """
    Saves a pandas DataFrame to a specified file path in various formats.
    save_path = Path(save_path)
    - dataframe (pd.DataFrame): The DataFrame to be saved.
    - save_path (Union[str, Path]): The path where the DataFrame will be saved. The file extension determines the format.

    Raises:
    - ValueError: If the file extension is not provided or is not supported.
    - Exception: If there is an error saving the DataFrame.
    """
    FILE_EXTENSION_ERROR_MESSAGE = "A valid file extension must be provided.E.g. .pkl or .parquet"
    
    # Checks
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    file_extension = save_path.suffix
    if dataframe is None:
        raise ValueError("The DataFrame must be provided")
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The DataFrame must be a pandas DataFrame")
    if file_extension is None or file_extension == "":
        raise ValueError(f"No file extension {file_extension} found. {FILE_EXTENSION_ERROR_MESSAGE}")
    
    try:
        logger.debug(f"Saving the DataFrame to {save_path} in {file_extension} format")
        # if file_extension == ".csv":
        #     dataframe.to_csv(save_path, index=True)
        # elif file_extension == ".xlsx":
        #     dataframe.to_excel(save_path)
        if file_extension == ".parquet":
            dataframe.to_parquet(save_path)
        elif file_extension == ".pkl":
            dataframe.to_pickle(save_path)
        else:
            raise ValueError(f"{FILE_EXTENSION_ERROR_MESSAGE}")
    except Exception as e:
        logger.exception(f"Error saving the DataFrame to {save_path}: {e}")
        raise

def read_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a pandas DataFrame from a specified file path in various formats.
    
    - file_path (Union[str, Path]): The path from where the DataFrame will be read. The file extension determines the format.

    Returns:
    - pd.DataFrame: The DataFrame read from the file.

    Raises:
    - ValueError: If the file extension is not provided or is not supported.
    - Exception: If there is an error reading the DataFrame.
    """
    FILE_EXTENSION_ERROR_MESSAGE = "A valid extension must be provided. E.g. .pkl or .parquet"
    
    # Checks
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    file_extension = file_path.suffix
    if file_extension is None or file_extension == "":
         raise ValueError(f"No file extension {file_extension} found. {FILE_EXTENSION_ERROR_MESSAGE}")
    
    try:
        logger.debug(f"Reading the DataFrame from {file_path} in {file_extension} format")
        if file_extension == ".csv":
            return pd.read_csv(file_path)
        # elif file_extension == ".xlsx":
        #     return pd.read_excel(file_path)
        if file_extension == ".parquet":
            return pd.read_parquet(file_path)
        elif file_extension == ".pkl":
            return pd.read_pickle(file_path)
        else:
            raise ValueError(FILE_EXTENSION_ERROR_MESSAGE)
    except Exception as e:
        logger.exception(f"Error reading the DataFrame from {file_path}: {e}")
        raise


def generate_model_file_name(run_type: str, file_extension: str) -> str:
    """
    Generates a model file name based on the run type, and timestamp.

    Args:
        run_type (str): The type of run (e.g., calibration, validation).
        file_extension (str): The file extension. Default is set in PipelineConfig().dataframe_format. E.g. .pt, .pkl, .h5

    Returns:
        str: The generated model file name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{run_type}_model_{timestamp}{file_extension}"


def generate_output_file_name(
    generated_file_type: str,
    run_type: str,
    timestamp: str,
    sequence_number: int,
    file_extension: str,
) -> str:
    """
    Generates a prediction file name based on the run type, generated file type, steps, and timestamp.

    Args:
        generated_file_type (str): The type of generated file (e.g., predictions, output).
        sequence_number (int): The sequence number.
        run_type (str): The type of run (e.g., calibration, validation).
        timestamp (str): The timestamp of the generated file.
        file_extension (str): The file extension. Default is set in PipelineConfig().dataframe_format. E.g. .pkl, .csv, .xlsx, .parquet

    Returns:
        str: The generated prediction file name.
    """
    # logger.info(f"sequence_number: {sequence_number}")
    if sequence_number is not None:
        return f"{generated_file_type}_{run_type}_{timestamp}_{str(sequence_number).zfill(2)}{file_extension}"
    else:
        return f"{generated_file_type}_{run_type}_{timestamp}{file_extension}"
    

def generate_evaluation_file_name(
    evaluation_type: str,
    conflict_type: str,
    run_type: str,
    timestamp: str,
    file_extension: str,
) -> str:
    """
    Generates an evaluation file name based on the run type, evaluation type, and timestamp.

    Args:
        evaluation_type (str): The type of evaluation file (e.g., step, month, ts).
        conflict_type (str): The type of conflict (e.g., sb, os, ns).
        run_type (str): The type of run (e.g., calibration, validation).
        timestamp (str): The timestamp of the generated file.
        file_extension (str): The file extension. Default is set in PipelineConfig().dataframe_format. E.g. .pkl, .csv, .xlsx, .parquet

    Returns:
        str: The generated prediction file name.
    """
    # logger.info(f"sequence_number: {sequence_number}")
    return f"eval_{run_type}_{conflict_type}_{evaluation_type}_{timestamp}{file_extension}"


def generate_evaluation_report_name(
    run_type: str,
    conflict_type: str,
    timestamp: str,
    file_extension: str,
) -> str:
    """
    Generates an evaluation file name based on the run type, evaluation type, and timestamp.

    Args:
        evaluation_type (str): The type of evaluation file (e.g., step, month, ts).
        conflict_type (str): The type of conflict (e.g., sb, os, ns).
        run_type (str): The type of run (e.g., calibration, validation).
        timestamp (str): The timestamp of the generated file.
        file_extension (str): The file extension. Default is set in PipelineConfig().dataframe_format. E.g. .pkl, .csv, .xlsx, .parquet

    Returns:
        str: The generated prediction file name.
    """
    # logger.info(f"sequence_number: {sequence_number}")
    return f"eval_{run_type}_{conflict_type}_{timestamp}{file_extension}"