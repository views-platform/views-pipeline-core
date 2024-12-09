import logging
logger = logging.getLogger(__name__)

# File format for dataframes
DATAFRAME_FORMAT = '.parquet'

# File format for models
MODEL_FORMAT = '.pkl'

def get_dataframe_format() -> str:
    """
    Getter for DATAFRAME_FORMAT.
    
    Returns:
    - str: The current file format for dataframes.
    """
    logger.debug(f"Getting dataframe format from config: {DATAFRAME_FORMAT}")
    return DATAFRAME_FORMAT

def get_model_format() -> str:
    """
    Getter for MODEL_FORMAT.
    
    Returns:
    - str: The current file format for models.
    """
    logger.debug(f"Getting model format from config: {MODEL_FORMAT}")
    return MODEL_FORMAT

def set_model_format(format: str):
    """
    Setter for MODEL_FORMAT.
    
    - format (str): The new file format for models.
    """
    global MODEL_FORMAT
    logger.debug(f"Setting model format in config: {format}")
    MODEL_FORMAT = format

def set_dataframe_format(format: str):
    """
    Setter for DATAFRAME_FORMAT.
    
    - format (str): The new file format for dataframes.
    """
    global DATAFRAME_FORMAT
    logger.debug(f"Setting dataframe format in config: {format}")
    DATAFRAME_FORMAT = format