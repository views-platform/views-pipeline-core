import logging
import numpy as np
import requests
import json
import pandas as pd
from typing import Union

logger = logging.getLogger(__name__)

def ensure_float64(df):
        """
        Check if the DataFrame only contains np.float64 types. If not, raise a warning
        and convert the DataFrame to use np.float64 for all its numeric columns.
        """
        non_float64_cols = df.select_dtypes(include=['number']).columns[
            df.select_dtypes(include=['number']).dtypes != np.float32]

        if len(non_float64_cols) > 0:
            logger.warning(
                f"DataFrame contains non-np.float64 numeric columns. Converting the following columns: {', '.join(non_float64_cols)}")

            for col in non_float64_cols:
                df[col] = df[col].astype(np.float32)
        return df

def download_json(url):
    """
    Downloads a JSON file from the given URL and returns its content as a string.

    Args:
        url (str): The URL of the JSON file to download.

    Returns:
        str: The content of the JSON file as a string if the download and decoding are successful.
        None: If there is an error during the download or decoding process.

    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request.
        json.JSONDecodeError: If there is an issue with decoding the JSON content.

    Logs:
        Error messages are logged if there is an issue with downloading or decoding the JSON file.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        return response.content.decode()  # Parse JSON response into a dictionary
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading JSON file: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file: {e}")
        return None
     
def convert_json_to_list_of_dicts(json_data):
    """
    Convert a string of newline-separated JSON objects into a list of dictionaries.

    Args:
        json_data (str): A string containing JSON objects separated by newlines.

    Returns:
        list: A list of dictionaries, each representing a JSON object.

    Raises:
        json.JSONDecodeError: If any of the JSON objects are invalid.
    """
    # Split the JSON data into individual JSON objects
    json_objects = json_data.strip().split('\n')
    # Convert each JSON object to a dictionary and store in a list
    list_of_dicts = [json.loads(obj) for obj in json_objects]
    return list_of_dicts

def replace_nan_values(df: pd.DataFrame, value: Union[int, float] = 0.0) -> pd.DataFrame:
        """
        Replaces NaN values in a DataFrame or a list of DataFrames with a specified value.

        This function iterates through the columns of the input DataFrame(s) and replaces
        NaN values with the provided value. For object-type columns containing lists, it
        replaces NaN values within the lists.

        Args:
            df (Union[pd.DataFrame, List[pd.DataFrame]]): The input DataFrame or a list of DataFrames
                where NaN values need to be replaced.
            value (Union[int, float], optional): The value to replace NaN values with. Defaults to 0.0.

        Returns:
            Union[pd.DataFrame, List[pd.DataFrame]]: The DataFrame or list of DataFrames with NaN
            values replaced.

        Raises:
            ValueError: If the input is neither a DataFrame nor a list of DataFrames.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 2, 3]})
            >>> replace_nan_values(df, value=0)
               A    B
            0  1.0  0.0
            1  0.0  2.0
            2  3.0  3.0
        """
        def _replace_nan(d):
            for column in d.columns:
                if d[column].dtype == "object":
                    d[column] = d[column].apply(
                        lambda x: [np.float64(value) if np.isnan(i) else float(i) for i in x]
                    )
                    d[column] = d[column].apply(
                        lambda x: [0.0 if i < 0 else float(i) for i in x]
                    )
                else:
                    d[column] = d[column].apply(
                        lambda x: np.float64(value) if np.isnan(x) else float(x)
                    )
                    d[column] = d[column].apply(
                        lambda x: 0.0 if x < 0 else float(x)
                    )
            return d
        if isinstance(df, list):
            for i, d in enumerate(df):
                df[i] = _replace_nan(d)
            return df
        elif isinstance(df, pd.DataFrame):
            df = _replace_nan(df)
            return df
        else:
            raise ValueError(f"Input should be a DataFrame or a list of DataFrames. Found {type(df)}")