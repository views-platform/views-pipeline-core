from typing import Optional
import re
import pandas as pd


def get_conflict_type_from_feature_name(feature_name: str) -> Optional[str]:
    """
    Get the UCDP defined conflict type from the feature name.

    Args:
        feature_name (str): The name of the feature.

    Returns:
        Optional[str]: The conflict type if found, otherwise None.
    """
    valid_conflict_types = {"ns": "non state", "os": "one sided", "sb": "state based"}
    tokens = feature_name.split("_")
    for conflict_type in valid_conflict_types.keys():
        if conflict_type in tokens:
            return conflict_type, valid_conflict_types[conflict_type]
    return "", ""


def filter_metrics_from_dict(
    evaluation_dict: dict,
    metrics: list[str],
    conflict_code: str,
    model_name: str = None,
) -> pd.DataFrame:
    """
    Filters metrics from an evaluation dictionary based on specified metric names and a conflict code.

    Args:
        evaluation_dict (dict): Dictionary containing evaluation results with metric names as keys.
        metrics (list[str]): List of metric names to filter for.
        conflict_code (str): Conflict code to filter keys by.
        model_name (str, optional): Name of the model to include as an index in the resulting DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing filtered metrics. If `model_name` is provided, it is used as the index.
    """
    result = {}
    for key in evaluation_dict.keys():
        tokens = re.split(r"[_/\-]", key.lower())
        # Ensure all metrics are present in tokens
        if all(m.lower() in tokens for m in metrics) and conflict_code.lower() in tokens:
            result[key] = evaluation_dict[key]
    if model_name:
        result = {"Model Name": model_name, **result}
        result = pd.DataFrame([result], columns=result.keys()).set_index("Model Name")
    else:
        result = pd.DataFrame([result], columns=result.keys())
    return result

def search_for_item_name(searchspace: list, keywords: list[str]) -> Optional[str]:
    result = None
    for key in searchspace:
        tokens = re.split(r"[_/\-]", key.lower())
        # Ensure all metrics are present in tokens
        if all(m.lower() in tokens for m in keywords):
            if result is None:
                result = key
            else:
                print(f"Multiple results found for keywords {keywords}: {result} and {key}. Refine your search criteria.")
    return result


# def filter_metrics_from_dict(evaluation_dict: dict, metric: str, conflict_code: str, model_name: str = None) -> pd.DataFrame:
#     result = {}
#     for key in evaluation_dict.keys():
#         tokens = re.split(r'[_/\-]', key.lower())
#         if metric.lower() in tokens and conflict_code.lower() in tokens:
#                 result[key] = evaluation_dict[key]
#     if model_name:
#         # Insert 'Model Name' as the first item in the result dict
#         result = {'Model Name': model_name, **result}
#         result = pd.DataFrame([result], columns=result.keys()).set_index('Model Name')
#     else:
#         result = pd.DataFrame([result], columns=result.keys())
#     return result
