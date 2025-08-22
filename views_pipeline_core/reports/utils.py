from typing import Optional, List
import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)


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
    metrics: List[str],
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
        # result[key] = search_for_item_name(searchspace=tokens, keywords=[*metrics, conflict_code])
    if model_name:
        result = {"Model Name": model_name, **result}
        result = pd.DataFrame([result], columns=result.keys()).set_index("Model Name")
    else:
        result = pd.DataFrame([result], columns=result.keys())
    return result

def search_for_item_name(searchspace: List[str], keywords: List[str]) -> Optional[str]:
    """
    Searches for an item name that contains all keyword parts. Returns the first match
    if unique, warns about multiple matches, and returns None if no matches found.

    Args:
        searchspace: List of strings to search through
        keywords: List of keywords/phrases to match

    Returns:
        First matching item if unique match found, otherwise None
    """
    # Handle empty keywords upfront
    if not keywords:
        return None

    # Preprocess keywords: split, normalize, and remove empties
    keyword_parts = []
    for kw in keywords:
        parts = re.split(r"[_/\-]", kw.lower())
        keyword_parts.extend(p for p in parts if p)
    
    # Handle case where keywords only contained separators
    if not keyword_parts:
        return None

    matches = []
    for item in searchspace:
        # Tokenize item and remove empty tokens
        tokens = [t for t in re.split(r"[_/\-]", item.lower()) if t]
        
        # Check if all keyword parts are present
        if all(kw_part in tokens for kw_part in keyword_parts):
            matches.append(item)

    # Handle results
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    
    print(f"Warning: Multiple matches found for {keywords}: {matches}. Returning first match.")
    return matches[0]

def filter_metrics_by_eval_type_and_metrics(evaluation_dict: dict, eval_type: str, metrics: list, conflict_code: str, model_name: str, keywords: list = []) -> pd.DataFrame:
    if not isinstance(metrics, list):
        raise ValueError(f"Metrics should be a list. Got {type(metrics)} instead.")
    if not all(isinstance(m, str) for m in metrics):
        raise ValueError(f"Metrics should be a list of strings. Got {[type(m) for m in metrics]} instead.")
    if not isinstance(eval_type, str):
        raise ValueError(f"Eval type should be a string. Got {type(eval_type)} instead.")
    if not isinstance(conflict_code, str):
        raise ValueError(f"Conflict code should be a string. Got {type(conflict_code)} instead.")
    if not isinstance(keywords, list):
        raise ValueError(f"Keywords should be a list. Got {type(keywords)} instead.")
    if not all(isinstance(k, str) for k in keywords):
        raise ValueError(f"Keywords should be a list of strings. Got {[type(k) for k in keywords]} instead.")
    if not isinstance(evaluation_dict, dict):
        raise ValueError(f"Evaluation dictionary should be a dictionary. Got {type(evaluation_dict)} instead.")

    target_metric_keys = []
    for metric in metrics:
        result = search_for_item_name(searchspace=list(evaluation_dict.keys()), keywords=[eval_type, metric, conflict_code, *keywords])
        if result:
            target_metric_keys.append(result)
    
    metric_dataframe = pd.DataFrame(
        [{k: evaluation_dict[k] for k in target_metric_keys}],
        columns=target_metric_keys,
        index=[model_name]
    )
    logger.debug(f"Filtered metrics DataFrame:\n{metric_dataframe}")
    
    return metric_dataframe


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
