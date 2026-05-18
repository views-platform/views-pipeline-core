from typing import Optional, List
import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def filter_metrics_from_dict(
    evaluation_dict: dict,
    metrics: List[str],
    target_identifier: str,
    model_name: str = None,
) -> pd.DataFrame:
    """
    Filters metrics from an evaluation dictionary based on specified metric names and a target identifier.

    Args:
        evaluation_dict (dict): Dictionary containing evaluation results with metric names as keys.
        metrics (list[str]): List of metric names to filter for (e.g. ['mse']).
        target_identifier (str): Target identifier to filter keys by (e.g. 'ged_sb_best').
        model_name (str, optional): Name of the model to include as an index in the resulting DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing filtered metrics. If `model_name` is provided, it is used as the index.
    """
    result = {}
    # Use namespacing pattern for robust segment matching
    target_kw = str(target_identifier).lower()
    target_pattern = rf"(^|[/_\-]) {re.escape(target_kw)} ($|[/_\-])"
    
    for key in evaluation_dict.keys():
        key_lower = key.lower()
        
        # Check if target identifier is present as a discrete segment
        if not re.search(target_pattern, key_lower, re.VERBOSE):
            continue

        # Check if all requested metrics are present in key (as segments)
        metrics_match = True
        for m in metrics:
            m_kw = str(m).lower()
            m_pattern = rf"(^|[/_\-]) {re.escape(m_kw)} ($|[/_\-])"
            if not re.search(m_pattern, key_lower, re.VERBOSE):
                metrics_match = False
                break
        
        if metrics_match:
            result[key] = evaluation_dict[key]
            
    if model_name:
        result = {"Model Name": model_name, **result}
        result = pd.DataFrame([result], columns=result.keys()).set_index("Model Name")
    else:
        result = pd.DataFrame([result], columns=result.keys())
    return result

def search_for_item_name(searchspace: List[str], keywords: List[str]) -> Optional[str]:
    """
    Searches for an item name that contains all keyword parts as discrete segments.
    Returns the first match if unique, warns about multiple matches, and returns None if no matches found.

    Args:
        searchspace: List of strings to search through
        keywords: List of keywords/phrases to match

    Returns:
        First matching item if unique match found, otherwise None
    """
    if not keywords:
        return None

    # Preprocess keywords: normalize
    keyword_list = [str(kw).lower() for kw in keywords if kw]
    
    if not keyword_list:
        return None

    matches = []
    for item in searchspace:
        item_lower = item.lower()
        match_all = True
        for kw in keyword_list:
            pattern = rf"(^|[/_\-]) {re.escape(kw)} ($|[/_\-])"
            if not re.search(pattern, item_lower, re.VERBOSE):
                match_all = False
                break
        
        if match_all:
            matches.append(item)

    # Handle results
    if not matches:
        return None
    
    if len(matches) > 1:
        logger.warning(f"Warning: Multiple matches found for {keywords}: {matches}. Returning first match.")
    
    return matches[0]

def search_for_item_name2(searchspace: List[str], keywords: List[str]) -> Optional[str]:
    """
    Searches for an item name that contains all keyword parts as discrete segments.
    Returns the first match if unique, warns about multiple matches, and returns None if no matches found.

    Args:
        searchspace: List of strings to search through (e.g. WandB keys)
        keywords: List of keywords/phrases to match (e.g. ['step-wise', 'mse', 'target_name'])

    Returns:
        First matching item if unique match found, otherwise None
    """
    if not keywords:
        return None

    # Preprocess keywords: normalize
    keyword_list = [str(kw).lower() for kw in keywords if kw]
    
    if not keyword_list:
        return None

    matches = []
    for item in searchspace:
        item_lower = item.lower()
        # Every keyword must be found as a discrete segment (delimited by / _ - or start/end)
        # We use a simple regex-based check for word boundaries or delimiters
        match_all = True
        for kw in keyword_list:
            # Pattern matches the keyword if it's surrounded by delimiters or at string boundaries
            pattern = rf"(^|[/_\-]) {re.escape(kw)} ($|[/_\-])"
            if not re.search(pattern, item_lower, re.VERBOSE):
                match_all = False
                break
        
        if match_all:
            matches.append(item)

    # Handle results
    if not matches:
        return None
    
    if len(matches) > 1:
        logger.warning(f"Warning: Multiple matches found for {keywords}: {matches}. Returning first match.")
    
    return matches[0]

def filter_metrics_by_eval_type_and_metrics(evaluation_dict: dict, eval_type: str, metrics: list, target_identifier: str, model_name: str, keywords: list = []) -> pd.DataFrame:
    """
    Filters metrics from an evaluation dictionary based on evaluation type, metric names, target identifier, and optional keywords, 
    and returns the results as a pandas DataFrame indexed by the model name.
    Args:
        evaluation_dict (dict): Dictionary containing evaluation results, where keys are metric identifiers.
        eval_type (str): The evaluation type to filter by (e.g., "classification", "regression").
        metrics (list): List of metric names (strings) to filter for.
        target_identifier (str): Target identifier to further filter metric keys.
        model_name (str): Name of the model, used as the index in the resulting DataFrame.
        keywords (list, optional): Additional keywords to refine the search for metric keys. Defaults to an empty list.
    Returns:
        pd.DataFrame: A DataFrame containing the filtered metrics, with columns as metric keys and a single row indexed by model_name.
    Raises:
        ValueError: If any of the input arguments are of incorrect type.
    """
    if not isinstance(metrics, list):
        raise ValueError(f"Metrics should be a list. Got {type(metrics)} instead.")
    if not all(isinstance(m, str) for m in metrics):
        raise ValueError(f"Metrics should be a list of strings. Got {[type(m) for m in metrics]} instead.")
    if not isinstance(eval_type, str):
        raise ValueError(f"Eval type should be a string. Got {type(eval_type)} instead.")
    if not isinstance(target_identifier, str):
        raise ValueError(f"Target identifier should be a string. Got {type(target_identifier)} instead.")
    if not isinstance(keywords, list):
        raise ValueError(f"Keywords should be a list. Got {type(keywords)} instead.")
    if not all(isinstance(k, str) for k in keywords):
        raise ValueError(f"Keywords should be a list of strings. Got {[type(k) for k in keywords]} instead.")
    if not isinstance(evaluation_dict, dict):
        raise ValueError(f"Evaluation dictionary should be a dictionary. Got {type(evaluation_dict)} instead.")

    target_metric_keys = []
    for metric in metrics:
        result = search_for_item_name2(searchspace=list(evaluation_dict.keys()), keywords=[eval_type, metric, target_identifier, *keywords])
        if result:
            target_metric_keys.append(result)
    
    metric_dataframe = pd.DataFrame(
        [{k: evaluation_dict[k] for k in target_metric_keys}],
        columns=target_metric_keys,
        index=[model_name]
    )
    logger.debug(f"Filtered metrics DataFrame:\n{metric_dataframe}")
    
    return metric_dataframe
