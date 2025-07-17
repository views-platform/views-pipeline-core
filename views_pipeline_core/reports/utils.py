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
    valid_conflict_types = {'ns': 'non state', 'os': 'one sided', 'sb': 'state based'}
    tokens = feature_name.split('_')
    for conflict_type in valid_conflict_types.keys():
        if conflict_type in tokens:
            return conflict_type, valid_conflict_types[conflict_type]
    return None

def filter_metrics_from_dict(evaluation_dict: dict, metric: str, conflict_code: str, model_name: str = None) -> pd.DataFrame:
    result = {}
    for key in evaluation_dict.keys():
        tokens = re.split(r'[_/\-]', key.lower())
        if metric.lower() in tokens and conflict_code.lower() in tokens:
                result[key] = evaluation_dict[key]
    if model_name:
        # Insert 'Model Name' as the first item in the result dict
        result = {'Model Name': model_name, **result}
        result = pd.DataFrame([result], columns=result.keys()).set_index('Model Name')
    else:
        result = pd.DataFrame([result], columns=result.keys())
    return result

