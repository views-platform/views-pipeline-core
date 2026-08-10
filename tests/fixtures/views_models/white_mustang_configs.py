# VENDORED FIXTURE — do not edit by hand.
#
# Verbatim copy of views-models `ensembles/white_mustang/configs/*.py` at
# views-models@2b13bf02 (default branch, merged — deliberately NOT rusty_bucket,
# which is mid-rebase on views-models#367 and currently invalid on its own).
#
# Refreshing: re-copy from views-models and update the sha above. Editing this
# to make a test pass defeats its only purpose — it exists to be a real artifact
# from the other side of the boundary, not a convenient one.

SOURCE_REPO = "views-models"
SOURCE_COMMIT = "2b13bf02"
SOURCE_ENSEMBLE = "white_mustang"


def get_meta_config():
    """
    Contains the metadata for the model (model architecture, name, target variable, and level of analysis).
    This config is for documentation purposes only, and modifying it will not affect the model, the training, or the evaluation.

    Returns:
    - meta_config (dict): A dictionary containing model meta configuration.
    """
    meta_config = {
        "name": "white_mustang",
        "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline", "locf_cmbaseline"],
        "regression_point_metrics": ["RMSLE", "MSE", "MSLE", "y_hat_bar"],
        "regression_targets": ["lr_ged_sb"],  # Double-check the target variables of each model
        "level": "pgm",
        "aggregation": "mean",
        "creator": "Xiaolong",
        "reconciliation": "pgm_cm_point",
        "reconcile_with": "cruel_summer"
    }
    return meta_config

def get_deployment_config():

    """
    Contains the configuration for deploying the model into different environments.
    This configuration is "behavioral" so modifying it will affect the model's runtime behavior and integration into the deployment system.

    Returns:
    - deployment_config (dict): A dictionary containing deployment settings, determining how the model is deployed, including status, endpoints, and resource allocation.
    """

    # More deployment settings can/will be added here
    deployment_config = {
       "deployment_status": "deployed", # shadow, deployed, baseline, or deprecated
    }

    return deployment_config

def get_modelset_config():
    """
    Contains the list of constituent models for the ensemble.

    Returns:
    - modelset_config (dict): A dictionary with the key 'models' listing constituent model names.
    """
    modelset_config = {
        "models": ["lavender_haze", "blank_space"],
    }
    return modelset_config
