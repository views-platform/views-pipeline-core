from typing import Dict
from views_pipeline_core.templates.utils import save_python_script
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def generate(
    script_path: Path,
    deployment_type: str = "shadow",
    additional_settings: Dict[str, any] = None,
) -> bool:
    """
    Generates a script that defines the `get_deployment_config` function for configuring the deployment status and settings.

    Parameters:
        script_path (Path): The path where the generated deployment configuration script will be saved.
                           This should be a valid writable path.
        deployment_type (str, optional):
            The type of deployment. Must be one of "shadow", "deployed", "baseline", or "deprecated".
            Default is "shadow".
            - "shadow": The deployment is shadowed and not yet active.
            - "deployed": The deployment is active and in use.
            - "baseline": The deployment is in a baseline state, for reference or comparison.
            - "deprecated": The deployment is deprecated and no longer supported.
        additional_settings (dict, optional):
            A dictionary of additional settings to include in the deployment configuration.
            These settings will be merged with the default configuration. Defaults to None.


    ## Deprecated — this mints the RETIRED vocabulary (ADR-057, #498)

    `views_pipeline_core.templates.model.template_config_maturity` is the current
    generator. This one is kept, working and unchanged, for one reason: views-models'
    scaffolders import it by module name, and their fleet cannot accept `maturity` yet —
    ~9 of their test modules plus `update_readme.py` and `run_integration_tests.sh`
    require `config_deployment.py` in every source directory. Breaking their CI to enforce
    a rename they cannot take is not a fix.

    There is deliberately **no runtime warning**. A warning here reaches a human running an
    interactive scaffolder who cannot act on it — switching today breaks their build — and
    a warning that instructs a reader to break their build is how a team learns to ignore
    warnings.

    **Deleted when views-models' scaffolders no longer import it**, which is also when
    ADR-057's window can close. Until then the census guard in
    `tests/test_templates_scaffold_the_current_vocabulary.py` pins this file as one of
    exactly two legacy minters, so a third cannot appear.

    Raises:
    ValueError: If `deployment_type` is not one of the valid types, or if
        `script_path` names the NEW vocabulary's file.

    Returns:
    bool: True if the script was written and compiled successfully, False otherwise.
    """
    if script_path.name == "config_maturity.py":
        raise ValueError(
            "template_config_deployment writes the RETIRED vocabulary "
            "(deployment_status / get_deployment_config), so it must not be written to "
            "config_maturity.py — the loader picks the entry point from the filename, and "
            "the result would load as None (ADR-057, #498). Use "
            "template_config_maturity for that file."
        )

    valid_types = {"shadow", "deployed", "baseline", "deprecated"}
    if deployment_type.lower() not in valid_types:
        logging.error(
            f"Invalid deployment_type: {deployment_type}. Must be one of {valid_types}."
        )
        raise ValueError(
            f"Invalid deployment_type: {deployment_type}. Must be one of {valid_types}."
        )

    deployment_config = {"deployment_status": deployment_type.lower()}

    # Merge additional settings if provided
    if additional_settings and isinstance(additional_settings, dict):
        deployment_config.update(additional_settings)

    # Generate the script code
    code = f"""\"\"\"
Deployment Configuration Script

This script defines the deployment configuration settings for the application. 
It includes the deployment status and any additional settings specified.

Deployment Status:
- shadow: The deployment is shadowed and not yet active.
- deployed: The deployment is active and in use.
- baseline: The deployment is in a baseline state, for reference or comparison.
- deprecated: The deployment is deprecated and no longer supported.

Additional settings can be included in the configuration dictionary as needed.

\"\"\"

def get_deployment_config():
    # Deployment settings
    deployment_config = {deployment_config}
    return deployment_config
"""
    return save_python_script(script_path, code)
