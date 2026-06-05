# Model Configuration Structure Reference Document

|ADR Info| Details|
|--------------|-----------|
| Subject      | Config File Structure|
| ADR Number   | 011       |
| Status       | Accepted  |
| Author       | Simon     |
| Date         | 29.07.2024|

## Context

The project's previous configuration management approach, utilizing a single config_model file, was found to be cumbersome and unclear, hindering operational efficiency and maintainability. There was a need to clarify and optimize the configuration structure to improve system architecture and ease future modifications.

## Decision

The decision was made to restructure the model configuration into distinct files, each tailored to specific functionalities aligned with the model's lifecycle stages—training, deployment, and documentation. This involves splitting the previous config_model into several targeted configuration files (see below).

### Configuration Files Overview

Here is a detailed table describing the new configuration files and their respective purposes:

| Configuration File                        | Type            | Scope           | Description                                                                                             |
|-------------------------------------------|-----------------|-----------------|---------------------------------------------------------------------------------------------------------|
| **config_deployment.py**                | Behavioral      | All             | Manages settings for model deployment across various environments, affecting runtime behavior.          |
| **config_hyperparameters.py**           | Operational     | All             | Defines hyperparameters that influence the training process of the model.                              |
| **config_meta.py**                      | Documentation   | All             | Contains metadata about the model, such as the algorithm used and the identity of the creator.          |
| **config_partitions.py**                | Operational     | All             | Defines temporal partition boundaries (train/test splits) per run type.                                 |
| **config_queryset.py**                  | Behavioral      | Models only     | Configures the input data specifications using a viewser queryset format.                              |
| **config_sweep.py**                     | Operational     | Models only     | Specifies methods for conducting hyperparameter sweeps to optimize model performance.                   |
| **config_modelset.py**                  | Operational (Optional) | Ensembles only | Lists constituent models for ensemble pipelines. Present only in ensemble directories. When present, its keys merge into `config_meta` with precedence (collision warning logged). Template: `views_pipeline_core/templates/ensemble/template_config_modelset.py`. |


## Consequences
**Positive Effects:**
- Improved clarity and maintainability of the configuration files.
- Enhanced ability for new developers to understand the system's architecture.
- Streamlined updates and modifications to model behavior without extensive system-wide impacts.

**Negative Effects:**
- Initial overhead of transitioning to a new configuration structure.
- Potential for initial confusion or errors as developers adjust to the new file distribution.

## Rationale

The division of the configuration into specific files is designed to:

- **Operational Configurations:** Operational configurations consist of parameters that directly affect model training and evaluation. Changes to these settings, such as learning rate, number of training epochs, and model-specific hyperparameters, will alter the model's behavior, impacting how it processes and learns from training data.

- **Behavioral Configurations:** Behavioral configurations include parameters that influence the model's deployment and runtime behavior. Modifying these settings affects how the model processes input data, integrates with other systems, and manages operations in real time. This category is critical for the model's adaptation to its operational environment and includes settings for data preprocessing, deployment strategies, and runtime management. It's important to note that changes here may require additional modifications to the model's source code; simply adjusting these configurations does not guarantee correct behavior without considering the specific model type and implementation.

- **Documentation Configurations:** Documentation configurations contain purely informational metadata about the model. Modifying these parameters does not impact the model’s training, behavior, or deployment. They are crucial for documentation purposes, aiding in compliance and maintenance, and include details such as the model’s architecture, purpose, and version.



### Mandatory vs Optional Configurations

The four base configuration files (config_deployment, config_hyperparameters, config_meta, config_partitions) are mandatory for **all** pipeline unit types (models and ensembles). Their absence during path construction with `validate=True` raises `FileNotFoundError`.

Two additional files are mandatory for **models only**: `config_queryset.py` and `config_sweep.py`. These are added in `_initialize_model_specific_scripts()` and are not required for ensembles.

`config_modelset.py` is the first **optional** configuration file. It applies only to ensembles and is silently skipped when absent. This distinction is enforced at the path-management layer: mandatory files are resolved via `_build_absolute_directory` (fail-loud), while optional files use a direct `Path.exists()` check.

### Optional Config Keys in `config_meta.py`

The following keys are optional in `config_meta.py`. When present, they are validated by `CoreConfigSniffer`; when absent, validation is silently skipped (no regression on existing models).

| Key | Valid Values | Purpose |
|-----|-------------|---------|
| `output_scale` | `"log"`, `"natural"` | Declares whether the model returns predictions in log-scale (no internal transform undo) or natural-scale (model undoes transforms internally). Used by `validate_output_scale_consistency()` to detect incompatible scales in ensemble constituent models. See C-158. |
| `evaluation_mode` | `"stochastic"`, `"point"` | Controls whether samples are kept or collapsed during evaluation. |
| `reconciliation` | `"pgm_cm_point"` | Enables hierarchical prediction reconciliation. Requires `reconcile_with`. |

### Considerations

- **Centralization vs. Duplication:** To avoid redundancy, certain information from the documentation configurations, such as levels in config_meta.py, could be used for orchestration. Changes in these settings should not impact model behavior but are crucial for ensuring that documentation influences operational decisions appropriately.

- **Error Handling:** Any modifications in documentation settings that do not align with the model’s operational parameters should generate informative error messages.

## Additional Notes

- **Partition Configurations:** While some current models use a local partition_config, we want to use set_partition from common_utils as a standard approach. This method can be adapted to accommodate unique needs in exceptional cases.
- **Integration of Querysets:** As part of the restructuring, the queryset integration has been moved to config_queryset.py to streamline data handling processes.

## Feedback and Suggestions

Please share your insights and suggestions to further refine our model configuration strategy. Feedback on the integration of documentation configurations in operational contexts and other areas of concern is particularly valuable.
