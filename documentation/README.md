
# Documentation of VIEWS Pipeline Core

## Overview
This repository contains the core components of the **views_pipeline_core** package, a structured pipeline for data processing, model management, and evaluation. Below is an overview of the directory structure along with descriptions of each component.

## Directory Structure

```
views_pipeline_core/
├── __init__.py                     # Package initialization
├── cli/                             # Command-line interface utilities
├── configs/                         # Configuration files and settings
├── data/                            # Data processing and storage
├── files/                           # File handling utilities
├── managers/                        # Model and ensemble management (see managers/README.md)
├── models/                          # Model definitions and related utilities
├── templates/                       # Template configurations for models and ensembles
├── wandb/                           # Weights & Biases logging and utilities
└── README.md                        # Main documentation
```

## Detailed Breakdown

### `cli/`
Contains utilities for command-line interactions with the pipeline.
- `utils.py`: Helper functions for CLI interactions.

**Detailed documentation available** [here](https://github.com/views-platform/views-pipeline-core/tree/main/views_pipeline_core/cli).

### `configs/`
Holds configuration files for different components of the pipeline:
- `drift_detection.py`: Configuration related to drift detection.
- `pipeline.py`: General pipeline configurations.
- `logging.yaml`: Logging settings.

**Detailed documentation available** [here](https://github.com/views-platform/views-pipeline-core/tree/main/views_pipeline_core/configs).

### `data/`
Handles data storage and loading.
- `dataloaders.py`: Functions for loading datasets.
- `handlers.py`: Data processing utilities.
- `utils.py`: Helper functions for managing data.

**Detailed documentation available** [here](https://github.com/views-platform/views-pipeline-core/tree/main/views_pipeline_core/dataloaders).

### `files/`
Handles file operations and management.
- `utils.py`: File processing utilities.

**Detailed documentation available** [here](https://github.com/views-platform/views-pipeline-core/tree/main/views_pipeline_core/files).

### `managers/`
Handles model and ensemble management. **Detailed documentation available in `managers/README.md`**

**Detailed documentation available** [here](https://github.com/views-platform/views-pipeline-core/tree/main/views_pipeline_core/managers).

### `models/`
Defines models and their outputs.
- `check.py`: Validates model consistency.
- `outputs.py`: Handles model outputs and transformations.

**Detailed documentation available** [here](https://github.com/views-platform/views-pipeline-core/tree/main/views_pipeline_core/models).

### `templates/`
Contains templates for different components of the pipeline, such as models and ensembles.
- `ensemble/`: Configuration templates for ensemble models.
- `model/`: Configuration templates for individual models.
- `package/`: Templates for packaging and deployment.

**Detailed documentation available** [here](https://github.com/views-platform/views-pipeline-core/tree/main/views_pipeline_core/templates).

### `wandb/`
Handles integration with Weights & Biases for experiment tracking.
- `utils.py`: Functions for logging results to W&B.

**Detailed documentation available** [here](https://github.com/views-platform/views-pipeline-core/tree/main/views_pipeline_core/wandb).



*For further details, please check the specific README files inside the respective directories.*
















