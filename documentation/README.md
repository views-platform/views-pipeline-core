
# Documentation of VIEWS Pipeline Core

## Overview
This repository contains the core components of the **views-pipeline-core** package, a structured pipeline for data processing, model management, and evaluation. Additionally, this folder includes the [ADR](https://github.com/views-platform/views-pipeline-core/tree/main/documentation/ADRs)s (Architectural Decision Records), which provide further context and details about the architectural decisions made during the development of the package. Below is an overview of the directory structure along with descriptions of each component.

## Directory Structure

```
.github/                              # GitHub configuration files
dist/                                 # Distribution files
documentation/                        # Documentation files
tests/                                # Folder containing tests
views_pipeline_core/                  # Core package
├── __init__.py                       # Package initialization
├── cli/                              # Command-line interface utilities
├── configs/                          # Configuration files and settings
├── data/                             # Data processing and storage
├── files/                            # File handling utilities
├── managers/                         # Model and ensemble management (see managers/README.md)
├── models/                           # Model definitions and related utilities
├── templates/                        # Template configurations for models and ensembles
├── wandb/                            # Weights & Biases logging and utilities
└── README.md                         # Main documentation
.gitignore                            # Git ignore rules 
README.md                             # Main project documentation
image.png                             # Image resources
pyproject.toml                        # Python project configuration

```

## Detailed Breakdown

### `cli/`
Contains utilities for command-line interactions with the pipeline.
- `utils.py`: Helper functions for CLI interactions.

**Detailed documentation available** [here](https://github.com/views-platform/views-pipeline-core/tree/main/views_pipeline_core/cli).

### `configs/`
Holds configuration files for different components of the pipeline:
- `drift_detection.py`: Configuration related to drift detection.
- `pipeline.py`: PipelineConfig class.
- `logging.yaml`: Logging settings.

**Detailed documentation available** [here](https://github.com/views-platform/views-pipeline-core/tree/main/views_pipeline_core/configs).

### `data/`
Handles data storage and loading.
- `dataloaders.py`: ViewsDataLoader class for loading datasets.
- `handlers.py`: ViewsDataset class.
- `utils.py`: Helper functions for managing data.

**Detailed documentation available** [here](https://github.com/views-platform/views-pipeline-core/tree/main/views_pipeline_core/dataloaders).

### `files/`
Handles file operations and management.
- `utils.py`: File processing utilities.

**Detailed documentation available** [here](https://github.com/views-platform/views-pipeline-core/tree/main/views_pipeline_core/files).

### `managers/`
Handles model and ensemble management. 
- `model.py`: ModelPathManager and ModelManager classes.
- `ensemble.py`: EnsemblePathManager and EnsembleManager classes.
- `package.py`: PackageManager class.
- `log.py`: LoggingManager class.

**Detailed documentation available** [here](https://github.com/views-platform/views-pipeline-core/tree/main/views_pipeline_core/managers).

### `models/`
Defines models and their outputs.
- `check.py`: Validates model consistency.
- `outputs.py`: Handles model outputs and transformations with ModelOutputs class.

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
















