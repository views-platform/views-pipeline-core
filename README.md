![GitHub License](https://img.shields.io/github/license/views-platform/views-pipeline-core)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/views-platform/views-pipeline-core)

<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://pbs.twimg.com/profile_banners/1237000633896652800/1717069203/1500x500" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>

# Welcome to view-pipeline-core repository! 

The **views-pipeline-core** contains all the necesary components for execution of the VIEWS pipeline. It is a modular and scalable pipeline that handles data ingestion, preprocessing, model and ensemble training, evaluation, and experiment tracking. In this README reference document you will find of the information about the views-pipeline-core – from the design rationale to all of the individual components in the structure.

## Table of Contents

<!-- toc -->

- [Design Objectives](#design-objectives)
- [CLI](#cli)
- [Configs](#configs)
  - [Drift Detection](#drift-detection)
  - [Logging](#logging)
  - [Pipeline](#pipeline)
- [DataLoader](#dataloader)
- [Managers](#managers)
  - [ModelPathManager](#modelpathmanager)
  - [EnsemblePathManager](#ensemblepathmanager)
  - [ModelManager](#modelmanager)
  - [EnsembleManager](#ensemblemanager)
  - [PackageManager](#packagemanager)
- [Utils for Weights & Biases](#utils-for-weights--biases)

<!-- tocstop -->

---

## Design Objectives

The development of the VIEWS pipeline is driven by several core objectives. 

- **Modularity** – The pipeline is divided into distinct modules, each with a specific task. This makes the codebase more maintainable and extensible.  
- **Reusability** – By separating concerns into different directories and scripts, components can be reused across multiple projects or pipeline stages.  
- **Scalability** – The modular design enables easy scaling. New features or components can be added while maintaining existing functionality.  
- **Maintainability** – The codebase is easier to understand and maintain when concerns are clearly separated and components have well-defined interfaces.  
- **Experiment Tracking** – By integrating with Weights and Biases, experiments are tracked and logged, making them more reproducible and easier to analyze.  
- **Testing** – Automated tests ensure the pipeline's accuracy and reliability over time, making it easier to identify and resolve issues.

---
## views-pipeline-core Contents:
## CLI

The `cli/utils.py` file automates command-line argument parsing and validation. It ensures that the user enters valid and compatible arguments when running various commands in the pipeline. This helps to prevent errors and ensures that the pipeline performs as expected.

### CLI: Key Functions

1. **Argument Validation:** These checks ensure that the user does not provide incompatible or missing arguments when interacting with the pipeline that could cause errors during execution. This also allows for more intuitive and straightforward pipeline execution.
2. **Error Handling** – Provides error messages and instructions to assist users in providing correct arguments. Attempting to use the --ensemble flag with the --sweep flag results in an error message and program exit. Instructions following the error message ensure that users can navigate through the pipeline in a simple and easily understandable manner.
3. **Guidance for Correct Usage** – Error messages provide suggestions for resolving the issue. In addition to the detailed error messages, the offered suggestions help users understand how to use the command-line interface correctly and ensures that the pipeline runs smoothly.

---

## Configs

### Drift Detection

Relying on large amounts of data for our forecasts, it is crucial to monitor any anomalies which may occur in the input data for our models, also known as **data drift**. Data drift is defined as changes in the statistical properties of data that occur over time and can have an impact on machine learning model performance. Thus, detecting and addressing data drift is critical to maintaining model accuracy and reliability. In the VIEWS pipeline, the `drift_detection.py` file includes a dictionary (drift_detection_partition_dict) that sets thresholds and parameters for detecting data drift.


### Logging

As a logging system plays a central role in monitoring and efficiency in the execution of the VIEWS pipeline, we have carefully considered all aspects, such as what elements are logged, where these logs are stored or the amount of detail the logs entail. These specific aspects can be easily configured by changing the `logging.yaml` file which directly configures the pipeline's logging system.

### Pipeline

The overall setting and properties of the pipeline are centralized in the PipelineConfig class, which can be found in the `pipeline.py` file.  This class contains methods for accessing and modifying global settings, which ensures consistency and ease of maintenance.

---

## DataLoader

To be able to handle and manage all the input data, the VIEWS pipeline relies on the `ViewsDataLoader` class. This class contains methods and functionality for managing the data operations in the pipeline. This includes retrieving data, validating data partitions, managing drift detection settings, and ensuring data consistency. The `ViewsDataLoader` class can be found in the `dataloaders.py` file. As its main focus is on the core component of data, it is necesary for the `ViewsDataLoader` to be able to seamlessly interact with other pipeline components. 

### How DataLoader Fits Into The Pipeline

1. **Data Ingestion and Preprocessing** – The ViewsDataLoader class collects and loads data from [viewser](https://github.com/prio-data/viewser) as a [pandas dataframe](https://pandas.pydata.org/docs/index.html). By providing methods for validating data partitions and handling drift detection configurations, the ViewsDataLoader class ensures data consistency and quality.
2. **Integration** – Works with other components like `ModelPathManager`.  
3. **Logging and Debugging** – Tracks data operations.  
4. **Maintaining Consistency** – Ensures data quality and consistency.

---

## Managers

### ModelPathManager

1. **Directory Management** – Handles model directories.  
2. **Script Management** – Manages script paths.  
3. **Artifact Management** – Handles artifact retrieval and storage.  

### EnsemblePathManager

Manages ensemble model paths and directories.

### ModelManager

Manages the lifecycle of machine learning models.

### EnsembleManager

Manages the lifecycle of ensemble models.

### PackageManager

Handles Python Poetry packages, dependencies, and releases.

---

## Utils for Weights & Biases

The `utils.py` file provides utility functions for tracking evaluation metrics.

### Key Functions

1. **Experiment Tracking** – Logs evaluations to Weights & Biases.  
2. **Model Evaluation** – Organizes and updates evaluation logs.  
3. **Performance Monitoring** – Tracks trends over time.  
4. **Integration** – Works with model managers for seamless logging.

---

## Templates

Predefined templates for configurations and documentation, ensuring consistency across the pipeline.

## Funding and Partners 

<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative; margin-top: 50px;">
  <img src="image.png" alt="Funder logos" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>


