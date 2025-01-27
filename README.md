![GitHub License](https://img.shields.io/github/license/views-platform/views-pipeline-core)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/views-platform/views-pipeline-core)

# Welcome to view-pipeline-core repository! 

The views-pipeline-core contains all the necesary components for execution of the VIEWS pipeline. It is a modular and scalable pipeline that handles data ingestion, preprocessing, model and ensemble training, evaluation, and experiment tracking. In this README reference document you will find of the information about the views-pipeline-core – from the design rationale to all of the individual components in the structure.

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



## Design Objectives

- **Modularity** – Distinct modules ensure maintainability and extensibility.  
- **Reusability** – Separated concerns allow reuse across projects.  
- **Scalability** – New features can be added with minimal disruption.  
- **Maintainability** – Clear separation of concerns and defined interfaces.  
- **Experiment Tracking** – Integrated with Weights & Biases for tracking.  
- **Testing** – Automated tests for accuracy and reliability.

---

## CLI

The `cli/utils.py` file automates argument parsing and validation.

### Key Functions

1. **Argument Validation** – Ensures valid arguments to prevent errors.
2. **Error Handling** – Provides meaningful error messages.
3. **Guidance for Correct Usage** – Helps users understand correct CLI usage.

---

## Configs

### Drift Detection

The `drift_detection.py` file contains configurations for detecting data drift.

### Logging

The `logging.yaml` file configures the pipeline's logging system.

### Pipeline

The `pipeline.py` file includes a centralized configuration class.

---

## DataLoader

The `dataloaders.py` file includes the `ViewsDataLoader` class.

### How It Fits Into The Pipeline

1. **Data Ingestion and Preprocessing** – Loads data from `viewser` to Pandas.  
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






