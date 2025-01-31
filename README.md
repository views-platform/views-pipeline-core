![GitHub License](https://img.shields.io/github/license/views-platform/views-pipeline-core)
![GitHub branch check runs](https://img.shields.io/github/check-runs/views-platform/views-pipeline-core/main)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/views-platform/views-pipeline-core)
![GitHub Release](https://img.shields.io/github/v/release/views-platform/views-pipeline-core)

<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://pbs.twimg.com/profile_banners/1237000633896652800/1717069203/1500x500" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>


# Welcome to views-pipeline-core repository! 

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


### Key Functions


1. **Argument Validation:** These checks ensure that the user does not provide incompatible or missing arguments when interacting with the pipeline that could cause errors during execution. This also allows for more intuitive and straightforward pipeline execution.
2. **Error Handling** – Provides error messages and instructions to assist users in providing correct arguments. Attempting to use the --ensemble flag with the --sweep flag results in an error message and program exit. Instructions following the error message ensure that users can navigate through the pipeline in a simple and easily understandable manner.
3. **Guidance for Correct Usage** – Error messages provide suggestions for resolving the issue. In addition to the detailed error messages, the offered suggestions help users understand how to use the command-line interface correctly and ensures that the pipeline runs smoothly.

---

## Configs

### Drift Detection

Relying on large amounts of data for our forecasts, it is crucial to monitor any possible anomalies which may occur in the input data for our models, also known as **data drift**. Data drift is defined as changes in the statistical properties of data that occur over time and can have an impact on machine learning model performance. Thus, detecting and addressing data drift is critical to maintaining model accuracy and reliability. In the VIEWS pipeline, the `drift_detection.py` file includes a dictionary (drift_detection_partition_dict) that sets thresholds and parameters for detecting data drift.


### Logging

As a logging system plays a central role in monitoring and efficiency in the execution of the VIEWS pipeline, we have carefully considered all aspects, such as what elements are logged, where these logs are stored or the amount of detail the logs entail. These specific aspects can be easily configured by changing the `logging.yaml` file which directly configures the pipeline's logging system.

### Pipeline

The overall setting and properties of the pipeline are centralized in the `PipelineConfig` class, which can be found in the `pipeline.py` file.  This class contains methods for accessing and modifying global settings, which ensures consistency and ease of maintenance.

---


## Dataloaders

To be able to handle and manage all the input data, the VIEWS pipeline relies on the `ViewsDataLoader` class. This class contains methods and functionality for managing the data operations in the pipeline. This includes retrieving data, validating data partitions, managing drift detection settings, and ensuring data consistency. The `ViewsDataLoader` class can be found in the `dataloaders.py` file. As its main focus is on the core component of data, it is necesary for the `ViewsDataLoader` to be able to seamlessly interact with other pipeline components. 

### How Dataloaders Fit Into The Pipeline


1. **Data Ingestion and Preprocessing** – The `ViewsDataLoader` class collects and loads data from [viewser](https://github.com/prio-data/viewser) as a [pandas dataframe](https://pandas.pydata.org/docs/index.html). By providing methods for validating data partitions and handling drift detection configurations, the `ViewsDataLoader` class ensures data consistency and quality.
2. **Integration** – The `ViewsDataLoader` class works with other pipeline components, including [`ModelPathManager`](#modelpathmanager) in order to manage model paths, as well as drift detection through configurations for monitoring data drift. This integration ensures that data operations are connected to other pipeline components, such as model training and evaluation, allowing for smooth execution.  
3. **Logging and Debugging** – The class additionally utilizes logging to track data operations and gain insight into the loading process. This is useful for debugging and tracking the pipeline's performance.
4. **Maintaining Consistency and Quality** – The `ViewsDataLoader` class ensures accurate data loading, validation, and processing. As such, it is crucial for ensuring and increasing the model reliability and validity.

---

## Managers


The VIEWS pipeline includes several management classs in order for the pipeline execution to be as automated as possible, while aslo maintaining and ensuring consistency and accuracy in the forecasting processes. The pipeline management classes include `ModelPathManager`, `EnsamblePathManager`, `ModelManager`, `EnsembleManager`, and `PackageManager`.


### ModelPathManager

The `ModelPathManager` class manages the paths and the directories associated with the VIEWS models.

1. **Initialization and Directory Management** – The `ModelPathManager` class manages the paths and directories associated with a model. It creates various directories for the model, including configurations, data, artifacts, and scripts.  
2. **Script Management** – The class also defines methods for initializing and managing script paths required by the model. This includes configuration scripts, main scripts, querysets and various utility scripts. 
3. **Artifact Management** – The class provides methods for managing model artifacts, including retrieving the latest path and handling artifact files.  

### EnsemblePathManager


Similarly to the `ModelPathManager`, the  `EnsamblePathManager` class manages the paths and directories associated with the VIEWS ensemble models. The class entails the same functionality as the `ModelPathManager`, adapted to the VIEWS ensembles. 

1. **Initialization and Directory Management** – The `EnsamblePathManager` class manages and validates and tracks various
directories for the ensemble, including configurations, data, artifacts, and scripts.   
2. **Script Management** – The class also defines methods for initializing and managing script paths required by the ensemble. This includes configuration scripts, main scripts, querysets and various utility scripts. 
3. **Artifact Management** – The class provides methods for managing ensemble artifacts, including retrieving the latest path and handling artifact files. 


### ModelManager

Consistency is crucial when developing our forecasting models. In order to monitor and manage the processes, the `model.py` file in the managers directory is crucial for views-pipeline-core. This file defines the `ModelManager` class, which follows and manages the lifecycle of machine learning models, including core aspects such as configuration, training, evaluation, and producing forecasts. The `ModelManager` class ensures that these processes are consistent, as well as that our end-users smoothly navigate through the VIEWS pipeline.

1. **Initialization** - The `ModelManager` class is initialized with a `ModelPathManager` instance and optional Weights&Biases  notifications flag. It allows for configurationg of various attributes related to model and data loading.
2. **Configuration Loading** - Additionally, the class also offers methods for loading model-specific configurations, including deployment, hyperparameters, metadata, and sweeps.
3. **Model Lifecycle Management** - The `ModelManager` class oversees the entire lifecycle of a machine learning model, including training, evaluation, and forecasting. The class ensures that these processes are followed through with consistency and accuracy.

### How ModelManager Fits Into The Pipeline

1. **Model Lifecycle Management** - The main task of the `ModelManager` class is to manage the entire lifecycle of machine learning models, including configuration loading, training, evaluation, and forecasting. The `ModelManager` ensures that models are managed consistently and efficiently across the entire pipeline. Additipnally, this removes any risk of faulty modelling procedures. 
2. **Integration with Other Components** - Apart from managing the models themselves, the `ModelManager` class works with other pipeline components, including `ModelPathManager` which directly deals with paths, as well as the `ViewsDataLoader` for any data related tasks. This integration allows for, and ensures that the models are trained and evaluated with the proper parameters and data.
3. **Configuration Management** -  The `ModelManager` class also deals with any model configurations such as loading deployment, hyperparameters, metadata, and sweep configurations to ensure proper model setup. This is crucial in keeping the pipeline reproducible.
4. **Logging and Debugging** - The class is also relies on logging in order to be able to track training, evaluation, and
forecasting progress, while providing insights for potential debugging and monitoring the pipeline behavior.



### EnsembleManager


The views-pipeline-core relies extensively on the `EnsembleManager` class, which can be found in the `ensemble.py` file located in the managers directory. Mirroring the functionality of the `ModelManager` class, the `EnsembleManager` is tasked with managing the lifecycles of all ensemble models, including configuration loading, training, evaluation, and forecasting.

1. **Initialization** - The `EnsembleManager` class is initialized with a `EnsamblePathManager` instance and optional Weights&Biases  notifications flag. It allows for configurationg of various attributes related to ensemble and data loading.
2. **Configuration Loading** - Additionally, the class also offers methods for loading ensemble-specific configurations, including deployment, hyperparameters, metadata, and sweeps.
3. **Model Lifecycle Management** - The `EnsembleManager` class oversees the entire lifecycle of a machine learning ensemble, including training, evaluation, and forecasting. The class ensures that these processes are followed through with consistency and accuracy.

### How EnsembleManager Fits Into The Pipeline

1. **Model Lifecycle Management** - The main task of the `EnsembleManager` class is to manage the entire lifecycle of machine learning ensembles, including configuration loading, training, evaluation, and forecasting. The `EnsembleManager` ensures that model ensembles are managed consistently and efficiently across the entire pipeline. Additipnally, this removes any risk of faulty modelling procedures. 
2. **Integration with Other Components** - Apart from managing the models themselves, the `EnsembleManager` class works with other pipeline components, including `EnsamblePathManager` which directly deals with paths, as well as the `ViewsDataLoader` for any data related tasks. This integration allows for, and ensures that the models are trained and evaluated with the proper parameters and data.
3. **Configuration Management** -  The `EnsembleManager` class also deals with any model ensemble configurations such as loading deployment, hyperparameters, metadata, and sweep configurations to ensure proper ensemble setup. This is crucial in keeping the pipeline reproducible.
4. **Logging and Debugging** - The class is also relies on logging in order to be able to track training, evaluation, and
forecasting progress, while providing insights for potential debugging and monitoring the pipeline behavior.

### PackageManager

In addition to `ModelManager` and `EnsembleManager`, views-pipeline-core also includes a `PackageManager`. The `PackageManager` is located in the `package.py` file, and includes methods and functionalities for handling the [Python Poetry](https://python-poetry.org/) packages. This includes everything from creating new packages, validating existing ones, adding dependencies, to getting the most recent release versions from GitHub.

### How PackageManager Fits Into The Pipeline

1. **Package Management** - The `PackageManager`class manages the lifecycle of Python Poetry packages in the pipeline. This includes creating new VIEWS packages, validating existing ones, and managing dependencies. This ensures that all of the packages belonging to VIEWS are properly organized and always up to date.
2. **Dependency Management** - The `PackageManager` class provides methods to add dependencies, as well as retrieve the latest release versions from GitHub. This directly ensures the pipeline relies on the most recent and compatible versions of its dependencies, which is critical for keeping the pipeline stable and functional.
3. **Integration with Other Components** -  The `PackageManager` class works with other pipeline components, including the `ModelPathManager`, in order to manage paths and directories. This integration allows for all the package management processes to be seamlessly connected to the rest of the VIEWS pipeline.
4. **Logging and Debugging** - Similarly to other views-pipeline-core classses,`PackageManager` also relies on logging to track the progress of package management tasks, providing insights for debugging and monitoring, as well as preventing any discrepancies. 

---

## Utils for Weights & Biases

Being able to track model performance is of utmost priority in the VIEWS pipeline. For this reason, the VIEWS pipeline is integrated with the [Weights&Biases(W&B)](https://wandb.ai/site/) platform. This allows us to track the evaluation metrics of all models and ensembles. The `utils.py` file contains utility functions for monitoring all evaluation metrics on W&B. The available functions assist in organizing and updating log dictionaries with evaluation metrics, which are critical for tracking and understanding model performance over time and across datasets.


### How Utils Fit Into The Pipeline

### Key Functions

1. **Experiment Tracking** – The utility functions located in the `utils.py` file are essential for integration with W&B, which tracks experimental models and processes in the pipeline. By logging evaluation metrics in W&B, the pipeline allows for monitoring model performance across multiple experiments, in turn simplifying comparisons between results and identifying the top-performing models or ensembles.
2. **Model Evaluation** – These functions organize and update log dictionaries with all of the evaluation metrics. By recording and organizing all of the logs, we ensure that the evaluation metrics are recorded in a structured manner, making the interpretation and analysis of the results easier, as well as more straightforward.
3. **Performance Monitoring** - In addition to offering logging model performance metrics, the VIEWS pipeline also generates log dictionaries for both month- and time-series-wise evaluation metrics to allow for tracking model performance over time and across
datasets. This directly aids in identifying trends, patterns, and potential issues with the models, allowing for timely and effective interventions and improvements.
4. **Integration with Other Components** – Utility functions in `utils.py` cn also be utilized by different pipeline components, such as model managers and evaluation scripts, in order to log evaluation metrics in W&B. Including this integration ensures that evaluation metrics are consistently logged and tracked across the pipeline.

---

## Templates

As a way of ensuring compatibility throughout the future developments of the VIEWS pipeline, the views-pipeline-core also includes several predefined templates which can be used to generate configuration files, documentation, and other necessary scripts for package, ensemble and model building. These templates allow for freedom in creating new models or ensembles, while also helping to maintain consistency and standardization across the pipeline while following VIEWS formats and design principles.


---
## Acknowledgements  

<p align="center">
  <img src="https://raw.githubusercontent.com/views-platform/docs/main/images/views_funders.png" alt="Views Funders" width="80%">
</p>

Special thanks to the **VIEWS MD&D Team** for their collaboration and support.  

