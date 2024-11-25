# ViEWS Pipeline Core

![GitHub License](https://img.shields.io/github/license/views-platform/views-pipeline-core)
![GitHub branch check runs](https://img.shields.io/github/check-runs/views-platform/views-pipeline-core/main)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/views-platform/views-pipeline-core)
![GitHub Release](https://img.shields.io/github/v/release/views-platform/views-pipeline-core)

<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://pbs.twimg.com/profile_banners/1237000633896652800/1717069203/1500x500" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>

The [**Violence & Impacts Early Warning System (VIEWS)**](https://viewsforecasting.org/) produces monthly predictions of future violent conflict at both a country and sub-country level of analysis. This repository contains code, configuration files, and documentation that encapsulates the entire process of developing, experimenting, training, evaluating, and deploying the VIEWS machine learning model pipeline.

Use our [interactive data dashboard](https://data.viewsforecasting.org/) to explore our latest predictions of future armed conflict.

> [!CAUTION]
> Please note that this pipeline is **actively under construction**. We're in the **early stages of development**, meaning it's **not yet ready for operational use**. We're working hard to bring you a robust and fully-functional tool, so stay tuned for updates!

## Table of contents

<!-- toc -->
- [Repository Contents, Structure, and Explanations](#repository-contents-structure-and-explanations)
- [Pipeline Execution](#pipeline-execution)
- [Pipeline Documentation](#pipeline-documentation)
- [About the VIEWS Project](#about-the-views-project)


<!-- tocstop -->


## Repository Contents, Structure, and Explanations
![VIEWS pipeline diagram](documentation/pipeline_diagram001.png)

### Repository Contents

This repository includes:

- **Code:** Source code for the VIEWS project's machine learning models and the full pipeline.
- **Configuration Files:** Settings and configurations for running the models, ensembles, and orchestration scripts.
- **Documentation:** Detailed instructions and information about the project and how to interact with the pipeline and the individual components.

### Pipeline Overview

The VIEWS machine learning pipeline involves several key processes:

- **Developing:** Creating and refining machine learning models.
- **Experimentation:** Testing and validating various model configurations and approaches.
- **Training:** Training models with relevant data.
- **Evaluating:** Assessing model performance and accuracy.
- **Deploying:** Implementing models in a production environment to generate monthly true-future forecasts 

## Pipeline Documentation
High-level documentation on the pipeline and its components can be found in the folder [`documentation`](https://github.com/views-platform/views-pipeline-core/tree/main/documentation). For a comprehensive understanding of the terms and concepts used, please consult the [`Glossary`](https://github.com/views-platform/views-pipeline-core/blob/main/documentation/glossary.md). To explore the rationale behind our architectural choices, visit the [`Architectural Decision Records (ADRs)`](https://github.com/views-platform/views-pipeline-core/tree/main/documentation/ADRs).

Additionally, refer to READMEs and docstrings of various functions and classes in the source code.

The operational fatalities model generates forecasts for state-based armed conflict during each month in a rolling 3-year window. 
The latest iteration, currently in production, is called [Fatalities002](https://viewsforecasting.org/early-warning-system/models/fatalities002/).

The following links cover **modelling documentation** for Fatalities002:
- [Prediction models and input variables in main ensemble](https://viewsforecasting.org/views_documentation_models_fatalities002/)
- [Levels of analysis and dependent variables](https://viewsforecasting.org/wp-content/uploads/VIEWS_documentation_LevelsandOutcomes.pdf)
- [Partitioning and time shifting data for training, calibration, testing/forecasting, model weighting, and out-of-sample evaluation](https://viewsforecasting.org/wp-content/uploads/VIEWS_Documentation_Partitioningandtimeshifting_Fatalities002.pdf)
- [Ensembling and calibration](https://viewsforecasting.org/wp-content/uploads/VIEWS_documentation_Ensembling_Fatalities002.pdf)

For VIEWS-specific **infrastructure documentation**, please refer to following GitHub repositories:
- [`ingester3`: Loading input data into the views database](https://github.com/UppsalaConflictDataProgram/ingester3)
- [`viewser`: Accessing input data from views database](https://github.com/prio-data/viewser)
- [`views_api`: Our API for accessing predictions](https://github.com/prio-data/views_api)

## About the VIEWS Project

The VIEWS project is a collaborative effort supported by leading research institutions focused on peace and conflict studies. For more information about the project, visit the [VIEWS Forecasting webpage](https://viewsforecasting.org/).

### Affiliations

- **Peace Research Institute Oslo (PRIO):**
  The [Peace Research Institute Oslo (PRIO)](https://www.prio.org/) conducts research on the conditions for peaceful relations between states, groups, and people. PRIO is dedicated to understanding the processes that lead to violence and those that create sustainable peace. About half of the VIEWS core team is currently located at PRIO.

- **Department of Peace and Conflict Research at the University of Uppsala:**
  The [Department of Peace and Conflict Research at the University of Uppsala](https://www.uu.se/en/department/peace-and-conflict-research) is a leading academic institution in the study of conflict resolution, peacebuilding, and security. The department is renowned for its research and education programs aimed at fostering a deeper understanding of conflict dynamics and peace processes. This department also hosts the [Uppsala Conflict Data Program (UCDP)](https://ucdp.uu.se/), a central data source for the VIEWS project. About half of the VIEWS core team is currently located at the University of Uppsala.
