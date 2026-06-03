# Log file for generated data


| ADR Info            | Details                     |
|---------------------|-----------------------------|
| Subject             | Log File for Generated Data |
| ADR Number          | 018                         |
| Status              | Accepted                    |
| Author              | Xiaolong                    |
| Date                | 09.09.2024                  |

## Context
In the context of the VIEWS pipeline, there is a need to create a log file to ensure that models and data are tracked accurately and meet certain criteria before running. 
These log file, used as metadata for deployment and orchestration,  will provide a detailed record of the data generation process, including the model artifact, the generated data, and the input raw data. 
This is critical to ensure the reliability and reproducibility of model outputs and to prevent outdated or incorrect data from being used in production systems.

For related ADRs on the generation of different log files and other general logging standards/routines, please see the ADRs below:  [NOTE: new relevant ADRs links should be added]

- [025_input_drift_detection_logging](/documentation/ADRs/025_input_drift_detection_logging.md)

- [026_log_files_for_offline_evaluation](/documentation/ADRs/026_log_files_for_offline_evaluation.md)

- [027_log_files_for_online_evaluation](/documentation/ADRs/027_log_files_for_online_evaluation.md)

- [028_log_files_for_model_training](/documentation/ADRs/028_log_files_for_model_training.md)

- [029_log_files_and_realtime_alerts](/documentation/ADRs/029_log_files_and_realtime_alerts.md)

- [034_log_level_standards](/documentation/ADRs/034_log_level_standards.md)

- [035_log_files_for_input_data](/documentation/ADRs/035_log_files_for_input_data.md)


## Decision
This decision involves implementing a logging system for all generated data and enforce ensemble model checks. 
This logging will involve creating a **.txt** log file in each model-specific folder. The log file will contain the following details:
- The name and timestamp of the model artifact that produced the data.
- The timestamp of when the data was generated.
- Possibly the data stamp of when the raw data used was fetched from VIEWS.
- The deployment status of the single model.

Additionally, ensemble models will enforce a set of preconditions before running:
- **Condition 1 (all run types):** The model artifact used must be trained within the current year (after July).
- **Condition 2 (forecasting only):** The generated data must be from the current month.
- **Condition 3 (forecasting only):** The raw data must also have been fetched in the current month.

In the deployment, when one tries to run an ensemble model, a model check must be passed before executing evaluation or forecasting. 
If any of these conditions are not met, the pipeline will automatically shut down and output a clear and verbose warning, detailing where the issue occurred.

### Amendment (2026-06-02): Context-dependent data freshness checks

Conditions 2 and 3 (data freshness) are only enforced for **forecasting** runs with **non-saved** data. They are skipped when:

- **`run_type` is `calibration` or `validation`:** These runs evaluate on fixed historical partitions. The raw data content is determined by the queryset and partition config, not by when it was fetched. Data fetched on May 27 and June 1 from the same queryset at the same `month_id` yields identical bytes.
- **`--saved` is set:** The ensemble reads pre-computed output from constituent models. It never re-fetches raw data or re-generates features. The fetch timestamps in the log file describe a prior computation whose artifacts are already materialized on disk.

Condition 1 (training cycle) continues to apply to all run types, because a model trained in a previous cycle may have been trained on a fundamentally different data vintage.

**Rationale:** The original ADR was written for production forecasting, where data freshness directly affects prediction quality. Applying calendar-month freshness checks to calibration/validation runs created a predictable failure window at every month boundary for development workflows spanning more than a few days (see issue #150).

## Consequences
**Positive Effects:**
- Improved traceability of generated data, which is essential for debugging, auditing, and reproducing results.
- Ensures models and data used in production are up-to-date and relevant, reducing the risk of using outdated or irrelevant information.
- Automatic shutdown of models that don’t meet the criteria prevents orchestration from becoming a waste of time.

**Negative Effects:**
- Additional efforts on maintaining logs and verifying the conditions of models and data may increase the complexity of the system.

## Rationale
The rationale behind this decision stems from the need for traceability and ensuring data integrity. 
Tracking the model artifacts and their corresponding data ensures that each output can be reproduced. 
Furthermore, enforcing time-based checks for models and data helps ensure that outdated information is not used, 
which could negatively impact predictions.

### Considerations
- The conditions for model and data checks need to be clearly defined and communicated to all team members.
- The implementation of these checks may require coordination with the development team to ensure compatibility.