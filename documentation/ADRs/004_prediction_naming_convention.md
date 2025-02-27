# Prediction Naming Convention


| ADR Info            | Details                      |
|---------------------|------------------------------|
| Subject             | Prediction Naming Convention |
| ADR Number          | 004                          |
| Status              | Accepted                     |
| Author              | Xiaolong                     |
| Date                | 11.09.2024                   |

## Context
In the context of the VIEWS pipeline, a standardized naming convention is required to ensure consistency, traceability, and clarity. 
This is particularly important for managing prediction versions, tracking when predictions were generated, and easily identifying which model artifact and dataset were used to create the predictions.


## Decision
### When run type is calibration or validation
The prediction naming convention for using a single model will follow this structure:
```
predictions_<run_type>_<timestamp>_<series_sequence_number>.pkl
```
- timestamp: The timestamp when the model was trained **(not when the prediction was generated)**. The format is`YYYYMMDD_HHMMSS`.
- series_sequence_number: When run type is calibration or validation, it produces a list of predictions, each of which is predicted _n_ step head (_n_ ranging from 0 to the maximum forecast step). So the sequence has the same range as steps.

The prediction naming convention for using an ensemble model will follow this structure:
```
predictions_<run_type>_<timestamp>_<series_sequence_number>.pkl
```
- model_name: The name of the model used for the ensemble prediction.
- timestamp: The timestamp when **the prediction was generated**. The format is`YYYYMMDD_HHMMSS`.
- series_sequence_number: When run type is calibration or validation, it produces a list of predictions, each of which is predicted _n_ step head (_n_ ranging from 0 to the maximum forecast step). So the sequence has the same range as steps.

### When run type is forecasting
The prediction naming convention for using a single model will follow this structure:
```
predictions_<run_type>_<timestamp>.pkl
```
- timestamp: The timestamp when the model was trained **(not when the prediction was generated)**. The format is`YYYYMMDD_HHMMSS`.

The prediction naming convention for using an ensemble model will follow this structure:
```
predictions_<run_type>_<timestamp>.pkl
```
- model_name: The name of the model used for the ensemble prediction.
- timestamp: The timestamp when **the prediction was generated**. The format is`YYYYMMDD_HHMMSS`.


## Consequences
**Positive Effects:**

- **Easier File Management**: Simplifies handling of prediction files, especially when dealing with multiple models or datasets.
- **Improved Traceability**: Facilitates identification of which model produces the prediction.
- **Enhanced Automation**: Enables smooth automation of tasks like archiving or fetching the latest predictions, as the timestamp provides a clear indicator of file recency.


**Negative Effects:**
- **Longer File Names**: Could be cumbersome in environments where shorter names are preferred.
- **Adjustment Required**: Existing scripts or systems may need updates to accommodate the new naming structure.

## Rationale
The decision to use this naming convention ensures that:

- Each file name is unique and informative, allowing easy identification of time of creation without needing to open the file.
- Including the timestamp makes it easy to log files for generated data (see ADR 009).
- Including the timestamp also helps distinguish between multiple runs of the same model, ensuring that no prediction is accidentally overwritten.
- This structure is easy to parse by both humans and automated systems, improving workflow integration and automation.

### Considerations
- **Timestamp Format**: Using `YYYYMMDD_HHMMSS aligns with standard formats but could introduce issues in systems operating across different time zones.
- **Model timstamp vs. Prediction timestamp**: The decision hasn't been made yet on whether the prediction timestamp should be the time the prediction was generated or the time the model was trained. This will be discussed further.
