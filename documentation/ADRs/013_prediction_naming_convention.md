# Prediction Naming Convention


| ADR Info            | Details                      |
|---------------------|------------------------------|
| Subject             | Prediction Naming Convention |
| ADR Number          | 013                          |
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
predictions_<run_type>_<timestamp>_<series_sequence_number>.parquet
```
- timestamp: The timestamp when the model was trained **(not when the prediction was generated)**. The format is`YYYYMMDD_HHMMSS`.
- series_sequence_number: When run type is calibration or validation, it produces a list of predictions, each of which is predicted _n_ step head (_n_ ranging from 0 to the maximum forecast step). So the sequence has the same range as steps.

The prediction naming convention for using an ensemble model will follow this structure:
```
predictions_<run_type>_<timestamp>_<series_sequence_number>.parquet
```
- model_name: The name of the model used for the ensemble prediction.
- timestamp: The timestamp when **the prediction was generated**. The format is`YYYYMMDD_HHMMSS`.
- series_sequence_number: When run type is calibration or validation, it produces a list of predictions, each of which is predicted _n_ step head (_n_ ranging from 0 to the maximum forecast step). So the sequence has the same range as steps.

### When run type is forecasting
The prediction naming convention for using a single model will follow this structure:
```
predictions_<run_type>_<timestamp>.parquet
```
- timestamp: The timestamp when the model was trained **(not when the prediction was generated)**. The format is`YYYYMMDD_HHMMSS`.

The prediction naming convention for using an ensemble model will follow this structure:
```
predictions_<run_type>_<timestamp>.parquet
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
- Including the timestamp makes it easy to log files for generated data (see ADR-018).
- Including the timestamp also helps distinguish between multiple runs of the same model, ensuring that no prediction is accidentally overwritten.
- This structure is easy to parse by both humans and automated systems, improving workflow integration and automation.

### Considerations
- **Timestamp Format**: Using `YYYYMMDD_HHMMSS` aligns with standard formats but could introduce issues in systems operating across different time zones.
- **Model timestamp vs. Prediction timestamp**: The timestamp is the model training time, not prediction generation time. This was decided and implemented.

### Multi-target models (amendment 2026-05-06)

When a model produces predictions for multiple targets, the filename must include the
target identifier to prevent file collisions:

**Calibration / Validation:**
```
predictions_<run_type>_<timestamp>_<target_identifier>_<series_sequence_number>.parquet
```

**Forecasting:**
```
predictions_<run_type>_<timestamp>_<target_identifier>.parquet
```

Single-target models and ensemble models omit the target identifier, preserving backward
compatibility with the original format above.

### Implementation Notes (2026-04-08, amended 2026-05-06)

- **Canonical implementation**: `PredictionFileNamer` (in `managers/prediction/file_namer.py`) delegates to `generate_output_file_name()` in `files/utils.py`.
- **File format migration**: The original ADR specified `.pkl` (pickle). The codebase has migrated to `.parquet` as the default format, controlled by `PipelineConfig.dataframe_format`. This ADR has been updated to reflect the current format.
- **Sequence number formatting**: Zero-padded to 2 digits (e.g., `_03` not `_3`).
- **Target identifier**: `generate_output_file_name()` accepts an optional `target_identifier` parameter (default `None`). When `None`, the filename format is unchanged from the original ADR.
