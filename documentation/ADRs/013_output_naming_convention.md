# Output Naming Convention


| ADR Info            | Details                  |
|---------------------|--------------------------|
| Subject             | Output Naming Convention |
| ADR Number          | 013                      |
| Status              | Accepted                 |
| Author              | Xiaolong                 |
| Date                | 11.09.2024               |

## Context
In the context of the VIEWS pipeline, a standardized naming convention is required to ensure consistency, traceability, and clarity. 
This is particularly important for managing prediction versions, tracking when predictions were generated, and easily identifying which model artifact and dataset were used to create the predictions.


## Decision
The evaluation naming convention for a single model follows this structure:
```
eval_<evaluation_schema>_<conflict_type>__<run_type>_<timestamp>.pkl
```
- evaluation_schema: Three ways to calculate metrics - month (month-wise), ts (time-series-wise), and step (step-wise).
- conflict_type: Three categories of armed conflict - sb (state-based), ns (non state-based), and os (one-sided).
- run_type: The type of run (e.g., calibration, validation).
- timestamp: The timestamp when **the model was trained**. The format is`YYYYMMDD_HHMMSS`.

The evaluation naming convention for an ensemble model has the same structure:
```
eval_<evaluation_schema>_<conflict_type>__<run_type>_<timestamp>.pkl
```
- evaluation_schema: Three ways to calculate metrics - month (month-wise), ts (time-series-wise), and step (step-wise).
- conflict_type: Three categories of armed conflict - sb (state-based), ns (non state-based), and os (one-sided).
- run_type: The type of run (e.g., calibration, validation).
- timestamp: The timestamp when **the evaluation is generated**. The format is`YYYYMMDD_HHMMSS`.

## Consequences
**Positive Effects:**

- **Easier File Management**: Simplifies handling of prediction files, especially when dealing with multiple models or datasets.
- **Improved Traceability**: Facilitates identification of which model produces the prediction/ output/ evaluation.
- **Enhanced Automation**: Enables smooth automation of tasks like archiving or fetching the latest predictions, as the timestamp provides a clear indicator of file recency.
- **Less Complicated Names**: Reduces the timestamp in the ensemble evaluation file name to one, minimizing unnecessary complexity.

**Negative Effects:**
- **Longer File Names**: Could be cumbersome in environments where shorter names are preferred.
- **Adjustment Required**: Existing scripts or systems may need updates to accommodate the new naming structure.
- **Timestamo inconsistency**: Differences in timestamp usage between single and ensemble models could lead to confusion.

## Rationale
The decision to use this naming convention ensures that:

- Each file name is unique and informative, allowing easy identification of the model, data version, and time of creation without needing to open the file.
- Including the timestamp makes it easy to log files for generated data (see ADR 009).
- This structure is easy to parse by both humans and automated systems, improving workflow integration and automation.

### Considerations
- **Timestamp Format**: Using `YYYYMMDD_HHMMSS aligns with standard formats but could introduce issues in systems operating across different time zones.
- **Model timstamp vs. Prediction timestamp**: The decision hasn't been made yet on whether the prediction timestamp should be the time the prediction was generated or the time the model was trained. This will be discussed further.
