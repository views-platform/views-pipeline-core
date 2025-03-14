# VIEWS Pipeline CLI Documentation

## Overview

This CLI tool supports three types of model runs with various operational modes:
- **Run Types**: `calibration`, `validation`, `forecasting`
- **Key Features**:
  - Hyperparameter sweeps
  - Model training & evaluation
  - Forecast generation
  - Artifact management
  - Data configuration options

## Command-Line Arguments

| Short Flag | Long Flag        | Description                                                                 | Valid Values/Type          | Default     | Constraints                                                                 |
|------------|------------------|-----------------------------------------------------------------------------|----------------------------|-------------|-----------------------------------------------------------------------------|
| `-r`       | `--run_type`     | Model execution mode                                                       | calibration/validation/forecasting | calibration | Required for all operations                                                |
| `-s`       | `--sweep`        | Enable hyperparameter sweep                                                | Boolean flag               | False       | Forces run_type=calibration<br>Auto-enables train/evaluate                 |
| `-t`       | `--train`        | Train new model                                                            | Boolean flag               | False       | Conflicts with artifact_name                                               |
| `-e`       | `--evaluate`     | Evaluate model performance                                                 | Boolean flag               | False       | Incompatible with forecasting run_type                                     |
| `-f`       | `--forecast`     | Generate predictions                                                       | Boolean flag               | False       | Requires run_type=forecasting                                              |
| `-a`       | `--artifact_name`| Load specific model artifact                                                | String                     | None        | Format: `<run_type>_model_<YMD_HMS>.pt`                                   |
| `-en`      | `--ensemble`     | Use ensemble model                                                         | Boolean flag               | False       | Incompatible with sweeps                                                   |
| `-sa`      | `--saved`        | Use locally stored data                                                    | Boolean flag               | False       | Required for non-training runs                                             |
| `-o`       | `--override_month`| Override current month context                                            | Integer                    | None        | Forces specific month                                                      |
| `-dd`      | `--drift_self_test`| Enable data drift detection                                               | Boolean flag               | False       | Performs data integrity checks                                            |
| `-et`      | `--eval_type`    | Evaluation mode                                                            | standard/long/complete/live | standard    | Controls evaluation depth                                                  |

## Argument Validation Rules

1. **Sweep Constraints**:
```bash
--sweep => requires --run_type=calibration
--sweep => auto-enables --train and --evaluate
--sweep => incompatible with --ensemble, --forecast, --artifact_name
```

2. **Operational Modes**:
```bash
--evaluate + --run_type=forecasting => Error
--forecast + non-forecasting run_type => Error
--train + --artifact_name => Error
```

3. **Data Requirements**:
```bash
Non-training runs => must use --saved
Month override => must be 1-12
Eval type => must be in approved list
```

## Example Use Cases

1. **Basic Calibration Run**
```bash
python main.py --run_type calibration --train --evaluate
```
* Trains new calibration model
* Performs standard evaluation
* Uses fresh data (no --saved flag)

2. **Hyperparameter Sweep**
```bash
python main.py --sweep --run_type calibration --saved --eval_type long
```
* Runs parameter search using saved data
* Performs extended evaluations
* Stores best performing artifacts

4. **Model Evaluation Only**
```bash
python main.py --run_type validation --evaluate --saved \
    --artifact_name validation_model_20240101_120000.pt \
    --eval_type complete
```
* Full validation suite on specified artifact
* Uses stored validation datasets
* Bypasses training phase

5. **Drift Detection Test**
```bash
python main.py --run_type calibration --train --drift_self_test --override_month 541
```
* Trains March model with fresh data
* Performs data integrity checks
* Uses current data (no --saved)

## Additional Notes

**Artifact Resolution Logic**
| Scenario              | Behavior                                      |
|-----------------------|-----------------------------------------------|
| No --artifact_name    | Uses latest matching run_type artifact        |
| Training mode         | Generates new artifact with timestamp         |
| Evaluation mode       | Requires existing artifact                    |

**Month Handling**
**Default**: Uses current system month     
**Override**: Affects data loading and artifact naming    

**Exit Codes**
* ```0```: Successful execution
* ```1```: Argument validation failure
* All errors produce specific corrective messages
