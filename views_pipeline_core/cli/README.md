# VIEWS Pipeline Core: CLI Argument Dataclasses Module

File: `views_pipeline_core/cli/args.py`  
Classes:
- `ModelArgs` (abstract base)
- `ForecastingModelArgs` (concrete implementation for model pipeline runs)
- (Commented) `PreprocessorModelArgs` prototype

Impersonal summary: Provides structured, validated command‑line argument parsing for pipeline execution.

---

## Overview

This module defines a dataclass-based pattern for command-line argument handling in the VIEWS forecasting pipeline. Instead of ad‑hoc `argparse` usage sprinkled across scripts, each pipeline type (forecasting, preprocessing, etc.) owns:
1. A dataclass describing its parameters.
2. Validation logic to enforce legal combinations.
3. Utilities to produce shell commands, dictionaries, and readable string representations.

The design goals are:
- Declarative argument specification (fields + defaults).
- Early failure on incompatible flag combinations.
- Simple integration with path managers for shell script invocation.
- Self-documenting runs (repr shows all parameters).
- Extensibility: add new pipeline types by subclassing `ModelArgs`.

---

## Base Class: ModelArgs

### Purpose
Abstract interface and minimal shared tooling for all argument dataclasses.

### Key Responsibilities
- Define required abstract methods every subclass must implement.
- Provide uniform parsing (`parse_args`), validation trigger (`__post_init__`), and formatting utilities (`__str__`, `__repr__`).
- Offer a controlled exit path (`_exit_with_error`) for invalid argument combinations (prints messages, exits with status code 1).

### Class API

#### Lifecycle
1. `parse_args()` builds an `ArgumentParser` via subclass implementation of `_create_parser`.
2. Parsed `Namespace` is converted to a dataclass instance using `from_namespace`.
3. Post-initialization triggers `_validate()` automatically.

#### Abstract Methods to Implement
| Method | Description |
|--------|-------------|
| `_create_parser()` | Return a fully configured `argparse.ArgumentParser`. |
| `from_namespace(args)` | Convert parsed namespace → dataclass instance. |
| `_validate()` | Raise or exit on illegal flag combinations. |
| `to_shell_command(model_path, script_name)` | Emit runnable shell command list (for subprocess). |
| `get_dict()` | Return dictionary form of all arguments. |

#### Utility
`_exit_with_error(*messages)` prints each message line and exits the process. Chosen over raising exceptions to mirror typical CLI behavior (immediate user feedback).

### Example (Hypothetical Subclass)
```python
class SimpleArgs(ModelArgs):
    mode: str = "run"

    @classmethod
    def _create_parser(cls):
        p = argparse.ArgumentParser()
        p.add_argument("--mode", choices=["run", "test"], default="run")
        return p

    @classmethod
    def from_namespace(cls, ns):
        return cls(mode=ns.mode)

    def _validate(self):
        if self.mode not in ("run", "test"):
            self._exit_with_error("Invalid mode")

    def to_shell_command(self, model_path, script_name="run.sh"):
        return [str(model_path.model_dir / script_name), "--mode", self.mode]

    def get_dict(self):
        return {"mode": self.mode}
```

---

## ForecastingModelArgs

### Purpose
Concrete argument set for the end‑to‑end forecasting pipeline: training, evaluation, forecasting, reporting, and optional operational features (prediction store, drift test, viewser update, monthly shortcut).

### Fields
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `run_type` | str | "calibration" | Pipeline context: calibration, validation, forecasting. |
| `sweep` | bool | False | Engage hyperparameter sweep (implies internal control of train/evaluate). |
| `train` | bool | False | Train a new artifact. |
| `evaluate` | bool | False | Evaluate predictions (not allowed with `run_type=forecasting`). |
| `forecast` | bool | False | Produce future horizon predictions (only with `run_type=forecasting`). |
| `prediction_store` | bool | False | Upload/read predictions via external prediction store integration. |
| `artifact_name` | Optional[str] | None | Use a specific pre-existing artifact instead of latest. |
| `saved` | bool | False | Use locally cached data instead of fresh fetch. |
| `override_timestep` | Optional[int] | None | Override current time index for forecasting (debug/testing). |
| `drift_self_test` | bool | False | Trigger drift detection internal self-test during data fetch. |
| `eval_type` | str | "standard" | Evaluation horizon type: standard, long, complete, live. |
| `report` | bool | False | Generate evaluation or forecast HTML report (requires evaluate or forecast). |
| `update_viewser` | bool | False | Perform incremental raw data update on zero-only months. |
| `wandb_notifications` | bool | False | Enable WandB alerts (run lifecycle + errors). |
| `monthly` | bool | False | Shorthand for production monthly run (sets multiple flags automatically). |

### CLI Construction
`_create_parser()` adds all flags with detailed help messages and combination notes for user guidance. This ensures consistent messaging across environments (local dev, CI, production triggers).

### Validation Logic Highlights (`_validate`)
Enforced constraints (exits with guidance):
1. Monthly flag cannot coexist with `--sweep` or `--evaluate`.
2. `--report` requires either `--evaluate` or `--forecast`.
3. Sweeps:
   - Must use `run_type=calibration`.
   - Cannot manually set `--train`, `--evaluate`, `--forecast`.
4. Forecasting runs cannot include `--evaluate`.
5. No-op protection: if no actionable flags (train/evaluate/forecast/sweep/report) → exit.
6. Mutual exclusivity: `--train` and `--artifact_name`.
7. `--forecast` requires `run_type=forecasting`.
8. If neither train nor sweep is set, `--saved` must be present (otherwise nothing to operate on).
9. `--eval_type` restricted to defined set.
10. `--prediction_store` only valid with `--forecast`.
11. Monthly shorthand auto-sets: forecasting + train + forecast + report + prediction_store + wandb_notifications.

These rules prevent ambiguous pipeline states (e.g., evaluating a forecasting run, generating a report without source predictions).

### Shell Command Generation
`to_shell_command(model_path, script_name="run.sh")` builds a reproducible executable list:
```python
[
  "/absolute/path/models/purple_alien/run.sh",
  "--run_type", "calibration",
  "--train",
  "--evaluate",
  "--eval_type", "standard",
  "--saved"
]
```
Only flags actually set appear. Optional parameters (`artifact_name`, `override_timestep`) appended when provided.

### Dictionary Export
`get_dict()` returns a full snapshot suitable for logging, serialization, or embedding in artifacts:
```python
{
  "run_type": "validation",
  "sweep": False,
  "train": True,
  "evaluate": True,
  ...
}
```

### Usage Examples

#### Basic Calibration Training & Evaluation
```python
from views_pipeline_core.cli.args import ForecastingModelArgs

args = ForecastingModelArgs.parse_args()  # From CLI normally
# Simulate manual creation:
args = ForecastingModelArgs(run_type="calibration", train=True, evaluate=True, saved=True)

print(args)
# ForecastingModelArgs({'run_type': 'calibration', 'sweep': False, 'train': True, ...})

cmd = args.to_shell_command(model_path)
# ['/path/to/models/purple_alien/run.sh', '--run_type', 'calibration', '--train', '--evaluate', '--saved', '--eval_type', 'standard']
```

#### Monthly Production Shortcut
```bash
python run.py --monthly
```
Expands internally to:
- `--run_type forecasting`
- `--train`
- `--forecast`
- `--report`
- `--prediction_store`
- `--wandb_notifications`

#### Forecast Only (Using Existing Artifact)
```bash
python run.py --run_type forecasting --forecast --saved --artifact_name calibration_model_20250101_120000.pt
```
Valid scenario: reuse trained calibration artifact for operational forecast generation.

#### Invalid Combination (Example)
```bash
python run.py --run_type forecasting --evaluate
```
Output:
```
Error: Forecasting runs cannot evaluate.
To fix: Remove --evaluate flag when --run_type is 'forecasting'.
```
Process exits with status code 1.

### Recommended Patterns
| Task | Flags |
|------|-------|
| Full calibration cycle | `--run_type calibration --train --evaluate --report` |
| Validation diagnostics | `--run_type validation --train --evaluate --report` |
| Operational forecast | `--monthly` or `--run_type forecasting --train --forecast --report --prediction_store` |
| Forecast reuse (no retrain) | `--run_type forecasting --forecast --saved --artifact_name <name>` |

### Best Practices
- Always include `--saved` when not training to avoid accidental empty operations.
- Use `--override_timestep` only for backtests or debugging (document decisions).
- Separate evaluation types by explicit `--eval_type` to avoid silent default reliance.
- Let monthly shorthand drive production automation—minimize custom flag mixtures for standard ops.

### Extensibility
To introduce a new pipeline type:
1. Subclass `ModelArgs`.
2. Implement parser, namespace conversion, validation, and shell building.
3. Add to CLI dispatch logic in the invoking script.

---

## (Commented) PreprocessorModelArgs Prototype

Although not active, the commented class serves as a template for non-forecast pipelines:
- Adds flags like `--process`, `--output_format`, `--validate_output`.
- Mirrors the same structural approach.
This pattern can be revived by uncommenting and integrating into entrypoint scripts.

---

## FAQ

| Question | Answer |
|----------|--------|
| Why exit instead of raise exceptions? | Standard CLI ergonomics: immediate user feedback without traceback noise. |
| Can I bypass validation for experimentation? | Not recommended; modify `_validate()` temporarily if absolutely necessary. |
| How do sweeps differ from single runs? | Sweeps auto-manage training/evaluation loops—user should not pass train/evaluate/forecast flags. |
| What is eval_type used for? | Determines horizon length and sequence count for evaluation metrics (e.g., 12 vs 36 months). |
| Does monthly imply evaluation? | No—monthly is operational forecasting (train + forecast + report). |
| Can I specify both artifact_name and train? | No—must choose between training a new model or using an existing artifact. |

---

## Integration Notes

| Component | Uses Arguments |
|-----------|----------------|
| `ForecastingModelManager` | Stage execution decisions (train/evaluate/forecast/report) |
| `ConfigurationManager` | Inject `run_type`, `eval_type`, `override_timestep` |
| DataLoader | Partition resolution (run_type drives partition choice) |
| Reporting Module | Conditional report generation (`--report`) |
| WandB | Notifications flag controls alert emission |

---

## Example End-to-End Invocation

```bash
python main.py \
  --run_type calibration \
  --train \
  --evaluate \
  --report \
  --wandb_notifications \
  --eval_type standard \
  --saved
```

Produces:
- Trained artifact (calibration).
- Evaluation predictions + metrics.
- HTML evaluation report.
- WandB run with alerts.

---