# Class Intent Contract: ForecastingModelArgs

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-04-01
**Related ADRs:** ADR-001 (Ontology of the Repository), ADR-003 (Authority of Declarations), ADR-009 (Boundary Contracts)

---

## 1. Purpose

Frozen dataclass that captures, validates, and serializes CLI arguments for the forecasting model pipeline. Extends the abstract `ModelArgs` base class. Acts as the single source of truth for what a pipeline run has been asked to do, enforcing all constraint combinations in `__post_init__` before any downstream code executes.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** execute pipeline stages; it only declares what should happen.
- Does **not** load data, train models, or evaluate results.
- Does **not** persist itself to disk or database.
- Does **not** interact with WandB, Appwrite, or any external service.
- Does **not** validate model-specific configuration (hyperparameters, target lists, etc.). That is the responsibility of config sniffers.
- Does **not** provide a programmatic builder pattern; construction is via the dataclass constructor or `parse_args()`.

---

## 3. Responsibilities and Guarantees

- **Constraint validation** in `_validate()` (called from `__post_init__`):
  - `monthly` flag auto-sets `run_type="forecasting"`, `train=True`, `forecast=True`, `report=True`, `prediction_store=True`, `wandb_notifications=True`. Rejects `sweep` or `evaluate` when `monthly` is set.
  - `report` requires either `evaluate` or `forecast`. With `run_type="calibration"`, `report` additionally requires `evaluate`.
  - `sweep` requires `run_type="calibration"` and rejects `train`, `evaluate`, and `forecast` flags.
  - `evaluate` is rejected when `run_type="forecasting"`.
  - `forecast` requires `run_type="forecasting"`.
  - `prediction_store` requires `forecast`.
  - At least one action flag (`train`, `evaluate`, `forecast`, `sweep`, `report`) must be set.
  - `train` and `artifact_name` are mutually exclusive.
  - If neither `train` nor `sweep` is set, `saved` must be `True`.
  - `eval_type` must be one of `"standard"`, `"long"`, `"complete"`, `"live"`.
- **`parse_args()`**: Class method that creates an `argparse.ArgumentParser`, parses `sys.argv`, and returns a validated `ForecastingModelArgs` instance.
- **`to_shell_command(model_path, script_name="run.sh")`**: Generates a `List[str]` shell command from the current arguments, suitable for `subprocess` calls.
- **`get_dict()`**: Returns all 14 fields as a plain dictionary.
- **`__str__` / `__repr__`**: Provide human-readable representations via `get_dict()`.

---

## 4. Inputs and Assumptions

- All fields have defaults. `run_type` defaults to `"calibration"`, all booleans default to `False`, `artifact_name` and `override_timestep` default to `None`, `eval_type` defaults to `"standard"`.
- `parse_args()` reads from `sys.argv`. The parser defines short flags: `-r` (run_type), `-s` (sweep), `-t` (train), `-e` (evaluate), `-f` (forecast), `-p` (prediction_store), `-a` (artifact_name), `-sa` (saved), `-o` (override_timestep), `-dd` (drift_self_test), `-et` (eval_type), `-re` (report), `-u` (update_viewser), `-wn` (wandb_notifications), `-m` (monthly).
- `to_shell_command()` expects a `model_path` object with a `.model_dir` attribute (a `Path`).

**Fields (14 total):**

| Field | Type | Default |
|---|---|---|
| `run_type` | `str` | `"calibration"` |
| `sweep` | `bool` | `False` |
| `train` | `bool` | `False` |
| `evaluate` | `bool` | `False` |
| `forecast` | `bool` | `False` |
| `prediction_store` | `bool` | `False` |
| `artifact_name` | `Optional[str]` | `None` |
| `saved` | `bool` | `False` |
| `override_timestep` | `Optional[int]` | `None` |
| `drift_self_test` | `bool` | `False` |
| `eval_type` | `str` | `"standard"` |
| `report` | `bool` | `False` |
| `update_viewser` | `bool` | `False` |
| `wandb_notifications` | `bool` | `False` |
| `monthly` | `bool` | `False` |

---

## 5. Outputs and Side Effects

- Construction either succeeds (returning a validated instance) or calls `sys.exit(1)` via `_exit_with_error()`.
- `parse_args()` reads `sys.argv` (side effect: consumes CLI arguments).
- `to_shell_command()` returns `List[str]`; no side effects.
- `get_dict()` returns `Dict`; no side effects.
- `_validate()` mutates `self` fields when `monthly=True` (sets `run_type`, `train`, `forecast`, `report`, `prediction_store`, `wandb_notifications`). This is the only mutation after construction.

---

## 6. Failure Modes and Loudness

- All constraint violations call `_exit_with_error(*messages)`, which prints error messages to stdout and calls `sys.exit(1)`. This is loud but makes unit testing harder -- tests must catch `SystemExit`.
- Error messages include both the problem description and a "To fix:" suggestion.
- There is no silent fallback or permissive mode. Invalid argument combinations always terminate the process.

---

## 7. Boundaries and Interactions

- **Inherits from**: `ModelArgs` (abstract base class defining `parse_args`, `_create_parser`, `from_namespace`, `_validate`, `to_shell_command`, `get_dict`).
- **Used by**: Pipeline orchestration scripts (e.g., `run.sh` wrappers, model `main.py` files) that parse CLI arguments before dispatching to model managers.
- **Consumed by**: Model managers and data loaders that read the validated args to decide which pipeline stages to execute.
- Has no dependencies on any other pipeline-core class beyond `ModelArgs`.

---

## 8. Examples of Correct Usage

```python
# Programmatic construction
args = ForecastingModelArgs(
    run_type="calibration",
    train=True,
)

# Monthly shorthand
args = ForecastingModelArgs(monthly=True)
assert args.run_type == "forecasting"
assert args.train is True
assert args.forecast is True

# From CLI
args = ForecastingModelArgs.parse_args()  # reads sys.argv

# Generate shell command
cmd = args.to_shell_command(model_path, script_name="run.sh")
# ['/path/to/model/run.sh', '--run_type', 'calibration', '--train', ...]

# Serialize to dict
d = args.get_dict()
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: No action flags (sys.exit(1))
ForecastingModelArgs(run_type="calibration")

# WRONG: sweep + forecast (sys.exit(1))
ForecastingModelArgs(run_type="calibration", sweep=True, forecast=True)

# WRONG: evaluate + forecasting run type (sys.exit(1))
ForecastingModelArgs(run_type="forecasting", evaluate=True, train=True)

# WRONG: train + artifact_name (sys.exit(1))
ForecastingModelArgs(run_type="calibration", train=True, artifact_name="model.pt")

# WRONG: forecast without forecasting run_type (sys.exit(1))
ForecastingModelArgs(run_type="calibration", train=True, forecast=True)

# WRONG: prediction_store without forecast (sys.exit(1))
ForecastingModelArgs(run_type="calibration", train=True, prediction_store=True)

# WRONG: invalid eval_type (sys.exit(1))
ForecastingModelArgs(run_type="calibration", train=True, eval_type="custom")
```

---

## 10. Test Alignment

Tests live in `tests/test_utils/test_forecasting_args.py`. Coverage includes:

- **`TestForecastingModelArgsInit`**: Default initialization fails, valid/custom initialization succeeds.
- **`TestValidationBasicRuns`**: Calibration with train, calibration with evaluate+saved, forecasting, no-action failure, validation and testing run types.
- **`TestValidationMonthly`**: Auto-set flags, sweep conflict, evaluate conflict.
- **`TestValidationSweep`**: Requires calibration, rejects train/evaluate/forecast flags, valid sweep.
- **`TestValidationForecast`**: Requires forecasting run_type, forecasting rejects evaluate, prediction_store requires forecast.
- **`TestValidationReport`**: Requires evaluate or forecast, calibration requires evaluate.
- **`TestValidationArtifact`**: Train/artifact conflict, no-train requires saved.
- **`TestValidationEvalType`**: All four valid types, invalid type rejection.
- **`TestParseArgs`**: Basic args, complex args, monthly shorthand, saved flag (all via `sys.argv` patching).
- **`TestToShellCommand`**: Basic command, complex command, artifact name, excludes false flags.
- **`TestGetDict`**: All fields present, values match attributes.
- **`TestStringRepresentations`**: `__str__` and `__repr__`.
- **`TestForecastingModelArgsIntegration`**: Full calibration, forecasting, and sweep workflows.

All validation failures are tested by catching `SystemExit`.

---

## 11. Evolution Notes

- The `_exit_with_error()` pattern using `sys.exit(1)` is inherited from `ModelArgs`. A future refactor could raise a custom exception instead, making testing cleaner.
- `run_type` accepts `"testing"` in addition to `"calibration"`, `"validation"`, and `"forecasting"` (the parser only defines the latter three as choices, but programmatic construction allows `"testing"`).
- The `monthly` flag mutates fields in `_validate()`, which is unusual for a dataclass. This is a deliberate convenience shorthand.

---

## 12. Known Deviations

- **`_exit_with_error()` calls `sys.exit(1)`**: This makes unit testing harder. Tests must catch `SystemExit` rather than a domain exception. This is inherited from the `ModelArgs` abstract base class.
- **`monthly` mutates fields**: The `_validate()` method sets `run_type`, `train`, `forecast`, `report`, `prediction_store`, and `wandb_notifications` when `monthly=True`. This means the dataclass is not truly frozen/immutable despite being a `@dataclass`.
- **`drift_self_test` field**: Appears in the dataclass and CLI parser but its validation and downstream behavior are minimal; it is simply passed through.

---

## End of Contract

This document defines the **intended meaning** of `ForecastingModelArgs`.
Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
