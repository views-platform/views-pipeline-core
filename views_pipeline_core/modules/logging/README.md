# VIEWS Pipeline Core: Logging Module

File: `views_pipeline_core/modules/logging/logging.py`  
Primary class: `LoggingModule`  
Depends on:  
- `ModelPathManager` (path orchestration)  
- `logging`, `logging.config` (Python standard library)  
- `yaml` (configuration parsing)  
- `importlib.resources` (packaged file access)  

---

## Overview

The `LoggingModule` provides centralized, configurable logging for all runtime layers of the VIEWS Pipeline Core (managers, modules, evaluation, reconciliation, reporting). It loads a packaged `logging.yaml` configuration, injects dynamic model-specific paths, ensures directory existence, and applies global fallbacks if configuration loading fails.

It is designed to meet production requirements:

| Requirement | Implementation |
|-------------|----------------|
| Model-local log isolation | Per-model log directory via `ModelPathManager.logging` |
| Config-driven handlers | YAML configuration parsed and applied (`dictConfig`) |
| Safe fallback | Automatic `basicConfig(INFO)` on failure |
| Dynamic path templating | `{LOG_PATH}` token replacement in handler filenames |
| External SDK noise suppression | Azure SDK level forced to `WARNING` |
| Runtime idempotency | Single initialization per model instance |
| Portability | Packaged config retrieval via `importlib.resources` |

---

## Class: LoggingModule

### Initialization

```python
from views_pipeline_core.managers.model import ModelPathManager
from views_pipeline_core.modules.logging.logging import LoggingModule

mpm = ModelPathManager("purple_alien")
log_module = LoggingModule(mpm)
logger = log_module.get_logger()
logger.info("Logging initialized.")
```

Args:
- `model_path` (`ModelPathManager`): Provides resolved filesystem locations (e.g. `<project_root>/models/purple_alien/logging/`).

Side effects:
- Ensures logging directory exists (`mkdir(parents=True, exist_ok=True)`).
- Prepares internal state for lazy logger setup.

Raises:
- `ValueError` if derived logging path is not a valid `Path`.

Attributes (internal):
- `_default_level` → `logging.INFO` fallback.
- `_logging_is_active` → flag to disable logging entirely (future extensibility).
- `_logging_path` → resolved directory for all file handlers.
- `_logger` → cached `logging.Logger` instance.

---

### Public Methods

#### get_logger
```python
def get_logger() -> logging.Logger
```
Returns a configured logger instance. Performs lazy initialization by calling `_setup_logging()` if not already created.

Returns:
- `logging.Logger`: Root logger (or configured hierarchy from YAML file).

Usage:
```python
logger = log_module.get_logger()
logger.debug("Debug detail.")
logger.error("Failure occurred.")
```

---

### Internal Methods

#### _setup_logging
```python
def _setup_logging() -> logging.Logger
```
Core initialization routine. Steps:
1. Ensure logging path exists.
2. Load YAML config via `_load_logging_config()`.
3. Replace `{LOG_PATH}` tokens in file handlers.
4. Ensure handler directories exist (`_ensure_log_directory`).
5. Apply configuration with `logging.config.dictConfig`.
6. Suppress verbose third-party logging (`azure` → WARNING).
7. Return configured root logger.

On failure:
- Calls `logging.basicConfig(level=self._default_level)`.
- Logs an error explaining fallback cause.

#### _load_logging_config
```python
def _load_logging_config() -> dict
```
Loads `logging.yaml` packaged under `views_pipeline_core.configs`. Parses YAML and returns dict.

Failure modes logged:
- File not found
- YAML syntax error
- Unexpected read exception

Returns:
- Dict (empty on failure; triggers fallback path).

#### _ensure_log_directory
```python
def _ensure_log_directory(log_path: str) -> None
```
Guarantees directory exists for a file-based handler path prior to handler activation.

---

## Example logging.yaml (Typical)

```yaml
version: 1
formatters:
  standard:
    format: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  file_main:
    class: logging.FileHandler
    level: INFO
    formatter: standard
    filename: "{LOG_PATH}/main.log"
  file_errors:
    class: logging.FileHandler
    level: ERROR
    formatter: standard
    filename: "{LOG_PATH}/errors.log"
loggers:
  views_pipeline_core:
    level: INFO
    handlers: [console, file_main, file_errors]
    propagate: False
root:
  level: INFO
  handlers: [console]
```

Token `{LOG_PATH}` is dynamically replaced with the model-specific logging directory.

---

## Usage Patterns

### Basic Integration in a Manager
```python
class MyModelManager:
    def __init__(self, model_path: ModelPathManager):
        self.logging_module = LoggingModule(model_path)
        self.logger = self.logging_module.get_logger()
        self.logger.info(f"Initializing manager for {model_path.model_name}")
```

### Structured Logging in Pipeline Stages
```python
logger.info("Starting training stage.")
try:
    train()
    logger.info("Training completed successfully.")
except Exception as e:
    logger.exception("Training failed with exception.")
    raise
```

### Multi-Handler Severity Separation
- Info-level operational messages → `main.log`
- Error-level stack traces → `errors.log`
- Real-time console output parallel to file persistence

---

## Best Practices

| Practice | Rationale |
|----------|-----------|
| Use `.get_logger()` once per manager | Avoid re-configuring handlers |
| Prefer structured messages (`key=value`) | Simplifies log parsing |
| Use `.exception()` inside except blocks | Includes traceback automatically |
| Avoid logging sensitive data (API keys, credentials) | Compliance & security |
| Include model name in contextual messages | Multi-model pipeline clarity |
| Separate debug-only handlers in development | Reduce production noise |

---

## Extensibility Suggestions

| Requirement | Approach |
|-------------|----------|
| Disable logging dynamically | Flip `_logging_is_active` before setup |
| Add JSON logs | Extend `logging.yaml` with JSON formatter |
| Rotate files | Swap `FileHandler` for `TimedRotatingFileHandler` |
| Remote aggregation (ELK, Loki) | Add `SocketHandler` or custom REST handler |
| Colored console output | Implement custom formatter or use `colorlog` |

---

## Failure Modes & Mitigation

| Symptom | Cause | Mitigation |
|---------|-------|-----------|
| Only console logs present | File handler path invalid | Inspect `{LOG_PATH}` substitution result |
| Fallback config applied | YAML parse failure | Validate YAML with online linter |
| Duplicate log lines | Logger propagation enabled | Set `propagate: False` in YAML logger section |
| Missing error file | Permission or directory creation failure | Check filesystem permissions for model path |

---

## Integration with Other Modules

| Module | Interaction |
|--------|-------------|
| WandBModule | Parallel logging + experiment tracking |
| Evaluation | Log metric aggregation status and anomalies |
| Drift Detection | Log drift alerts before sending notifications |
| Reconciliation | Log task batching, failures, completion summary |
| Reporting | Log report generation start/end + file path |
| Transformation | Log reversible transform lifecycle events |

---

## Security & Compliance

- Writes only within model-controlled directory tree.
- No network operations.
- Sanitization of injected paths enforced via `Path` validation.
- Suitable for production deployment with minimal surface area.

---

## FAQ

| Question | Answer |
|----------|--------|
| Can I change logging level at runtime? | Yes: `logger.setLevel(logging.DEBUG)` or per-handler adjustment. |
| Are handlers recreated on subsequent `get_logger()` calls? | No; initialization is idempotent. |
| How do I silence third-party libraries? | Add their logger names to YAML with higher thresholds. |
| Can I use this outside model context? | Yes—provide a mock `ModelPathManager` with `.logging` attribute. |
| What if I need per-stage log files? | Extend YAML with additional handlers keyed to custom loggers. |

---