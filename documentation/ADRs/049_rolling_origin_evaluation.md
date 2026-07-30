# ADR-049: Rolling-Origin Evaluation Protocol

**Status:** Accepted
**Date:** 2026-04-08
**Deciders:** Project maintainers

---

## Context

The VIEWS pipeline evaluates forecasting models using a **rolling-origin
cross-validation** protocol. A model produces 36-month forecasts; the evaluation
repeats this from 13 different starting points (origins), each shifted forward by
one month. This produces 13 overlapping forecast sequences that can be scored
against historical actuals.

The rolling-origin protocol is the single most important evaluation methodology in
the pipeline. It determines how model performance is measured, how evaluation
metrics are aggregated, and how the test partition must be sized. Despite this
centrality, the protocol was never formally documented. The formula was duplicated
across `model.py` and `evaluation/stage.py` (C-39), the constants were scattered
across `core_config_sniffer.py`, and the relationship between partition bounds and
sequence counts was implicit.

Domain story S-03 introduced `ForecastHorizon` as a value object to consolidate the
formula. This ADR formalises the protocol itself so that the domain object, the
sniffer validation, and the evaluation orchestration all share a documented contract.

---

## Decision

### 1. Protocol parameters

The rolling-origin evaluation is fully specified by three parameters:

| Parameter | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| `time_steps` | T | 36 | Months forecast ahead per sequence |
| `stride` | d | 1 | Months between consecutive origins |
| `max_shift_count` | K | 12 | Number of forward shifts from base origin |

Derived quantities:

| Quantity | Formula | Default value |
|---------|---------|---------------|
| Number of sequences | K + 1 | 13 |
| Required test partition length | T + K | 48 months |

### 2. Base-origin resolution

The **base origin** is the month immediately before the first prediction month. Its
value depends on the run type:

| Run type | Base origin formula | Rationale |
|----------|--------------------|-----------|
| Calibration / Validation | `test_partition.start - 1` | Anchor to the test partition boundary |
| Forecasting | `data_loader.month_last` | Anchor to the latest available data |

The calibration formula `test_partition.start - 1` is always correct regardless of
whether a gap exists between the train and test partitions. The previous formula
`train_partition.end` was incorrect when gaps existed (off-by-one bug fixed in
commit `463b413`).

### 3. Step-mapping formula

Each sequence `i` (for i = 0, 1, ..., K) produces a mapping from month IDs to
lead-time steps:

```
mapping_i = { base_origin + i*d + s : s  for s in range(T) }
```

With default parameters (T=36, d=1, K=12):
- Sequence 0: months [base_origin+1, base_origin+36] mapped to steps [0, 35]
- Sequence 1: months [base_origin+2, base_origin+37] mapped to steps [0, 35]
- ...
- Sequence 12: months [base_origin+13, base_origin+48] mapped to steps [0, 35]

### 4. Single source of truth

The canonical implementation of this formula is `ForecastHorizon.build_step_mappings()`
in `views_pipeline_core/domain/horizon.py`. Both `model.py` and `evaluation/stage.py`
currently duplicate this formula (C-39). Story S-07 will replace both call sites with
`ForecastHorizon.build_step_mappings()`.

Until S-07 is complete, changes to the rolling-origin formula must be applied to all
three locations:

1. `domain/horizon.py` — canonical definition
2. `managers/model/model.py:1719-1722` — legacy duplication
3. `managers/evaluation/stage.py:251-253` — stage duplication

### 5. Partition validation

`CoreConfigSniffer._check_evaluation_contract()` validates that the test partition
is sized correctly for the configured horizon:

```
test_length = test_end - test_start + 1  (inclusive)
required    = time_steps + max_shift_count  (= 48 for defaults)
```

If `test_length != required`, the sniffer raises `NotImplementedError`. This
validation runs before any model execution begins.

### 6. Temporal contiguity assumption

The step-mapping formula assumes month IDs within the test partition are contiguous
(no gaps). This is a **Critical Business Rule** of VIEWS conflict forecasting: the
system monitors armed conflict continuously, and temporal gaps in the observation
record would indicate data corruption, not sparse conflict data.

This assumption is validated indirectly by `TemporalPartition` (S-02) which enforces
`start <= end` and by the partition length check above. Explicit contiguity
validation (checking every month_id between start and end exists in the data) is
deferred to S-06.

### 7. Legacy compatibility

The `legacy_compatibility=True` flag passed to the evaluator (C-29) controls
step-wise metric truncation: when enabled, metrics are computed only on months
present in all sequences (shortest-sequence truncation). This preserves the
behaviour of the deleted `EvaluationManager` wrapper and is a domain policy decision
about how to handle partial-coverage evaluation windows. Flipping this flag to
`False` requires numeric equivalence verification.

---

## Consequences

### Positive

- The rolling-origin protocol is formally documented for the first time. New
  contributors can understand the 13-sequence structure without reverse-engineering
  the code.
- `ForecastHorizon.build_step_mappings()` provides a testable, single-source
  implementation of the formula.
- The partition length requirement (48 months for defaults) is derivable from the
  protocol parameters rather than being a magic number.
- Base-origin resolution is explicitly documented per run type, preventing
  recurrence of the off-by-one bug.

### Negative

- The formula is still duplicated in two locations until S-07 is complete.
- `SUPPORTED_TIME_STEPS = {36}` and `SUPPORTED_STRIDES = {1}` in the config sniffer
  mean the protocol currently supports only one configuration. Extending to other
  horizons requires updating both the constants and the tests.
- The `legacy_compatibility` flag remains an unresolved domain policy decision.

---

## Rationale

Rolling-origin cross-validation is standard practice in time-series forecasting
evaluation (Tashman 2000, Hyndman & Athanasopoulos 2021). The VIEWS-specific
parameters (T=36, d=1, K=12) reflect the 3-year forecast horizon used for conflict
prediction and the 4-year test partition available in the VIEWS dataset. The
relationship `required_test_length = T + K` is a mathematical necessity: K forward
shifts of a T-length window require T + K months of test data.

The formula consolidation via `ForecastHorizon` follows the Domain-Driven Design
principle that business rules should live in the domain layer, not be scattered
across orchestration code. The existing duplication (C-39) is a maintenance hazard:
changing the formula in one location but not the other would produce silently
incorrect evaluation results.

---

## References

- **ADR-041:** Sniffer Pattern (validation of partition bounds)
- **ADR-042:** PredictionFrame Adoption (transport format for sequences)
- **ADR-045:** Pipeline Stage Architecture (EvaluationStage owns orchestration)
- **C-17:** Temporal contiguity assumption
- **C-29:** Legacy compatibility flag
- **C-39:** Step-mapping formula duplication
- **S-03:** ForecastHorizon domain value object
- **S-07:** Consolidate step-mapping via ForecastHorizon
