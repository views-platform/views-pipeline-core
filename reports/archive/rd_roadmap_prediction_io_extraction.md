# R&D Roadmap: Prediction I/O Extraction (Option 1)

**Date:** 2026-03-17
**Scope:** Extract `_save_predictions`, `_save_evaluations`, `_save_model_artifact`, `_generate_evaluation_table` from `ForecastingModelManager` into `PredictionIOManager`
**Lines moved:** ~236 of 2,077 (ForecastingModelManager) / 3,436 (model.py)

---

## 1. Current State

Four I/O methods live inside ForecastingModelManager despite having zero coupling to orchestration logic. They are **leaf nodes** in the call graph — they persist data but never call back into the orchestration core.

| Method | Lines | Fan-in | Purpose |
|--------|-------|--------|---------|
| `_save_predictions()` | 2766-2862 (97) | 4 call sites | Write parquet + upload to prediction store + Appwrite |
| `_save_evaluations()` | 2666-2764 (99) | 1 call site | Write step/time-series/month evaluation metrics |
| `_save_model_artifact()` | 2583-2626 (44) | 1 call site | Upload model artifact to WandB |
| `_generate_evaluation_table()` | 3260-3297 (38) | 1 call site | Format metrics as markdown table |

**Why they're in the wrong place (SRP):** ForecastingModelManager has two reasons to change — orchestration logic AND persistence mechanism. These methods embody the persistence concern exclusively.

**Dependencies consumed:** `self.configs`, `self._model_path`, `self._wandb_module`, `self._datastore`, `self._pred_store_name`, `self._use_prediction_store`

**Dependencies NOT consumed:** No abstract methods, no `self.args` (except for naming in `_save_predictions`), no `self._data_loader`, no `self._partition_dict`

---

## 2. Target State

```
ForecastingModelManager (orchestration only)
  └── self._io: PredictionIOManager (injected)
        ├── save_predictions()
        ├── save_evaluations()
        ├── save_model_artifact()
        └── generate_evaluation_table()
```

`PredictionIOManager` lives in `views_pipeline_core/managers/prediction/io.py`. It receives all its dependencies via constructor — no access to ForecastingModelManager internals.

---

## 3. Why This Extraction First

1. **Directly enables the forecast shipping project.** The PredictionSaver protocol (forecast shipping Task 2) will operate on `PredictionIOManager` — a 236-line focused module — instead of a 3,436-line god class.
2. **Lowest risk of all three extractions.** Leaf nodes only. No call graph restructuring.
3. **No subclass impact.** External model repos (hydranet, stepshifter, baseline) never override these methods.
4. **No EnsembleManager impact.** EnsembleManager calls `_save_predictions()` via inheritance — the delegation pattern preserves this transparently.

---

## 4. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Tests mocking `_save_predictions` break | High | Update mocks to target `self._io.save_predictions` |
| EnsembleManager calling inherited method | Medium | Keep thin delegation method on ForecastingModelManager for backward compat |
| Missing context in PredictionIOManager | Low | Constructor receives explicit deps; no hidden state |

---

## 5. What NOT to change

- Method signatures (same parameters, same return types)
- Method behavior (move code, don't rewrite it)
- The orchestration methods that call these I/O methods
- Abstract method contracts
- External model repo interfaces
