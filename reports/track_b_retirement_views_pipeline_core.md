# Track B Retirement: views-pipeline-core

**Date:** 2026-05-26
**Context:** Retiring the list-in-cell parquet delivery path (Track B) from the PredictionFrame evaluation loop for HydraNet models. The forecasting path already uses composed savers and does not depend on Track B.

---

## Background

When a model declares `prediction_format: "prediction_frame"`, the evaluation orchestration in `model.py:1280-1345` writes three outputs per origin per target:

| Track | Format | Purpose | Consumers |
|-------|--------|---------|-----------|
| A (staging) | `.npy` + `.npz` | Metrics mmap reload | EvaluationStage (internal, ephemeral) |
| A+ (permanent) | `.npy` + `.npz` | PF ensemble consumption | PredictionFrameEnsembleManager |
| B (delivery) | list-in-cell `.parquet` | Legacy DF consumers | See "Who reads eval-path Track B" below |

The forecasting path (`model.py:1477-1498` → `ForecastingStage._save_via_savers()`) already bypasses this entirely and uses composed `PredictionSaver` implementations (`LocalParquetSaver`, `AppwriteSaver`, `ViewsForecastsSaver`). That path is clean, protocol-based, and correct.

---

## What Needs Investigation

### 1. Who reads eval-path Track B parquets for PF models?

The eval-path parquets follow the naming convention:
```
predictions_{run_type}_{YYYYMMDD_HHMMSS}_{target}_{seq:02d}.parquet
```

**Known consumers:**

| Consumer | File | Lines | What it does | Impact of removal |
|----------|------|-------|-------------|-------------------|
| `EvaluationReportTemplate._graph_step_sample_predictions()` | `templates/reports/evaluation.py` | 310-360 | Reads 3 sequenced parquets to render sample prediction line graphs in HTML eval report | Silently skips with warning — graphs absent from report |
| `ReportingStage.generate_forecast_report()` | `reporting/stage.py` | 74-78 | Reads latest prediction parquet for forecast HTML report | **NOT affected** — this reads FORECAST parquets (written by `LocalParquetSaver` in the forecasting path), not eval parquets |
| `PredictionIOManager._upload_to_prediction_store()` | `prediction/io.py` | 115-121 | Would upload to prediction store | **Already broken** — raises `NotImplementedError` for Arrow Tables. Never worked for PF eval path. |
| `DataFrameEnsembleManager._load_or_generate_prediction()` | `ensemble/dataframe_ensemble.py` | 719-799 | Loads parquets for DF ensemble aggregation | **NOT affected** — no DF ensemble includes HydraNet models (verified: only `golden_hour` includes them, and it's a PF ensemble) |

**Investigation question:** Are there any other consumers? Grep for:
```bash
grep -rn "_get_generated_predictions_data_file_paths\|read_parquet.*predictions_" views_pipeline_core/ tests/
```

### 2. The `_origin_sink` closure (model.py:1300-1345)

This is the 45-line nested closure that does all three Track writes inline. It uses `nonlocal`, captures `converter`, `staging_path`, `_run_type`, `_ts`, `all_targets`, `n_sequences` from the enclosing scope.

**Current structure:**
```python
def _origin_sink(origin_idx, pf_dict):
    for target in list(pf_dict.keys()):
        pf = pf_dict.pop(target)
        pf.save(staging_path / ...)          # Track A
        pf.save(data_generated / ...)        # Track A+
        if not skip_predictions_delivery:    # Track B
            table = converter.to_arrow_table(pf, ...)
            self._save_predictions(table, ...)
            del table
        del pf
        gc.collect()
```

**Questions to explore:**
- With `skip_predictions_delivery: True`, the `converter` import and instantiation (line 1288-1291) become dead code. Should we guard them?
- Should Track B removal simplify this to just two `pf.save()` calls? If so, do we still need `PredictionFrameConverter` imported in this scope at all?
- The `_save_predictions` call (line 1335-1339) delegates to `PredictionIOManager` which handles file naming. If we remove Track B, `PredictionIOManager` is no longer involved in the eval path for PF models. Is that correct? What else uses `PredictionIOManager` in the eval path?

### 3. The eval report sample graphs fallback

`EvaluationReportTemplate._graph_step_sample_predictions()` at `templates/reports/evaluation.py:310-360`:

- Currently reads sequenced parquets via `_get_generated_predictions_data_file_paths()`
- Could instead read from Track A+ numpy dirs via `_get_generated_pf_prediction_paths()`
- The data is the same — PredictionFrame samples — just in numpy format

**Questions to explore:**
- What does the graph actually plot? (Sample predictions vs actuals for a few origins.) Can it work from PredictionFrame directly?
- `_get_generated_pf_prediction_paths()` returns directories sorted newest-first. Each directory contains `origin_{i}/{target}/y_pred.npy`. The graph would need to load a few PFs, extract samples, and plot. Is this worth implementing, or are these graphs low-value enough to drop?
- If implementing: does the graph need the full DataFrame schema (month_id, priogrid_id columns) or just the prediction values? PredictionFrame has `identifiers['time']` and `identifiers['unit']` — that may be sufficient.

### 4. The `skip_predictions_delivery` config flag

Currently a per-model hyperparameter. This is the immediate lever.

**Questions to explore:**
- Should this become the default for PF models? i.e., if `prediction_format == "prediction_frame"`, should Track B default to off?
- If so, the guard at model.py:1331 would become:
  ```python
  if not self.configs.get("skip_predictions_delivery", self._prediction_format == "prediction_frame"):
  ```
- Or should we leave it explicit? The advantage of explicit is that if someone creates a PF model that participates in a DF ensemble (unusual but possible), they can opt back in.

---

## Suggested Implementation Steps

1. **Investigate consumer completeness** — Verify no hidden consumers of eval-path Track B parquets exist beyond what's listed above. Check tests, scripts, notebooks.

2. **Decide on eval sample graphs** — Either accept their loss or implement a numpy-backed fallback in `_graph_step_sample_predictions()`. This is a ~30 line change if desired.

3. **Set the flag in views-models** — `skip_predictions_delivery: True` for all HydraNet models. (See companion plan for views-models.)

4. **Optionally: default Track B off for PF models** — Modify the guard in `model.py:1331` so the default matches the prediction format. This removes the need for explicit config in every PF model.

5. **Optionally: simplify `_origin_sink`** — If Track B is permanently off for PF models, remove the converter import/instantiation and the conditional block. Reduces the closure from 45 lines to ~20.

6. **Do NOT refactor** `_origin_sink` into composed savers for the eval path. The eval path has different concerns than forecasting (streaming origins, staging for mmap, cleanup). Making it match the forecasting architecture would require rethinking how EvaluationStage interacts with the orchestration loop. That's a large blast radius for no user-facing benefit.

---

## Files in Scope

| File | What might change | Risk |
|------|-------------------|------|
| `views_pipeline_core/managers/model/model.py:1280-1345` | Simplify `_origin_sink` (optional), change default (optional) | Medium — shared by all PF models |
| `views_pipeline_core/templates/reports/evaluation.py:310-360` | Numpy fallback for sample graphs (optional) | Low — isolated template method |
| `views_pipeline_core/managers/prediction/io.py` | Nothing changes | N/A |
| `views_pipeline_core/managers/prediction/savers.py` | Nothing changes | N/A |
| `views_pipeline_core/managers/forecasting/stage.py` | Nothing changes | N/A |
| `views_pipeline_core/managers/reporting/stage.py` | Nothing changes (reads forecast parquets, not eval parquets) | N/A |
| `views_pipeline_core/data/model_path.py` | Nothing changes | N/A |

---

## What NOT to Touch

- `PredictionIOManager` — works, mixed concerns but contained, not growing
- `ForecastingStage` + saver protocol — already clean target architecture
- `EnsembleManager` / `DataFrameEnsembleManager` — not in HydraNet's data path
- `PredictionFrameEnsembleManager` — already correct, reads Track A+
- `_get_generated_predictions_data_file_paths()` — still needed for DF models and forecast reports

---

## Open Questions

1. Is the `EvaluationReportTemplate` sample graph valuable enough to implement a numpy fallback? Or is it a feature nobody looks at?
2. Should `skip_predictions_delivery` default to `True` for PF models at the framework level, or remain an explicit opt-in per model?
3. Are there any future PF models that might participate in a DF ensemble? If yes, they'd need Track B. If no (and the trajectory is all-PF-all-the-time for new models), defaulting to off is safe.
