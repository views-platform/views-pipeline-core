# Track B Retirement: views-hydranet

**Date:** 2026-05-26
**Context:** The views-hydranet repo contains the HydraNet engine: model architecture, training, inference orchestration, and the `HydranetManager` class. This document assesses whether any work is needed in this repo for Track B retirement.

---

## Background

`HydranetManager` (at `views_hydranet/manager/hydranet_manager.py`) inherits from `ForecastingModelManager` and overrides three abstract methods:

| Method | Returns | Parquet awareness |
|--------|---------|-------------------|
| `_evaluate_model_artifact()` | `Dict[str, List[PredictionFrame]]` | **None** — returns numpy objects |
| `_evaluate_model_artifact_streaming()` | `None` (calls `origin_sink`) | **None** — calls sink with PF dicts |
| `_forecast_model_artifact()` | `Dict[str, PredictionFrame]` | **None** — returns numpy objects |

The key insight: **HydranetManager has zero parquet coupling.** It produces PredictionFrame objects and hands them to the facade. All parquet conversion happens in `views-pipeline-core` (the `_origin_sink` closure and `ForecastingStage` savers).

---

## What Needs Investigation

### 1. Does HydranetManager have any residual DataFrame output paths?

**Known:** The current implementation returns `Dict[str, PredictionFrame]` from all three methods.

**To verify:**
- Is there any legacy code path in `hydranet_manager.py` that produces DataFrames?
- Is there a `to_prediction_df()` or `to_dataframe()` call anywhere in the inference/evaluation path?
- Check `InferenceOrchestrator.generate_prediction_frames()` — does it have a DataFrame fallback?

```bash
grep -rn "DataFrame\|to_prediction_df\|to_dataframe\|parquet" views_hydranet/
```

### 2. The `set_dataframe_format` call

`HydranetManager.__init__()` line 148:
```python
self.set_dataframe_format(format=".parquet")
```

This calls a method on the parent class (`ForecastingModelManager`). Questions:
- What does `set_dataframe_format()` do? (It likely sets `PipelineConfig.dataframe_format` — the extension used by `PredictionFileNamer` and `save_dataframe()`.)
- Is this still necessary if Track B is retired for HydraNet? The forecasting savers use their own naming logic. The eval path won't write parquets anymore.
- Is this a no-op for PF models, or does it affect something else (like raw data caching format)?

### 3. The `prepare_actuals_df()` method

`HydranetManager.prepare_actuals_df()` (line 155-162) applies the "Instructional Blueprint" to the ground-truth DataFrame. This is called during evaluation to prepare actuals for metric computation.

**Question:** Does this method's output format matter for Track B? No — it's consumed by `EvaluationStage.evaluate()` which computes metrics. It doesn't produce prediction outputs. **Not in splash zone.**

### 4. The `DataFetcher` and data caching path

`DataFetcher` (at `views_hydranet/utils/data_fetcher.py`) loads raw data from a parquet cache on disk. This is the **input** data path (features), not the **output** prediction path.

**Question:** Does `skip_predictions_delivery` affect data caching? No — data caching is controlled by `use_saved` / the raw data path. Completely independent. **Not in splash zone.**

### 5. Does `InferenceOrchestrator` produce anything other than PredictionFrames?

The orchestrator generates predictions and returns them as PredictionFrame objects. But does it also write anything to disk directly? Or does it only return values?

```bash
grep -rn "save\|write\|parquet\|to_disk" views_hydranet/utils/inference_orchestrator.py
```

If the orchestrator is purely computational (takes volumes, returns PFs), then it's completely outside the splash zone.

---

## Assessment: Is Any Work Needed Here?

**Almost certainly not.** The views-hydranet repo is already clean with respect to Track B:

1. `HydranetManager` returns PredictionFrame objects — no parquet awareness.
2. The streaming interface (`_evaluate_model_artifact_streaming` + `origin_sink`) is the facade's responsibility, not HydraNet's.
3. All parquet conversion happens downstream in views-pipeline-core.
4. `InferenceOrchestrator` appears to be purely computational.

The only item worth investigating is the `set_dataframe_format(".parquet")` call — it might be dead code for PF models, but removing it has no user-facing benefit and risks breaking something subtle in the parent class's initialization.

---

## Suggested Investigation Steps

1. **Grep for DataFrame/parquet references** in the inference and evaluation paths:
   ```bash
   grep -rn "DataFrame\|parquet\|to_prediction_df" views_hydranet/utils/inference_orchestrator.py views_hydranet/manager/
   ```

2. **Verify `InferenceOrchestrator` is pure** — confirm it doesn't write to disk, only returns PredictionFrame dicts.

3. **Check `set_dataframe_format` impact** — trace what this method does in `ForecastingModelManager`. If it only affects `PipelineConfig.dataframe_format` (used by file namers), it's harmless.

4. **Check for any WandB logging that references parquets** — does the HydraNet training or eval path log parquet file paths to WandB? If so, those log messages would become stale (referring to files that no longer exist).

---

## Files in Scope

| File | What might change | Risk |
|------|-------------------|------|
| `views_hydranet/manager/hydranet_manager.py` | Possibly remove `set_dataframe_format` call (optional, low value) | Low |
| `views_hydranet/utils/inference_orchestrator.py` | Nothing — verify only | N/A |
| `views_hydranet/utils/data_fetcher.py` | Nothing — input data, not output | N/A |

---

## What NOT to Touch

- Model architecture code (`architectures/`)
- Training code (`train/`)
- Volume handling (`utils/volume_handler.py`)
- Feature scaling (`utils/feature_scaler.py`)
- Data sniffing (`utils/data_sniffer.py`)
- Visual diagnostics (`utils/visual_diagnostics.py`)
- The streaming protocol (`generate_prediction_frames_streaming`) — it's the correct interface

---

## Conclusion

**views-hydranet likely needs zero changes for Track B retirement.** The repo is already fully PredictionFrame-native. The parquet conversion happens entirely in views-pipeline-core's facade layer. The investigation steps above are confirmatory — verifying that there are no hidden couplings — rather than preparatory for a change.

If you discover something unexpected (e.g., the orchestrator writes temp parquets, or there's a DataFrame conversion in a legacy path), document it and assess whether it's dead code (remove) or active code (plan carefully).

---

## Appendix: views-faoapi

The views-faoapi repo is **not in the splash zone at all**. It:
- Receives parquet files from Appwrite (uploaded by `AppwriteSaver` on the forecasting path)
- Never reads from the local filesystem of model runs
- Never interacts with Track A, A+, or B directly
- Only cares that parquets arrive in Appwrite with correct metadata

As long as the **forecasting** path continues to upload parquets via `AppwriteSaver` (which it does — this is the composed saver chain, independent of Track B), views-faoapi is unaffected.

If views-faoapi ever needs adaptation (e.g., to accept a different prediction format from Appwrite), that's a separate, future concern unrelated to Track B retirement.
