# ADR-053: Eval-Path Track B Retirement

**Status:** Accepted
**Date:** 2026-05-27
**Deciders:** Simon, VIEWS platform team

---

## Context

The pipeline writes PredictionFrame outputs through multiple persistence tracks
during model evaluation. Two of these are informally called "Track B" but are
completely independent mechanisms that evolved separately:

| Aspect | Eval-path Track B (this ADR) | Forecast-path Track B (ADR-048) |
|--------|------------------------------|--------------------------------|
| Location | `model.py:1331-1343` (inline `_origin_sink` closure) | `ForecastingStage._save_via_savers()` |
| Protocol | Ad-hoc inline code | `PredictionSaver` protocol (ADR-048) |
| Control | `skip_predictions_delivery` config key | Saver list composition |
| Format | List-in-cell `.parquet` via `to_arrow_table` | Columnar `.parquet` via `LocalParquetSaver` |

### Eval-path Track B mechanism

Inside `ForecastingModelManager._execute_model_evaluation()`, the `_origin_sink`
closure writes each origin's PredictionFrame outputs to three tracks:

- **Track A (staging):** Compact `.npy` / `.npz` to `_pf_staging/origin_{i}/{target}/`
  for metrics mmap reload. Ephemeral.
- **Track A+ (permanent):** Compact `.npy` / `.npz` to
  `predictions_{run_type}_{ts}/origin_{i}/{target}/` for PF ensemble consumption.
- **Track B (delivery):** List-in-cell `.parquet` via
  `PredictionFrameConverter.to_arrow_table(pf)` then
  `PredictionIOManager.save_predictions()`. Controlled by the
  `skip_predictions_delivery` config key.

Track B predates the composed `PredictionSaver` protocol (ADR-048) and is unrelated
to the forecasting-path Track B (`LocalParquetSaver`). ADR-048 does not mention
this mechanism.

### Problems motivating retirement

1. **No active consumer.** No DataFrame ensemble includes any PredictionFrame model.
   The only consumer is `EvaluationReportTemplate._add_prediction_sample_graphs()`,
   which reads Track B parquets to render sample prediction line graphs. At PGM scale
   (172k grid cells), this produces 172k Plotly traces and a multi-GB HTML file that
   no browser can render (C-105). The method was designed for CM scale (~50 entities).

2. **Memory cost.** `PredictionFrameConverter.to_arrow_table()` internally calls
   `to_prediction_df()`, which causes a 33x memory explosion (measured: 4,766 MB peak
   for a 179 MB PredictionFrame). At 64 samples PGM scale, this alone exceeds
   workstation RAM (C-40).

3. **Silent default.** Prior to this decision, `skip_predictions_delivery` defaulted
   to `False` via `self.configs.get("skip_predictions_delivery", False)`. Models that
   omitted the key silently produced Track B parquets — paying the memory cost with no
   consumer reading the output.

---

## Decision

Make `skip_predictions_delivery` a **mandatory** config key for all PredictionFrame
models. The key has no default value — omitting it raises `KeyError` at config
validation time via `CoreConfigSniffer._check_skip_predictions_delivery()`.

- **`True`** (the common case): Skip eval-path Track B. No list-in-cell parquet
  conversion, no `to_arrow_table()` call, no memory spike.
- **`False`**: Produce eval-path Track B parquets as before.

### What is NOT affected

- **Track A (staging numpy)** and **Track A+ (permanent numpy)** always write
  regardless of this key. They are the primary persistence tracks for PF evaluation.
- **Forecast-path Track B** (`LocalParquetSaver` via `ForecastingStage`) is
  controlled by saver list composition (ADR-048), not by this config key.
- **DataFrame models** are unaffected. The `skip_predictions_delivery` check only
  fires when `prediction_format = "prediction_frame"`.

---

## Consequences

### Positive

- Eliminates the 33x memory explosion from `to_arrow_table()` for PF models that
  set `True`, which is all current production models.
- Mandatory key with no default enforces explicit intent declaration, consistent with
  ADR-008 (Observability and Explicit Failure) and the codebase's Fail Loud
  philosophy.
- Clear separation between the two Track B mechanisms prevents future confusion when
  searching documentation for "Track B."

### Negative

- 19 PredictionFrame models in views-models must add the key: 12 baseline models
  that previously omitted it, plus 7 HydraNet models that previously set `False`.
  A companion PR spec exists at
  `reports/views_models_mandatory_skip_predictions_delivery_spec.md`.
- Eval report sample graphs (C-105) lose their data source for PF models that set
  `True`. A future PR will address this with a numpy-backed fallback using Track A+
  data and entity sampling.
- PGMDataset scaling (C-106) remains a separate deferred concern — the numpy
  fallback bypasses PGMDataset entirely.

---

## Rationale

The decision rests on three observations:

1. **No active consumer at PGM scale.** No DF ensemble includes any PF model. The
   eval report sample graphs are the only consumer, and they are broken at PGM scale
   (C-105). Retiring Track B for PF models removes cost with no functional loss.

2. **Unacceptable memory cost.** The `to_prediction_df()` intermediate creates one
   Python float object per prediction value. At 64 samples PGM scale, this produces
   ~5.5 million float objects per target per origin, peaking at 4.8-6.4 GB with
   2.3 GB of permanent heap fragmentation (C-40). This is the single largest memory
   spike in the evaluation path.

3. **Fail Loud over silent defaults.** The previous `get("skip_predictions_delivery",
   False)` pattern silently produced expensive, unread parquets. Making the key
   mandatory forces each model to declare intent. If a model genuinely needs
   eval-path parquets, it sets `False` explicitly — the cost is acknowledged, not
   hidden.

---

## References

- **ADR-008:** Observability and Explicit Failure — mandatory key, no silent defaults
- **ADR-009:** Boundary Contracts and Configuration Validation — CoreConfigSniffer
  enforcement
- **ADR-041:** The Sniffer Pattern — `_check_skip_predictions_delivery()` method
- **ADR-042:** PredictionFrame Adoption — PF model config requirements
- **ADR-048:** Prediction Saver Protocol — forecasting-path Track B (separate
  mechanism, documented there)
- **CIC:** CoreConfigSniffer §3 — `skip_predictions_delivery` guarantee
- **C-40:** Memory scaling concern (motivating factor)
- **C-105:** Eval report sample graphs scale-blindness (consequence)
- **C-106:** PGMDataset scale guard (deferred)
- **C-109:** This ADR resolves C-109 (no ADR documented the decision)
- **C-110:** views-models companion spec:
  `reports/views_models_mandatory_skip_predictions_delivery_spec.md`
