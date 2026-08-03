# Track B Retirement: views-models

**Date:** 2026-05-26
**Context:** The views-models repo contains the HydraNet model configurations and the ensemble configurations. This is where the `skip_predictions_delivery` flag lives and where ensemble consumption patterns are defined.

---

## Background

Five HydraNet models exist in `views-models/models/`:

| Model | Algorithm | Prediction Format | Ensemble Membership |
|-------|-----------|-------------------|---------------------|
| `purple_alien` | HydraNet | `prediction_frame` | `golden_hour` (PF ensemble) |
| `blue_stranger` | HydraNet | `prediction_frame` | `golden_hour` (PF ensemble) |
| `violet_visitor` | HydraNet | `prediction_frame` | `golden_hour` (PF ensemble) |
| `bright_starship` | HydraNet | `prediction_frame` | None currently |
| `heavy_freighter` | HydraNet | `prediction_frame` | None currently |

One PF ensemble consumes them:

| Ensemble | Manager | Aggregation | Models |
|----------|---------|-------------|--------|
| `golden_hour` | `PredictionFrameEnsembleManager` | `concat` | purple_alien, blue_stranger, violet_visitor |

One synthetic test ensemble also uses PF:

| Ensemble | Manager | Aggregation | Models |
|----------|---------|-------------|--------|
| `synthetic_chant` | `PredictionFrameEnsembleManager` | `concat` | lucid_dream, vivid_dream, waking_dream |

---

## What Needs Investigation

### 1. Current state of `skip_predictions_delivery` across HydraNet models

**Known:**
- `purple_alien/configs/config_hyperparameters.py:114` — `'skip_predictions_delivery': False`

**To verify:**
- Do `blue_stranger`, `violet_visitor`, `bright_starship`, `heavy_freighter` all have this flag? What's its current value?
- If the flag is absent, the default in pipeline-core is `False` (Track B fires).

```bash
grep -rn "skip_predictions_delivery" models/*/configs/
```

### 2. Do any HydraNet models participate in DataFrame ensembles?

**Known:** Only `golden_hour` references HydraNet models, and it's a PF ensemble.

**To verify:**
```bash
grep -rn "purple_alien\|blue_stranger\|violet_visitor\|bright_starship\|heavy_freighter" ensembles/*/configs/config_meta.py
```

If any DF ensemble (one using `EnsembleManager` or `DataFrameEnsembleManager` in its `main.py`) lists a HydraNet model, removing Track B parquets breaks that ensemble. This seems unlikely but must be confirmed.

### 3. What about `bright_starship` and `heavy_freighter`?

These two don't appear in any ensemble. Questions:
- Are they standalone models only? If so, who consumes their predictions?
- Do they upload to Appwrite / the prediction store? Check their `main.py` for `use_prediction_store` argument usage.
- If they upload: the forecasting saver chain handles that independently. Track B eval parquets are still unnecessary.

### 4. The `golden_hour` ensemble — does it work end-to-end?

The ensemble uses `PredictionFrameEnsembleManager` and reads Track A+ numpy. Questions to explore:
- Has `golden_hour` been tested with all three constituent models producing output?
- Do the Track A+ directories exist on disk for all three models with matching timestamps?
- The ensemble uses `concat` aggregation — does it successfully concatenate across 3 models × 3 targets × 13 origins?

```bash
ls models/purple_alien/data/generated/predictions_calibration_*/origin_0/
ls models/blue_stranger/data/generated/predictions_calibration_*/origin_0/
ls models/violet_visitor/data/generated/predictions_calibration_*/origin_0/
```

### 5. Are there any other ensembles we should migrate to PF?

Look at all ensembles:
```bash
ls ensembles/
```

For each one:
- What's in `main.py` — `EnsembleManager`, `DataFrameEnsembleManager`, or `PredictionFrameEnsembleManager`?
- Which constituent models use `prediction_format: "prediction_frame"`?
- If all constituent models of an ensemble are PF models, that ensemble should probably migrate to `PredictionFrameEnsembleManager`.

This is exploratory — the answer determines whether there are more Track B retirements to come beyond HydraNet.

---

## Suggested Implementation Steps

1. **Audit all HydraNet configs** — Verify the current value of `skip_predictions_delivery` in all 5 models.

2. **Set `skip_predictions_delivery: True`** in all 5 HydraNet model configs:
   - `models/purple_alien/configs/config_hyperparameters.py`
   - `models/blue_stranger/configs/config_hyperparameters.py`
   - `models/violet_visitor/configs/config_hyperparameters.py`
   - `models/bright_starship/configs/config_hyperparameters.py`
   - `models/heavy_freighter/configs/config_hyperparameters.py`

3. **Verify `golden_hour` ensemble** — Run calibration or validation with all 3 sub-models + ensemble to confirm end-to-end. The sub-models should produce Track A+ numpy only (no eval parquets). The ensemble should read those and produce its own aggregated output.

4. **Verify forecasting delivery** — Run a forecasting pass on one HydraNet model with `--prediction_store True` to confirm that `LocalParquetSaver` and `AppwriteSaver` still fire correctly on the forecasting path (which is separate from the eval path).

5. **Optionally: survey other ensembles** — Identify if any other ensembles should migrate to PF format. This is future work, not a blocker.

---

## Files in Scope

| File | What might change | Risk |
|------|-------------------|------|
| `models/purple_alien/configs/config_hyperparameters.py` | Set flag | Zero |
| `models/blue_stranger/configs/config_hyperparameters.py` | Set flag | Zero |
| `models/violet_visitor/configs/config_hyperparameters.py` | Set flag | Zero |
| `models/bright_starship/configs/config_hyperparameters.py` | Set flag | Zero |
| `models/heavy_freighter/configs/config_hyperparameters.py` | Set flag | Zero |
| `ensembles/golden_hour/` | Nothing changes (already PF) | N/A |
| `ensembles/synthetic_chant/` | Nothing changes (test ensemble) | N/A |

---

## What NOT to Touch

- Ensemble `main.py` files — already correct
- `config_meta.py` files — already declare `prediction_frame`
- Any DF-mode ensemble — they're on a different code path entirely
- Any non-HydraNet model — they stay on whatever format they currently declare

---

## Open Questions

1. Do `blue_stranger`, `violet_visitor`, and `heavy_freighter` have `skip_predictions_delivery` in their configs at all? If not, should it be added explicitly or should the pipeline-core default handle it?
2. Is `bright_starship` still using the Phase 1 workaround (`_ensure_data()`, `args.saved = True`)? If so, does Track B retirement interact with that workaround at all? (Probably not — the workaround is about data fetching, not prediction output.)
3. Has `golden_hour` ever been run end-to-end? Is there a test run we can reference, or does this need to be the first?
4. Are there plans to add more HydraNet models to `golden_hour` or create new PF ensembles? If so, the default-off behavior becomes more important.
