# PR Spec: Add mandatory `skip_predictions_delivery` to all PF models in views-models

## Context

`views-pipeline-core` PR #87 (branch `fix/mandatory-skip-predictions-delivery`, merged into
`development`) makes `skip_predictions_delivery` a **mandatory config key** for any model
whose `prediction_format` is `"prediction_frame"`. The key must be a `bool`. If it is
missing or non-bool, `CoreConfigSniffer._check_skip_predictions_delivery()` raises
immediately at config validation time, before any inference runs.

This PR in **views-models** adds the key to every PF model that lacks it, and flips
existing `False` values to `True` (since Track B parquet delivery is suspended for PF
models — no consumer reads them).

**Confirmed failure:** `lucid_dream` was run without the key and crashed exactly as
expected:

```
KeyError: "CoreConfigSniffer: 'skip_predictions_delivery' is required when
prediction_format='prediction_frame'. Set it to True (skip eval-path parquets)
or False (produce them) in config_hyperparameters.py."
```

---

## Repo

- **Repository:** `views-platform/views-models`
- **Path:** `/home/simon/Documents/scripts/views_platform/views-models/`
- **Base branch:** whatever the current working branch is (likely `main` or `development`)
- **Feature branch name:** `fix/mandatory-skip-predictions-delivery`

---

## What determines a PF model

A model is a PredictionFrame model if its `config_meta.py` contains:

```python
"prediction_format": "prediction_frame",
```

The `prediction_format` key lives in **config_meta.py** (NOT config_hyperparameters.py).
The `skip_predictions_delivery` key lives in **config_hyperparameters.py**.

Both end up in the merged config dict that `CoreConfigSniffer` validates.

---

## Models to modify

### Group A: 12 models MISSING `skip_predictions_delivery` entirely

These will crash immediately on any run type. Add `"skip_predictions_delivery": True` to
each model's `config_hyperparameters.py`.

| # | Model | Manager Class | Config File |
|---|-------|---------------|-------------|
| 1 | `black_ranger` | `BaselineForecastingModelManager` | `models/black_ranger/configs/config_hyperparameters.py` |
| 2 | `blue_ranger` | `BaselineForecastingModelManager` | `models/blue_ranger/configs/config_hyperparameters.py` |
| 3 | `green_ranger` | `BaselineForecastingModelManager` | `models/green_ranger/configs/config_hyperparameters.py` |
| 4 | `heavy_strider` | `BaselineForecastingModelManager` | `models/heavy_strider/configs/config_hyperparameters.py` |
| 5 | `light_strider` | `BaselineForecastingModelManager` | `models/light_strider/configs/config_hyperparameters.py` |
| 6 | `lucid_dream` | `BaselineForecastingModelManager` | `models/lucid_dream/configs/config_hyperparameters.py` |
| 7 | `pink_ranger` | `BaselineForecastingModelManager` | `models/pink_ranger/configs/config_hyperparameters.py` |
| 8 | `red_ranger` | `BaselineForecastingModelManager` | `models/red_ranger/configs/config_hyperparameters.py` |
| 9 | `vivid_dream` | `BaselineForecastingModelManager` | `models/vivid_dream/configs/config_hyperparameters.py` |
| 10 | `waking_dream` | `BaselineForecastingModelManager` | `models/waking_dream/configs/config_hyperparameters.py` |
| 11 | `white_ranger` | `BaselineForecastingModelManager` | `models/white_ranger/configs/config_hyperparameters.py` |
| 12 | `yellow_ranger` | `BaselineForecastingModelManager` | `models/yellow_ranger/configs/config_hyperparameters.py` |

### Group B: 7 models with `skip_predictions_delivery: False` — flip to `True`

These currently produce Track B parquets that no consumer reads. Flip `False` to `True`
and remove the stale `#True,` comment.

| # | Model | Manager Class | Config File | Current Line |
|---|-------|---------------|-------------|-------------|
| 1 | `blazing_meteor` | `HydranetManager` | `models/blazing_meteor/configs/config_hyperparameters.py` | ~line 113 |
| 2 | `blue_stranger` | `HydranetManager` | `models/blue_stranger/configs/config_hyperparameters.py` | ~line 121 |
| 3 | `bold_comet` | `HydranetManager` | `models/bold_comet/configs/config_hyperparameters.py` | ~line 114 |
| 4 | `bright_starship` | `HydranetManager` | `models/bright_starship/configs/config_hyperparameters.py` | ~line 114 |
| 5 | `heavy_freighter` | `HydranetManager` | `models/heavy_freighter/configs/config_hyperparameters.py` | ~line 114 |
| 6 | `purple_alien` | `HydranetManager` | `models/purple_alien/configs/config_hyperparameters.py` | ~line 114 |
| 7 | `violet_visitor` | `HydranetManager` | `models/violet_visitor/configs/config_hyperparameters.py` | ~line 119 |

---

## Exact changes

### Group A: Add key to baseline models

Each of the 12 baseline models has a small `hyperparameters` dict in
`config_hyperparameters.py`. The dict is returned by `get_hp_config()`.

**Pattern for ranger models** (black, blue, green, pink, red, yellow):

Current config looks like:
```python
hyperparameters = {
    'steps': [*range(1, 36 + 1, 1)],
    'time_steps': 36,
    'window_months': 18,
    'lambda_mix': 0.05,
    'n_samples': 256,
}
```

Change to:
```python
hyperparameters = {
    'steps': [*range(1, 36 + 1, 1)],
    'time_steps': 36,
    'window_months': 18,
    'lambda_mix': 0.05,
    'n_samples': 256,
    'skip_predictions_delivery': True,
}
```

**Pattern for strider models** (heavy_strider, light_strider) and **white_ranger**:

Current config looks like:
```python
hyperparameters = {
    "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "steps": list(range(1, 37)),
    "time_steps": 36,
    "window_months": 36,
    "n_samples": 64,
    "seed": 42,
}
```

Change to:
```python
hyperparameters = {
    "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "steps": list(range(1, 37)),
    "time_steps": 36,
    "window_months": 36,
    "n_samples": 64,
    "seed": 42,
    "skip_predictions_delivery": True,
}
```

**Pattern for dream models** (lucid_dream, vivid_dream, waking_dream):

Current config looks like:
```python
hyperparameters = {
    'steps': [*range(1, 36 + 1, 1)],
    'time_steps': 36,
    'window_months': 18,
    'n_samples': 64,
}
```

Change to:
```python
hyperparameters = {
    'steps': [*range(1, 36 + 1, 1)],
    'time_steps': 36,
    'window_months': 18,
    'n_samples': 64,
    'skip_predictions_delivery': True,
}
```

**IMPORTANT NOTES for Group A:**
- Match the quoting style of the existing dict (some use single quotes, some double quotes)
- Add the key as the **last entry** in the dict, before the closing `}`
- Value is `True` (boolean, not string)
- No comment needed — `True` is the intended production value

### Group B: Flip existing False to True in HydraNet models

Each of the 7 HydraNet models has a line like:

```python
'skip_predictions_delivery':  False, #True,
```

Change each to:

```python
'skip_predictions_delivery': True,
```

That means:
1. Change `False` to `True`
2. Remove the `#True,` comment (it's stale — `True` IS the value now)
3. Normalize the double-space before `False` to single-space (cosmetic)

---

## Files modified (total: 19)

```
models/black_ranger/configs/config_hyperparameters.py     # ADD key
models/blue_ranger/configs/config_hyperparameters.py      # ADD key
models/green_ranger/configs/config_hyperparameters.py     # ADD key
models/heavy_strider/configs/config_hyperparameters.py    # ADD key
models/light_strider/configs/config_hyperparameters.py    # ADD key
models/lucid_dream/configs/config_hyperparameters.py      # ADD key
models/pink_ranger/configs/config_hyperparameters.py      # ADD key
models/red_ranger/configs/config_hyperparameters.py       # ADD key
models/vivid_dream/configs/config_hyperparameters.py      # ADD key
models/waking_dream/configs/config_hyperparameters.py     # ADD key
models/white_ranger/configs/config_hyperparameters.py     # ADD key
models/yellow_ranger/configs/config_hyperparameters.py    # ADD key
models/blazing_meteor/configs/config_hyperparameters.py   # FLIP False→True
models/blue_stranger/configs/config_hyperparameters.py    # FLIP False→True
models/bold_comet/configs/config_hyperparameters.py       # FLIP False→True
models/bright_starship/configs/config_hyperparameters.py  # FLIP False→True
models/heavy_freighter/configs/config_hyperparameters.py  # FLIP False→True
models/purple_alien/configs/config_hyperparameters.py     # FLIP False→True
models/violet_visitor/configs/config_hyperparameters.py   # FLIP False→True
```

## Files NOT modified

- No `config_meta.py` files (prediction_format stays as-is)
- No `config_deployment.py` files
- No `main.py` files
- No manager code
- No test files
- Nothing in views-pipeline-core or views-hydranet or views-baseline

---

## Why True and not False

`skip_predictions_delivery: True` means "do NOT produce Track B list-in-cell parquets
during evaluation." This is the correct value because:

1. **No consumer exists.** PF ensemble models read Track A+ numpy files
   (`predictions_{run_type}_{ts}/origin_i/target/`), not Track B parquets.
2. **Track B at PGM scale is broken.** `to_prediction_df()` creates 5.5M Python float
   objects per target per origin (~4.8-6.4 GB peak + 2.3 GB permanent fragmentation).
3. **Track A and A+ are unaffected.** Setting this to `True` only suppresses the
   parquet conversion step. The numpy staging files (Track A) and permanent numpy files
   (Track A+) are always written regardless of this flag.
4. **Forecasting is unaffected.** This flag only controls behavior during evaluation runs.
   Forecasting uses composed savers through `ForecastingStage` and never touches this flag.

If a model needs Track B parquets in the future (e.g., after a PyArrow-native fix),
set the value to `False` explicitly.

---

## Verification

After making all changes, verify by running one model from each group:

```bash
# Group A baseline model — was crashing, should now pass config validation
cd models/lucid_dream && python main.py -r calibration -t

# Group B HydraNet model — was producing unnecessary parquets, should now skip Track B
cd models/bright_starship && python main.py -r calibration -t
```

Both should pass `CoreConfigSniffer` validation without errors. Look for the log line:
```
CoreConfigSniffer: Config audited (run_type='calibration').
```

To verify the key is being read correctly without running full inference, you can also
check that the sniffer passes by running any model with `--run_type forecasting` (which
skips evaluation entirely but still validates config):

```bash
cd models/black_ranger && python main.py -r forecasting -t
```

---

## Commit message

```
fix: add mandatory skip_predictions_delivery to all 19 PF model configs

12 baseline models gain skip_predictions_delivery: True (was missing entirely,
now required by CoreConfigSniffer). 7 HydraNet models flip from False to True
(Track B parquet delivery suspended — no consumer reads them at PGM scale).
```

---

## Risk assessment

**Risk: Zero.** This change only adds/modifies a single config key per model. The key
controls whether an optional parquet conversion step runs during evaluation. Setting it
to `True` skips that step. All inference, training, forecasting, and Track A/A+ numpy
output is completely unaffected.

**Rollback:** If any model needs Track B parquets, set its `skip_predictions_delivery`
back to `False`. The conversion code is still present in `model.py` behind the flag.
