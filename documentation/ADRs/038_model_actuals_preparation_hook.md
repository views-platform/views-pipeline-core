# Model-Specific Actuals Preparation Hook

| ADR Info   | Details                                    |
|------------|--------------------------------------------|
| Subject    | Extensible actuals preparation in evaluation |
| ADR Number | 038                                        |
| Status     | Accepted                                   |
| Author     | Simon                                      |
| Date       | 22.02.2026                                 |

## Context

Until recently, all VIEWS models shared an implicit assumption: every target variable listed in `config["targets"]` exists as a column in the viewser database. The evaluation loop in `ForecastingModelManager._evaluate_prediction_dataframe` reflected this assumption by loading the raw actuals parquet directly and immediately slicing it by target names.

This assumption breaks as the model ecosystem diversifies. HydraNet (Symmetric Feature Lifecycle, defined in the HydraNet model repo) introduced *manufactured targets* — binary classification signals (e.g. `by_sb_best`, `by_ns_best`, `by_os_best`) derived from raw count columns (`ged_sb`, `ged_ns`, `ged_os`) via a threshold operation. These derived signals are manufactured in the model repo at training time; they have never been written to the shared database, and we explicitly do not want them to be.

The broader architectural direction is clear: transformations, feature engineering, and derivations belong in individual model repos, not in the shared database. The database stores canonical raw inputs only. As models grow more bespoke this gap will widen — more models will define targets that require on-the-fly computation from raw actuals before evaluation can proceed.

The immediate symptom was a `KeyError` crash during HydraNet evaluation:

```
KeyError: "['by_sb_best', 'by_ns_best', 'by_os_best'] not in index"
```

## Decision

Introduce a **Template Method hook** on `ModelManager`:

```python
def prepare_actuals_df(self, df: pd.DataFrame) -> pd.DataFrame:
    return df  # no-op by default
```

This method is called inside `_evaluate_prediction_dataframe` immediately after the raw actuals DataFrame is loaded from disk and before it is sliced by target column names:

```python
df_viewser = read_dataframe(df_path)
df_viewser = self.prepare_actuals_df(df_viewser)   # ← the hook
df_actual  = df_viewser[self.configs["targets"]]
```

Model managers that manufacture derived targets override the hook to add those columns before the slice occurs:

```python
# In HydraNetManager (views-hydranet repo)
def prepare_actuals_df(self, df: pd.DataFrame) -> pd.DataFrame:
    return DataFetcher.apply_blueprint(df, self.configs)
```

### Overview

The core library provides raw material (the viewser DataFrame). The model manager — the only entity that understands the model's data contract — is given a single, clearly-defined opportunity to augment that material before the core proceeds. The core remains completely agnostic about what augmentation, if any, takes place.

## Consequences

**Positive Effects:**
- **Zero regression risk.** The default is a no-op. Every existing model manager inherits it silently and is completely unaffected.
- **Clean separation of responsibilities.** The derivation logic for `by_sb_best` lives in the HydraNet repo alongside the training code that creates the same signal — the definition is co-located with its use.
- **Generic and scalable.** Any future model that requires lags, interaction terms, scaling, or any other on-the-fly transformation of actuals can override the same hook without touching the core.
- **Enforces the architectural direction.** Derived signals must not be pushed into the shared database. This hook makes that constraint practically enforceable.

**Negative Effects:**
- **Implicit contract.** Model developers building managers with derived targets must know to override `prepare_actuals_df`. If they forget, they will get a `KeyError` at evaluation time rather than a clear error at configuration time.
- **No enforcement of the return contract.** The core trusts that the override returns a DataFrame containing all columns in `config["targets"]`. A broken override will still produce a `KeyError`, though the traceback will now point into the override rather than the core.

## Rationale

The Template Method pattern is the correct tool here. The algorithm skeleton (load → prepare → slice → evaluate) is fixed in the core; the preparation step is variable and belongs to the subclass. This is a textbook application of the Open/Closed Principle: the core is open for extension via override but closed for modification.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| Write derived targets into the viewser database | Contradicts the architectural direction. The database stores canonical raw inputs; derivations are model-specific and must not pollute shared storage. |
| Pass a callable in config | Couples config dictionaries to Python function objects; hostile to serialisation, logging, and sweep runs. |
| Override `_evaluate_prediction_dataframe` entirely in HydraNet | Requires duplicating ~150 lines of core logic, creating a maintenance burden and divergence risk. |
| Separate slice logic for regression vs. classification targets | Adds complexity without solving the underlying problem; derived signals could exist for any task type. |

### Considerations

- The hook is called once per evaluation partition (e.g. once for `calibration`, once for `validation`). There is no meaningful performance concern.
- The hook receives the full raw actuals DataFrame. Overrides should treat it as read-only and return a new or extended DataFrame rather than mutating in place, though the core does not enforce this.
- Both branches of the `if not ensemble` block in `_evaluate_prediction_dataframe` set `df_viewser` before the hook is called; ensemble and non-ensemble evaluation are both covered by the single hook call.

## Additional Notes

The investigation that led to this decision is documented in full at:
`reports/investigations/2026-02-21_evaluation_handshake_hardening_plan.md`

The implementation lives in `views_pipeline_core/managers/model/model.py`. The hook definition is on `ModelManager`; the call site is in `ForecastingModelManager._evaluate_prediction_dataframe`.

The practical guide for model developers implementing this hook is in:
`views_pipeline_core/managers/model/README.md` — Section 2.4 (Extension Points: prepare_actuals_df).

## Feedback and Suggestions

Feedback welcomed, particularly on whether a runtime guard should be added to raise a descriptive error when the post-hook DataFrame is still missing expected target columns (i.e. fail fast with a clear message rather than allowing the subsequent `KeyError` from pandas).
