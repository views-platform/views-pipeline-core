# Sprint Plan: C-109 + C-110 — Track B Retirement Documentation Completeness

**Risk register entries:** C-109 (Tier 4), C-110 (Tier 4)
**Target branch:** `docs/track-b-retirement-adr`
**Base branch:** `development`
**Estimated effort:** 2–3 hours
**Priority score:** C-110 = 2.0 (Imminent), C-109 = 1.0 (Plausible)

---

## 1. Problem Statement

PR #87 (`fix/mandatory-skip-predictions-delivery`) merged into development on
2026-05-26. It flipped the effective default for `skip_predictions_delivery` from
`False` to mandatory-explicit, with the pipeline-core code now reading
`self.configs["skip_predictions_delivery"]` (hard `KeyError` if absent). This changes
observable system behavior: PF models that previously produced eval-path Track B
parquets will stop doing so unless they explicitly set the key to `False`.

Two documentation gaps remain:

### C-109: No ADR Documents the Decision

The decision rationale lives in `reports/track_b_retirement_views_pipeline_core.md`
(an investigation report) but not in `documentation/ADRs/`. ADR-048 (Prediction Saver
Protocol) documents Track B for the **forecasting path** (`LocalParquetSaver`) but
does not mention the separate eval-path Track B mechanism (the inline `_origin_sink` →
`to_arrow_table` → `_save_predictions` flow). A developer searching ADRs for Track B
history would find only the forecasting path reference.

The ADR directory currently contains 51 numbered ADRs (000–052, with gaps at 017, 019,
030, 032, 033). The next available number is **053**.

### C-110: 12 PF Models Silently Affected

The investigation report (`track_b_retirement_views_pipeline_core.md`) analyzed only
the 5 HydraNet models. 12 baseline PF models (`black_ranger`, `blue_ranger`,
`green_ranger`, `heavy_strider`, `light_strider`, `lucid_dream`, `pink_ranger`,
`red_ranger`, `vivid_dream`, `waking_dream`, `white_ranger`, `yellow_ranger`) omit
`skip_predictions_delivery` from their configs entirely. Under the old default they
produced eval-path parquets; after PR #87 they will crash with `KeyError` unless the
views-models companion PR adds the key.

The views-models PR spec (`reports/views_models_mandatory_skip_predictions_delivery_spec.md`)
already covers all 19 PF models (12 baseline + 7 HydraNet). C-110 is resolved once
that spec is executed. This sprint documents the gap and cross-references the spec.

---

## 2. Scope

This PR contains:
1. A new ADR (053) documenting the eval-path Track B retirement decision
2. Updates to ADR-048 cross-referencing ADR-053 for completeness
3. Risk register updates for C-109 and C-110

This PR does **not** contain:
- Any code changes
- Changes to views-models (that's a separate repo PR per the spec)
- Changes to the forecasting-path Track B (that's ADR-048 / `LocalParquetSaver`)

---

## 3. ADR-053 Content Plan

### Title
`053_eval_path_track_b_retirement` (to be created)

### Status
Accepted

### Context

The eval-path Track B mechanism is an inline code path inside the `_origin_sink`
closure in `ForecastingModelManager._execute_model_evaluation()` (`model.py:1331-1343`).
When `skip_predictions_delivery` is `False`, it:

1. Calls `PredictionFrameConverter.to_arrow_table(pf)` to convert the PredictionFrame
   to a list-in-cell Arrow table
2. Calls `self._save_predictions(table, ...)` which delegates to
   `PredictionIOManager.save_predictions()` to write a `.parquet` file

This predates the composed `PredictionSaver` protocol (ADR-048) and is unrelated to
the forecasting-path Track B (`LocalParquetSaver`). The two Track B mechanisms evolved
independently:

| Aspect | Eval-path Track B | Forecast-path Track B |
|--------|-------------------|----------------------|
| Location | `model.py:1331-1343` (inline closure) | `ForecastingStage._save_via_savers()` |
| Protocol | Ad-hoc inline code | `PredictionSaver` protocol (ADR-048) |
| Control | `skip_predictions_delivery` config key | Saver list composition |
| Format | list-in-cell `.parquet` via `to_arrow_table` | Columnar `.parquet` via `LocalParquetSaver` |

### Decision

Make `skip_predictions_delivery` a **mandatory** config key for all PredictionFrame
models (enforced by `CoreConfigSniffer._check_skip_predictions_delivery()`). The key
has no default — models must explicitly declare their intent.

For PF models that do not need eval-path parquets (the common case), set
`skip_predictions_delivery: True`. Track A (staging numpy) and Track A+ (permanent
numpy) are unaffected and always write.

### Rationale

1. **No active consumer:** No DF ensemble includes any PF model. The
   `EvaluationReportTemplate._add_prediction_sample_graphs()` method reads Track B
   parquets but produces 172k Plotly traces at PGM scale — never worked for HydraNet
   (C-105).

2. **Memory cost:** `PredictionFrameConverter.to_arrow_table()` triggers a 33x memory
   explosion (`to_prediction_df()` intermediate). At 64 samples pgm scale, this alone
   exceeds workstation RAM (C-40).

3. **Fail Loud principle:** Rather than silently defaulting to skip or not-skip, the
   mandatory key forces each model to declare intent. This aligns with ADR-008 and
   the codebase's "if it is not in the config, fail super loud" philosophy.

### Consequences

- 12 baseline PF models and 7 HydraNet models in views-models must add the key
  (see `reports/views_models_mandatory_skip_predictions_delivery_spec.md`)
- Eval-path parquets are no longer produced for models that set `True`
- Scale-aware eval report graphs (C-105) will be addressed in a future PR using
  numpy-backed data instead of Track B parquets
- PGMDataset scaling (C-106) deferred to a separate assessment

### ADR Cross-References

- ADR-008 (Observability and Explicit Failure) — mandatory key, no silent defaults
- ADR-009 (Boundary Contracts) — CoreConfigSniffer enforcement
- ADR-041 (Sniffer Pattern) — `_check_skip_predictions_delivery()` method
- ADR-042 (PredictionFrame Adoption) — PF model config requirements
- ADR-048 (Prediction Saver Protocol) — forecasting-path Track B (separate mechanism)
- CIC: CoreConfigSniffer §3 — guarantee documented

---

## 4. ADR-048 Update

Add a brief cross-reference in ADR-048's "Context" or "Related" section:

> **Note:** This ADR covers the forecasting-path Track B (`LocalParquetSaver` via
> composed savers in `ForecastingStage`). The eval-path Track B (inline list-in-cell
> parquet writes in `_origin_sink`) is a separate, older mechanism documented in
> ADR-053.

This prevents confusion when a developer searches for "Track B" and finds only
ADR-048.

---

## 5. Risk Register Updates

### C-109 → Resolved

The ADR documents the decision. Resolution text:

> ADR-053 documents the eval-path Track B
> retirement decision, distinguishes it from the forecasting-path Track B (ADR-048),
> and cross-references the views-models companion spec.

### C-110 → Resolved (conditional)

C-110 is resolved once the views-models companion PR lands. Two options:

- **If the views-models PR is merged before this PR:** Mark C-110 as Resolved with
  date and PR reference.
- **If still pending:** Mark C-110 as "Mitigated — views-models PR spec written
  (`reports/views_models_mandatory_skip_predictions_delivery_spec.md`), execution
  pending." Update to Resolved when the views-models PR merges.

### Header Count Update

Adjust the header counts based on how many entries move to Resolved.

---

## 6. Files Modified

| File | Change |
|------|--------|
| `documentation/ADRs/053_eval_path_track_b_retirement` | **New** — full ADR (to be created in this sprint) |
| `documentation/ADRs/048_prediction_saver_protocol.md` | Add cross-reference to ADR-053 |
| `reports/technical_risk_register.md` | Resolve C-109; update C-110 status; adjust header counts |

---

## 7. Acceptance Criteria

- [ ] ADR-053 exists with Status: Accepted
- [ ] ADR-053 distinguishes eval-path Track B from forecast-path Track B
- [ ] ADR-053 references the views-models spec, C-105, C-106, and related ADRs
- [ ] ADR-048 cross-references ADR-053
- [ ] C-109 marked Resolved in risk register
- [ ] C-110 status updated (Resolved or Mitigated-pending)
- [ ] Header counts accurate
- [ ] No code changes in this PR (documentation only)

---

## 8. Risk Assessment

**Blast radius:** Zero. Documentation-only PR. No code changes, no behavior changes,
no test changes. The ADR formalizes a decision already implemented in PR #87.

**Timing dependency:** C-110 resolution depends on the views-models companion PR. If
that PR hasn't been created yet, this sprint documents the gap and the spec; the
register entry stays open until the views-models PR merges. This is acceptable — the
documentation gap (C-109) is the primary deliverable.
