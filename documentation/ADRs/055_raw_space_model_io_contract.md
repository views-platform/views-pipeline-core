# ADR-055: Raw-Space Model I/O Contract

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Model boundary numerical scale contract |
| ADR Number          | 055               |
| Status              | Proposed          |
| Author              | Simon Polichinel von der Maase |
| Date                | 08.06.2026        |

**Consulted:** views-hydranet ADR-003 (Laws 5–6), ADR-046, ADR-028, evaluation_contract_v1; views-stepshifter ADR-003, target_transform_declarative story; views-r2darts2 ADR-012; pipeline-core ADR-003, ADR-042, CICs for ForecastingModelManager, DatasetTransformationModule, PredictionFrame

**Resolves:** GitHub issue #174. Fills the `⟨PENDING — views-pipeline-core ADR-XXX⟩` placeholder in views-stepshifter ADR-003.

---

## Context

The VIEWS forecasting pipeline has a structural boundary between **views-pipeline-core** (orchestration, persistence, evaluation) and **model repositories** (views-hydranet, views-stepshifter, views-r2darts2, views-baseline). Models implement abstract methods (`_forecast_model_artifact`, `_evaluate_model_artifact`) that return predictions consumed by the pipeline. The question this ADR answers: **what numerical scale must those predictions be in?**

### The convention exists but was never ratified

Multiple documents state that models return predictions in raw (untransformed) target space:

- `plans/2026-03-15_prediction_frame_two_track_status.md` Item 3: *"transformations are now the model repo's responsibility — models must return predictions in the original scale."*
- `plans/2026-06-01_pfe_production_roadmap.md` §7 / risk C-140.
- The `ForecastingModelManager` CIC governs return **type** and **step-window coverage** but says nothing about numerical **scale**.

The convention is real but lives only in planning documents. It has never been ratified as a binding ADR in this repo.

### Model repos have local governance of varying strength

Each model repo has approached the raw-I/O boundary differently:

**views-hydranet** has the strongest governance. Three documents converge on the same rule:
- ADR-046 (Symmetric Feature Lifecycle): *"These features MUST be inverted back to raw count space before evaluation."* Separates Value Transformations (in-place, must invert) from Feature Derivations (additive, no inversion).
- ADR-003, Law 5 (Explicit Transformation): *"Every transformation must be triggered by a specific config entry. If the config is silent, the data remains raw."* Law 6 (Prefix-Purity): the `lr_` prefix describes semantic intent, not numerical scale — the config is the authority.
- evaluation_contract_v1.md §2.2: *"Raw Scale Handoff: All predictions MUST be inverse-transformed back to Raw Count Scale before being placed in the DataFrame."*
- ADR-028 (Numerical Stability Guards) documents what happens when inversion goes wrong: `expm1` on an un-clamped log-space value of 100 produces 2.6 × 10⁴³, an immediate `Inf` in float32.

Hydranet is compliant today. Its `FeatureScaler.inverse_transform_volume()` applies `np.expm1()` before predictions leave the model.

**views-r2darts2** has ADR-012 (Scaling Pipeline and Calibration Integrity), which mandates correctness of the scaling machinery — Darts native `Pipeline`, `global_fit=True`, sample-dimension preservation. It governs the **how** of transforms but does not explicitly state a raw-I/O boundary contract. R2darts2 transforms internally via Darts Pipeline and its models return predictions in raw space, but this is a property of its implementation, not a documented contract.

**views-stepshifter** has just drafted ADR-003 (Raw Target Space I/O Contract), which is the most detailed local expression: 8 decision clauses, two enforcement guards (a characterization test and a load-time transform-name guard), a ShurfModel interim rule, and a `TRANSFORMS` registry pattern (per `target_transform_declarative.md`). It explicitly defers platform-wide authority to this ADR.

**views-baseline** has no documented transform governance.

### The motivating incident

Commit `5fcfe43` (2025-11-20) silently added `np.log1p` / `np.expm1` inside views-stepshifter's `StepshifterModel`, forcing log-space training with no config declaration. Commit `08ee2eb` (2026-04-11) reverted it, citing "the platform convention" — a convention that existed only as aspiration. During the 5-month window between those commits, trained artifacts were serialized in log-space.

On 2026-06-04, a calibration run for ensemble `big_chungus` (`lr_ged_sb`, cm) scored MSLE 2.519 — worse than an all-zeros baseline (2.147) and ~3× the fatalities002 retrospective gold standard (0.835). The four deep-learning (r2darts2) constituents, which transform internally via `asinh`, scored 0.43–0.57. The ~19 stepshifter-family constituents, loaded from artifacts potentially trained during the `5fcfe43` log-space window, were broken. **The proximate cause is still under investigation** (views-models #111) — poor transform hygiene (a scale mismatch between training-time and inference-time transform state) is the leading hypothesis but is not yet confirmed. Regardless of the specific cause, the **absence of a binding scale contract** is what permitted the ambiguity to exist in the first place.

### Pipeline-core removed its undo blocks

Pipeline-core previously contained `# TEMPORARY` transform-undo blocks in `model.py` that inverted transforms at save time. These were removed per the 2026-03-15 plan. Today, **neither** pipeline-core **nor** (post-revert) stepshifter inverts transforms. Responsibility was pushed to model repos but never ratified. This ADR ratifies that push.

### PredictionFrame does not encode scale — by design

`PredictionFrame` (ADR-042) is a column-name-free, scale-metadata-free NumPy transport — a dense `(N, S)` array with `time` / `unit` identifiers. It carries no `lr_` / `ln_` / `lx_` prefix and no scale metadata. This is a **deliberate design choice**, not a limitation — metadata could be added, but encoding scale information in ad hoc naming conventions is exactly the unsafe practice this ADR exists to eliminate (Clause 5). PredictionFrame's refusal to carry scale signals forces the correct architecture: scale is governed by the model's config declaration, not by the transport object.

---

## Decision

**Model repositories return predictions in raw target space. Target-space transforms are model-internal, config-declared, and inverted before output. The column-name prefix scheme is deprecated as a scale signal.**

### Clause 1 — Raw input, raw output

Models consume target data in raw target space and return predictions in raw target space. This applies to all abstract method return sites: `_forecast_model_artifact`, `_evaluate_model_artifact`, `_evaluate_sweep`.

"Raw target space" means the **physical quantity's natural scale** — actual counts, rates, or probabilities as they exist in the underlying data source, before any mathematical transformation. Queryset-level transforms (e.g., `ops.ln()` in viewser) that deliver pre-transformed targets to models are **not compliant** with this contract — they create the same ambiguity this ADR exists to eliminate, just at a different boundary. Models must not depend on receiving pre-transformed input from querysets; querysets must deliver targets in their natural scale.

### Clause 2 — Transforms are model-internal

Any target-space transform applied during training (log1p, asinh, standardization, etc.) is the model library's internal concern. It must be:
- (a) applied **inside** the model library,
- (b) **declared in configuration** (not hardcoded),
- (c) **inverted before predictions leave the model**.

A silent, undeclared transform — the `5fcfe43` pattern — is a contract violation.

### Clause 3 — Config-declared transform is the sole source of truth for scale

The model's configuration (or the artifact-serialized transform name, for loaded artifacts) is the only authority on what transform was applied and therefore what inverse is needed. Not a column-name prefix. Not a transform-history log. Not inference from value ranges.

This is ADR-003 (Authority of Declarations over Inference) applied to numerical scale. ADR-003 already forbids inferring `level` "from index column names." Inferring scale from a column-name prefix is the same forbidden pattern.

### Clause 4 — Ownership of inversion

The model library is the **sole owner** of target-space inversion. Pipeline-core (orchestration, persistence, evaluation) and views-evaluation do not undo model transforms. A model that emits non-raw predictions and relies on a downstream consumer to invert them is non-compliant.

This codifies the removal of pipeline-core's `# TEMPORARY` transform-undo blocks and makes their re-introduction a contract violation.

### Clause 5 — The `lr_`/`ln_`/`lx_` prefix scheme is deprecated as a scale signal

The column-name prefix scheme implemented by `DatasetTransformationModule` (`ln_`, `lx_`, `lr_` prefixes + in-memory `transformation_history`) is **not an authoritative signal for numerical scale and must not be treated as one**. The reasons:

1. **The boundary it guarded no longer exists.** Transformed data no longer travels outside a model library. There is no cross-boundary handoff for a prefix to annotate.
2. **A column name is intention, not enforced truth.** The `DatasetTransformationModule` CIC documents its own silent-corruption modes: an undo with the wrong offset *"completes without error but produces incorrect values"*; a duplicate transform is *"silently skipped."* A column labeled `ln_` is not evidence the values are in log space.
3. **PredictionFrame cannot carry it.** PredictionFrame (ADR-042) has no column names and no scale metadata. The prefix scheme is structurally incompatible with the pipeline's transport direction.
4. **The would-be source of truth is unreliable.** `DatasetTransformationModule.transformation_history` exists in memory only, is never persisted, and is unvalidated.

**What this means in practice:**
- No code — new or existing — should depend on a column-name prefix to determine numerical scale. A column named `ln_ged_sb` is **not evidence** that the values are in log-space.
- Existing code that reads prefixes for **identity** purposes (e.g., `pred_` output prefix, `lr_` as a target name component) is unaffected — the prefix remains part of the column's identity, just not a scale signal.
- The `DatasetTransformationModule` continues to function for its current consumers (forecast reporting in views-reporting). Its prefix-renaming behavior is not removed; the **treatment of those prefixes as authoritative scale evidence** is what this ADR prohibits.
- Model repos that have adopted config-declared transforms (hydranet, stepshifter) already do not depend on the prefix as a scale signal.

### Clause 6 — Scope: targets; queryset transforms are non-compliant

This contract governs the **target column** (the quantity being predicted). Queryset-level target transforms (e.g., `ops.ln()` in viewser delivering `ln_ged_sb` instead of raw `ged_sb`) are **non-compliant** with this contract — they shift the ambiguity upstream rather than eliminating it. Target data must arrive at the model in its natural scale; any compression the model needs must be applied internally (Clause 2).

Feature transforms at queryset level are a separate concern and are not governed by this ADR. A future `FeatureFrame` input-side contract (if needed) is a separate decision. **Note:** the transformation and feature engineering landscape is actively changing — a pivot away from viewser is under consideration, and new approaches to how transforms are applied and managed are forthcoming. This ADR establishes the target-side contract; the feature-side story will evolve alongside those changes.

### Clause 7 — Implementation is per-repo

This ADR mandates the **contract** (raw I/O, config-declared, model-owned inversion) but not the **mechanism**. Each model repo implements enforcement locally:

| Repository | Mechanism | Status |
|---|---|---|
| views-hydranet | `FeatureScaler` + `TRANSFORMS` registry + ADR-046/003/evaluation_contract | Compliant |
| views-stepshifter | `TRANSFORMS` registry + `target_transform` config key + `ReproducibilityGate` (per ADR-003 and `target_transform_declarative.md` story) | ADR ratified; mechanism in progress |
| views-r2darts2 | Darts native `Pipeline` + `global_fit=True` (ADR-012) | Compliant by implementation; recommend local ADR for explicitness |
| views-baseline | TBD | Must declare compliance or document why exempt |

Pipeline-core does not import or validate model-internal transform registries. The contract is enforced at the boundary by its effects: predictions must be in raw space when they reach the pipeline.

### Clause 8 — Existing `output_scale` infrastructure

Pipeline-core already has infrastructure that tracks output scale: an optional `output_scale` config key (`"log"` or `"natural"`, ADR-011 §Optional Config Keys) validated by `CoreConfigSniffer._check_output_scale()`, and an ensemble-level consistency check `validate_output_scale_consistency()` (C-158). This infrastructure predates this ADR.

This ADR does not replace or retire that infrastructure — it provides the governing contract that gives it meaning:

- **`output_scale: "log"` is a contract violation.** A model declaring `"log"` is stating that it does not undo its transforms internally. Under Clauses 1 and 2, this is non-compliant. The declaration exists to make non-compliance **visible** for migration tracking — it is not permission to remain non-compliant.
- **`output_scale: "natural"` is the compliant declaration.** It confirms the model returns raw-space predictions.
- **`validate_output_scale_consistency()`** catches mixed-scale ensembles at the ensemble boundary. Per-repo mechanisms (Clause 7) enforce individual model compliance. There is currently no pipeline-core check that a single model's predictions are in raw space.
- **End state:** once all models comply, `"log"` should be rejected by `_check_output_scale()` as an invalid value.

---

## Consequences

### Positive

- **Closes the governance gap.** The ForecastingModelManager CIC governs return type and format; this ADR adds return scale. The model boundary contract is now complete.
- **Ratifies the premise of the `08ee2eb` revert** and the removal of pipeline-core's undo blocks. Both were correct in direction but rested on unratified convention. This ADR supplies the authority they assumed.
- **Kills a class of silent bug.** An undeclared transform produces scale-mismatched output with no error signal. Under this contract, the absence of a config declaration means the transform was never applied — and any model that silently applies one is non-compliant.
- **Aligns with PredictionFrame migration.** PredictionFrame cannot encode scale. This contract makes that a non-issue: output is always raw, so there is nothing to encode.
- **Provides the platform authority** that views-stepshifter ADR-003 explicitly defers to (`⟨PENDING — views-pipeline-core ADR-XXX⟩` → ADR-055).

### Negative / Obligations

- **Every model repo must be verifiably compliant.** Hydranet and stepshifter (post-mechanism) are. R2darts2 and baseline need explicit confirmation or local documentation.
- **Frozen artifacts from non-compliant windows.** Artifacts trained during the `5fcfe43` window (2025-11 → 2026-04) in views-stepshifter may emit log-space predictions when loaded under current raw-output code. These must be audited and retrained. This is the open thread of the 2026-06-08 investigation (views-models #111).
- **Prefix deprecation requires downstream awareness.** Any external consumer that parses `ln_`/`lx_` prefixes to infer scale must be migrated. Within the platform, the primary consumer is views-reporting's `DatasetTransformationModule`, which is unaffected (it operates on its own undo logic, not cross-boundary inference).
- **Ensemble-level enforcement exists; single-model enforcement does not.** `validate_output_scale_consistency()` catches mixed-scale ensembles (Clause 8). Per-repo mechanisms enforce individual model compliance (Clause 7). There is no pipeline-core check that a single model's predictions are in raw space — this relies on per-repo discipline. A future scale-plausibility sniffer at the model→evaluation handoff (cf. hydranet ADR-028's clamping strategy) could close this gap but is not required by this ADR.

---

## Rationale

### Why not keep the prefix scheme as the scale contract?

A column name is intention, not truth. `DatasetTransformationModule`'s own CIC documents the failure modes: wrong-offset undo produces silently incorrect values; duplicate transforms are silently skipped. And PredictionFrame has no column names to carry the prefix. The scheme worked when transforms were applied centrally and the prefix rename happened atomically with the math; it cannot survive the decentralization of transforms to model repos.

### Why not keep inversion in pipeline-core?

Pipeline-core already removed its `# TEMPORARY` undo blocks. The 2026-03-15 plan designates inversion as the model repo's responsibility. Re-adding it centrally would re-couple every model's scale semantics to the core, require pipeline-core to know which transform each model used (violating separation of concerns), and re-open risk C-140.

### Why not mandate a specific implementation (shared registry, shared protocol)?

Model repos in views-platform must not import from each other (package-isolation rule). Hydranet uses `FeatureScaler` + a config-validated `TRANSFORMS` dict. Stepshifter is building a local `TRANSFORMS` registry + `ReproducibilityGate` integration. R2darts2 uses Darts native `Pipeline`. Each mechanism is adapted to its framework. Mandating one mechanism would force unnatural abstractions. The contract is what matters — raw in, raw out, config-declared — not the implementation.

### Why deprecate the prefix scheme rather than hard-remove it?

The prefixes remain useful as **identity** markers (the `lr_` in `lr_ged_sb` is part of the target name, not a scale annotation). Removing them entirely would be a gratuitous breaking change to column naming. What is deprecated is the **interpretation** of those prefixes as evidence of numerical scale. This is a semantic change, not a deletion.

---

## Documents Amended by This ADR

| Document | Amendment |
|---|---|
| `CICs/ForecastingModelManager.md` §3 | Add: *"Guarantees that predictions received from abstract methods are in raw target space (ADR-055). Pipeline-core does not apply inverse transforms."* |
| `CICs/DatasetTransformationModule.md` §1 | Add deprecation note: *"The prefix-renaming behavior (`ln_`/`lx_`/`lr_`) is deprecated as a cross-boundary scale signal (ADR-055). It remains functional for in-module undo operations. The `ln_`/`lx_`/`lr_` prefix names are legacy identity conventions retained for backward compatibility — they do not indicate the current numerical scale of the data."* |
| `CICs/CoreConfigSniffer.md` | Note that `_check_output_scale()` is positioned under ADR-055 Clause 8 as transitional enforcement. `output_scale: "log"` is a self-declaration of non-compliance; `"natural"` is compliant. |
| ADR-009 §Boundary Types in This Project | Add row: **Model output scale** boundary, enforced by `validate_output_scale_consistency()` (ensemble) + per-repo mechanisms (single model), validates raw target space (ADR-055). |
| ADR-011 §Optional Config Keys in `config_meta.py` | Add note to `output_scale` row: *"Positioned under ADR-055 Clause 8 as transitional enforcement. `"natural"` = compliant (model returns raw). `"log"` = transitional non-compliance (model does not undo transforms internally). Once all models comply with ADR-055, `"log"` becomes an invalid value."* |
| ADR-003 | No amendment needed. ADR-055 is a direct application of its principle; a cross-reference in the References section is sufficient. |
| ADR-042 | No amendment needed. PredictionFrame's lack of scale metadata is a design strength under this contract, not a gap. |

---

## Cross-Repo References

### views-hydranet (compliant — precedent, not subordinate)
- [ADR-003: Philosophy of Engineering](https://github.com/views-platform/views-hydranet/blob/main/docs/ADRs/active/003_philosophy_of_engineering_and_semantic_authority.md) — Law 5 (Explicit Transformation): *"Every transformation must be triggered by a specific config entry."* Law 6 (Prefix-Purity): *"Column prefixes describe Semantic Intent, not Numerical Scale. The config Fit/Transform state is the only authoritative record."*
- [ADR-046: Symmetric Feature Lifecycle](https://github.com/views-platform/views-hydranet/blob/main/docs/ADRs/active/046_symmetric_feature_lifecycle.md) — Ontological distinction between transformations (must invert) and derivations (no inversion).
- [evaluation_contract_v1.md §2.2](https://github.com/views-platform/views-hydranet/blob/main/docs/specs/evaluation_contract_v1.md) — *"Raw Scale Handoff: All predictions MUST be inverse-transformed back to Raw Count Scale."*
- [ADR-028: Numerical Stability Guards](https://github.com/views-platform/views-hydranet/blob/main/docs/ADRs/active/028_numerical_stability_guards.md) — Documents the `expm1` amplification trap when inversion is applied to un-clamped log-space values.

### views-stepshifter (defers to this ADR)
- [ADR-003: Raw Target Space I/O Contract](https://github.com/views-platform/views-stepshifter/blob/main/docs/ADRs/003_raw_target_space_io_contract.md) — 8 decision clauses, enforcement guards E1/E2, ShurfModel interim rule. Its `⟨PENDING — views-pipeline-core ADR-XXX⟩` placeholder is resolved by this ADR as ADR-055.
- [target_transform_declarative.md](https://github.com/views-platform/views-stepshifter/blob/main/docs/stories/target_transform_declarative.md) — The `TRANSFORMS` registry + `target_transform` config key mechanism.
- Commits: [`5fcfe43`](https://github.com/views-platform/views-stepshifter/commit/5fcfe4330701c8f581969ba847e49a17277828d5) (the silent transform); [`08ee2eb`](https://github.com/views-platform/views-stepshifter/commit/08ee2ebb673348cea11ed6afaefc99058129f191) (the revert).

### views-r2darts2 (compliant by implementation)
- [ADR-012: Scaling Pipeline and Calibration Integrity](https://github.com/views-platform/views-r2darts2/blob/main/docs/ADRs/012_scaling_pipeline_and_calibration_integrity.md) — Mandates Darts native `Pipeline`, `global_fit=True`, sample-dimension preservation. Governs transform correctness but does not explicitly state a raw-I/O boundary. Recommend a local ADR analogous to stepshifter ADR-003 for explicitness.

### views-pipeline-core (this repo)
- [ADR-003: Authority of Declarations over Inference](003_authority_of_declarations_over_inference.md) — The governing principle. Scale is declared in config, not inferred from column names.
- [ADR-042: PredictionFrame Adoption](042_prediction_frame_adoption.md) — PredictionFrame has no column names and no scale metadata. The structural forcing function.
- [CIC: ForecastingModelManager](../CICs/ForecastingModelManager.md) — Governs return type/format; this ADR adds return scale.
- [CIC: DatasetTransformationModule](../CICs/DatasetTransformationModule.md) — The `lr_`/`ln_`/`lx_` prefix + in-memory history machinery. Documents silent-corruption modes.
- [CIC: PredictionFrame](../CICs/PredictionFrame.md) — Column-name-free, scale-metadata-free transport.
- Plans: `plans/2026-03-15_prediction_frame_two_track_status.md` Item 3; `plans/2026-06-01_pfe_production_roadmap.md` §7 / risk C-140.
- Investigation: views-models #111 (frozen artifact audit); GitHub #174 (this ADR's story issue).

---

## Feedback and Suggestions

Open questions deferred from this ADR for separate decisions:

1. **Pipeline-core single-model boundary guard** (tracked as D-29 in the risk register). Should core add a fail-loud scale-plausibility check (e.g., "predictions for a fatality target should not all be < 1.0 when raw counts are expected") at the model→evaluation handoff? Hydranet ADR-028 demonstrates the pattern. This would be a new sniffer rule, not part of this contract. Ensemble-level enforcement already exists via `validate_output_scale_consistency()` (Clause 8).
2. **Pipeline-core characterization test** (tracked as C-175 in the risk register). A test asserting no transform operations (`expm1`, `log1p`, `inverse_transform`) in `model.py` or `ForecastingStage` would prevent regression. The existing `test_output_scale_validation.py` suite covers ensemble-level enforcement but not the single-model raw-I/O invariant.
3. **`output_scale: "log"` retirement timeline** (tracked as D-30 in the risk register). Once all model repos comply with this ADR, `"log"` should become an invalid value in `SUPPORTED_OUTPUT_SCALES`. No timeline is set — it depends on per-repo migration progress.
4. **FeatureFrame input-side contract.** No `FeatureFrame` exists yet. If one is created, does the symmetric contract (raw features in) need its own ADR, or is this one amended?
5. **Shared transform registry.** Each model repo maintains a local `TRANSFORMS` registry. If a `views-model-utils` shared library is created, the registries could be consolidated. Until then, local copies are correct per the package-isolation rule.
6. **views-baseline compliance.** Baseline models (all-zeros, LOCF, average) trivially satisfy raw-I/O (they apply no transform), but this should be documented explicitly.

---
*End of ADR-055.*
