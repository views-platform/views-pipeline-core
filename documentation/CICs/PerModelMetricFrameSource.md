# Class Intent Contract: PerModelMetricFrameSource

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-09-18
**Related ADRs:** ADR-018 (evaluation-of-record rendered from an injected source; views-reporting), ADR-044 (register C-202, C-328)

---

## 1. Purpose

Implements views-reporting's `EvaluationSource` port for the evaluation report by looking each
model's persisted `MetricFrame` up under **that model's own** `data/generated`. One
`MetricFrameFileSource` (theirs) per model, composed, each rooted where `EvaluationStage`
actually wrote. Exists because a single source rooted at the subject's directory found the
subject and no one else — every constituent and baseline row of every ensemble's evaluation
report was silently absent (#485).

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** define the on-disk layout. `root/<model>/<run_type>/metricframe_<target>` is
  views-reporting's `MetricFrameFileSource._frame_dir`, the locked cross-repo contract (C-202).
- Does **not** load frames itself; delegates to `MetricFrameFileSource.metric_frame`.
- Does **not** decide which models a report compares. The template asks by name.
- Does **not** copy, symlink or relocate frames.

---

## 3. Responsibilities and Guarantees

- `metric_frame(model)` returns that model's frame at the bound `(run_type, target)`, looked up
  under `root_for(model)`, or `None` when absent — the port's absent/transient taxonomy is
  preserved unchanged (a corrupt frame still raises).
- `root_for(primary_model)` is the `primary_root` the caller supplied; for any other model it
  is `root_of(model)`, by default `ModelPathManager(model, validate=False).data_generated` —
  the ensemble managers' path formula for a constituent (they pass ``validate=True`` and
  raise on a missing model; this resolves and reports absence).
- Absence is **logged at INFO with the root probed**, so a missing row is traceable to a
  directory.
- `provenance()` reads the subject's own frame.
- One file source per model, built lazily and reused within the instance.

---

## 4. Inputs and Assumptions

- `primary_model`, `primary_root`: the subject and its `data_generated`. An ensemble lives
  under `ensembles/`, which `ModelPathManager` does not resolve by name — hence supplied.
- `run_type`, `target`: bound at construction (a report is per target).
- `root_of`: optional `name → Path`; injectable for tests that lay out real per-model
  directories without a views-models checkout.
- Assumes every comparison model is a *model* (under `models/`), which is what constituent and
  baseline lists contain today. A comparison model that is itself an ensemble would resolve
  to `models/<name>/…` and report absent.

---

## 5. Outputs and Side Effects

- Returns `MetricFrame` or `None`; no writes. One INFO line per absent lookup.

---

## 6. Failure Modes and Loudness

- Absent frame → `None` + INFO naming model, run_type, target, root. Loud-but-soft by the
  port's contract; the report announces the missing row.
- Corrupt/unreadable frame → propagates from `MetricFrame.load` (transient, theirs).
- A model name that fails `ModelPathManager.validate_model_name` → `ValueError` from the
  default `root_of` (a config fault; loud).
- views-reporting absent → `ImportError` at first lookup; the reporting stage probes for the
  consumer before constructing this class (`_require_evaluation_source_consumer`).

---

## 7. Boundaries and Interactions

- **Depends on:** `views_reporting.sources.MetricFrameFileSource` (imported lazily — the
  consumer is optional), `ModelPathManager`.
- **Constructed by:** `ReportingStage.generate_evaluation_report`, once per target, and handed to
  `EvaluationReportTemplate.generate(source=…)`.
- **Trust boundary:** views-reporting's seam test (`tests/test_vpc_seam_contract.py`) pins the
  stage's source construction by AST; it must now look for this class (their #287).

---

## 8. Examples of Correct Usage

```python
source = PerModelMetricFrameSource(
    primary_model="big_chungus",
    primary_root=context.model_path.data_generated,   # ensembles/big_chungus/data/generated
    run_type="calibration",
    target="lr_sb",
)
source.metric_frame("big_chungus")    # ensembles/big_chungus/data/generated/big_chungus/calibration/metricframe_lr_sb
source.metric_frame("black_ranger")   # models/black_ranger/data/generated/black_ranger/calibration/metricframe_lr_sb
```

---

## 9. Examples of Incorrect Usage

```python
# One root for everyone — the #485 shape. The subject resolves; every other model probes a
# directory nothing writes and comes back None.
MetricFrameFileSource(root=context.model_path.data_generated, ...)
```

---

## 10. Test Alignment

- `tests/test_managers/test_metric_frame_source.py` — real `MetricFrameFileSource` over
  production's layout (ensemble root + per-model roots on `tmp_path`); only `MetricFrame.load`
  is stubbed to record the directory handed to it. Covers: constituent and baseline found under
  their own roots; the subject under the supplied root without consulting `root_of`; absence
  → `None` + the root in the log; provenance routed to the subject; the locked layout unchanged
  per root; the default `root_of` is `ModelPathManager(name, validate=False)`.
- `tests/test_managers/test_reporting_stage.py::TestEvaluationReport::test_a_constituent_row_resolves_under_the_constituents_own_directory`
  — the stage end to end: ensemble subject, constituent under `models/`, both found by the
  source the template receives.

---

## 11. Evolution Notes

- If a comparison model can ever be an ensemble, `root_of` needs a second lookup
  (`EnsemblePathManager`); today no config lists one. The trigger is a `models:` or baseline
  entry naming something under `ensembles/`.

---

## End of Contract
