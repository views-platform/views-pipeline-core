# Migration Plan: Upstreaming the Adapter to Orchestration

**Status:** Ready for Execution
**Target Repo:** `views-pipeline-core`

## 1. Rationale
To fulfill the "Uncle Bob" mandate (ADR-011) and the "Pure Math Engine" vision (ADR-010), the Evaluation repository must stop performing data alignment. Alignment logic (the "Join") is inherently an orchestration concern and belongs in the Pipeline Core.

## 2. Phase 1: Dual-Entry Support (COMPLETED)
We have refactored `EvaluationManager.evaluate` to accept a pre-adapted `EvaluationFrame`.
- **New Param**: `ef: EvaluationFrame`
- **New Param**: `verify_parity: bool`

## 3. Phase 2: Orchestrator Integration (Execution Steps)
The following steps should be taken in the `views-pipeline-core` repository:

1. **Mirror Adapter**: Copy the logic from `views_evaluation/adapters/pandas.py` into the Pipeline Core.
2. **Shadow Run**: Update the evaluation step in Pipeline Core to:
   - Perform adaptation locally.
   - Call `manager.evaluate(actual, preds, target, config, ef=local_ef, verify_parity=True)`.
3. **Observation**: Monitor for `ValueError: Parity Failure`. If none occur over a representative set of runs, bit-wise parity is confirmed.

## 4. Phase 3: The Purge
Once parity is confirmed:
1. **Switch**: Pass only `ef` to the Evaluation repo.
2. **Cleanup**: Remove `PandasAdapter`, `EvaluationManager`, and the `pandas` dependency from the Evaluation repository.

The system will then be a "Pure Math Engine."
