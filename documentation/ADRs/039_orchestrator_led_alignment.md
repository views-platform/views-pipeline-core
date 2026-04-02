# ADR-057: Orchestrator-Led Data Alignment & PredictionFrame Standard

**Status:** Accepted  
**Date:** 2026-02-27  
**Deciders:** Project maintainers, Gemini CLI  

---

## Context

The legacy forecasting evaluation path relied on the `views-evaluation` library to perform data alignment. This required the library to depend on `pandas` and understand MultiIndex intersection logic.

Furthermore, transporting 1000+ posterior samples as lists inside Pandas cells ("list-in-cell") led to massive memory bloat and slow serialization, especially for high-resolution models like HydraNet.

## Decision

To fulfill the "Pure Math Engine" vision, we will migrate data alignment ownership to the Orchestrator (`views-pipeline-core`) and standardize on a framework-agnostic transport container.

### 1. Alignment Responsibility
The "Join" (intersection of truth and predictions) is now an orchestration concern. 
- The Pipeline Core will load actuals and predictions.
- The Pipeline Core will use a local `EvaluationAdapter` to align these datasets into a synchronized set of NumPy arrays.
- The Evaluation library will receive only "pure" data (no Pandas dependencies).

### 2. The PredictionFrame Standard
We introduce `PredictionFrame` as the universal inference output container.
- **Internal State**: Synchronized NumPy arrays for `y_pred` and `identifiers`.
- **Decoupling**: Models return `PredictionFrame`, allowing them to bypass Pandas entirely.
- **Explicit Metadata**: All spatiotemporal identifiers must be explicitly provided (as per ADR-058).

### 3. Dual-Track Migration & Parity Proving
To ensure zero regression during the transition:
- The orchestrator will implement an `_audit_parity` logic.
- It will run both the legacy (library-internal) and new (orchestrator-led) paths simultaneously.
- Results must match bit-for-bit (verified via `pd.testing.assert_frame_equal` on result DataFrames).

## Consequences

### Positive
- **Performance**: Dense NumPy tensors replace bloated Pandas objects, reducing memory footprint by ~10x for sample-based forecasts.
- **Decoupling**: Evaluation repo becomes a pure math engine with minimal dependencies.
- **Transparency**: Data alignment errors fail loudly in the orchestrator before reaching the math kernel.

### Negative
- Temporary complexity increase due to dual-track parity auditing.
- Requires all models to eventually migrate to the `PredictionFrame` contract.
