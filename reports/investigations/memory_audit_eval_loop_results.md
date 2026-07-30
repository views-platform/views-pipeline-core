# Memory Audit Results: Evaluation Loop Under the PredictionFrame Path

**Date**: 2026-03-10
**Script**: `reports/investigations/memory_audit_eval_loop.py`
**Scale**: 13 sequences × 2 targets × 32,400 cells × 36 steps × 32 samples (float32)
**PF size at this scale**: 142 MB (y_pred) + ~37 MB (identifiers) = ~179 MB per PF

---

## Verdict Summary

| Hypothesis | Result | Actual vs Expected |
|-----------|--------|--------------------|
| H1 — PF accumulation is linear | ✅ CONFIRMED | +320 MB/seq (expected 285; diff = identifier arrays) |
| H2 — inference tensors don't release | ✅ FALSIFIED (numpy) | numpy mmap releases to OS immediately; del+gc sufficient |
| H3 — from_prediction_frames is the largest spike | ✅ CONFIRMED | +3,138 MB peak; del+gc recovers correctly |
| H4 — un-freed input data (X) accumulates | ✅ CONFIRMED | +2,145 MB excess when X not freed |
| H5 — Python allocator fragmentation | ✅ CONFIRMED (small-obj only) | Large numpy (>1 MB) uses mmap; returns to OS. Small Python objects DO fragment. |
| H6 — to_prediction_df() list explosion | ✅ CONFIRMED BUT 4.7× WORSE | Peak: 4,766 MB (predicted 997 MB); residual: 2,265 MB stuck |

---

## H1 — PF Linear Accumulation: CONFIRMED

**Observation**: every sequence × target append adds exactly +320 MB to RSS.
- 142 MB (y_pred float32) + ~37 MB (4 identifier arrays × int64) + object overhead ≈ 180 MB per PF × 2 targets = 360 MB. Actual is 320 MB — close, with minor process-level variance.
- After 13 sequences: +4,166 MB total.
- `del raw_preds; gc.collect()` recovers **all of it** (RSS drops from 5,031 → 1,026 MB).

**Key finding**: accumulation is deterministic and fully recoverable. The growth between sequences is **not PyTorch** — it is pure numpy accumulation, and it behaves correctly.

---

## H2 — Inference Tensor Release: FALSIFIED (for numpy)

**Observation**: simulated "torch tensor" (712 MB numpy alloc) frees exactly 391 MB after `del` (the remaining 321 MB = the PF copy created from it). `gc.collect()` adds nothing.

**Why**: large numpy arrays (>1–2 MB) use `mmap(2)` for allocation. When freed, the OS reclaims the pages immediately. No arena fragmentation.

**Critical caveat — PyTorch is different**: PyTorch uses a **caching memory allocator** (`c10::CachingAllocator`). After `del tensor`, PyTorch returns memory to its **internal pool** but **not to the OS**. RSS stays high between sequences. This is the most likely cause of the user's observed spikes between rolling origins in HydraNet. This script cannot test PyTorch behavior — it must be verified in the HydraNet repo with a torch-specific probe.

**Action needed**: In HydraNet's `_evaluate_model_artifact`, add `torch.cuda.empty_cache()` (for GPU) or check whether CPU tensors are being explicitly freed. Without this, every inference sequence holds ~N× output size in PyTorch's pool even after the Python tensor is deleted.

---

## H3 — from_prediction_frames Peak: CONFIRMED

**Observation**:
- Building 13 PFs: +2,121 MB (RSS: 865 → 2,986 MB)
- Calling `from_prediction_frames()`: +3,138 MB additional (RSS: 2,986 → 6,124 MB)
- `del predictions; gc.collect()`: –1,851 MB (frees input PFs, EF remains)
- `del ef; gc.collect()`: –2,429 MB (frees EF + identifier arrays)

**Peak**: 6,124 MB — all of raw_preds + EF output coexist simultaneously. With 2 targets processed sequentially (pop-per-target pattern), the worst-case peak is:

```
remaining raw_preds (1 target × 13 seqs = 1.95 GB)
+ current target's 13 PFs (1.95 GB)
+ EF y_pred_out (1.95 GB)
+ identifier arrays (~0.9 GB)
────────────────────────────────
Peak: ~6.75 GB  (pipeline-core only; no inference tensors)
```

**del+gc pattern works correctly**: each call to `from_prediction_frames` is self-contained. The pop/del/gc pattern in `_evaluate_prediction_dataframe()` is correct. No leak.

---

## H4 — Un-freed Input Data: CONFIRMED

**Observation**:
- Without `del X`: total RSS +5,535 MB after 13 sequences (PF + X both accumulate)
- With `del X`: total RSS +3,390 MB after 13 sequences (PF only)
- Difference: **+2,145 MB** from un-freed viewser data (13 × ~178 MB)

**Action needed**: HydraNet's `_evaluate_model_artifact` loop must explicitly `del X` (and any data tensors) after each sequence's inference completes. This is model-repo responsibility but has a ~2 GB impact.

---

## H5 — Python Allocator Fragmentation: CONFIRMED (small objects only)

**Observation**:
- Large numpy arrays: fragmentation = 0. mmap pages are returned to OS on `del`.
- Python small objects (lists, floats, object-dtype arrays): **DO fragment**. After `to_prediction_df()` allocates and frees ~37 million Python float objects + 1.2 million Python list objects, ~2.3 GB of OS pages remain "stuck" in the process RSS for the lifetime of the process. (Python's malloc arenas are not returned to OS unless fully empty.)

---

## H6 — to_prediction_df() List Explosion: CONFIRMED AND FAR WORSE THAN PREDICTED

**Observation**:
- Predicted peak: 7× PF size = 997 MB
- **Actual peak: 4,766 MB (+33×)**
- After `del df_for_save; gc.collect()`: recovers –2,501 MB
- After `del pf; gc.collect()`: **0 MB recovered** — RSS stays 2,265 MB above baseline **permanently**

```
Phase                      RSS       Delta
─────────────────────────────────────────
Baseline                  1,844 MB   —
After create PF           1,844 MB  +0 MB   (fits in pre-allocated pages)
After to_prediction_df()  6,610 MB  +4,766 MB  ← PEAK
After del df + gc         4,109 MB  -2,501 MB
After del pf + gc         4,109 MB  +0 MB    ← FRAGMENTATION
```

**Mechanism**: `[list(row) for row in pf.y_pred]` creates 37.3M Python float objects + 1.17M Python list objects. CPython's pymalloc allocates these in 256 KB "arenas". Arenas are only returned to the OS when every object within them is freed. After freeing the DataFrame and GC-ing, many arenas still contain unrelated live objects → ~2.3 GB of pages permanently attached to the process.

**Why 33× not 7×**:
- Python list header per row: 56 bytes + 8 bytes × 32 pointers = 312 bytes
- Python floats per row: 32 × 28 bytes = 896 bytes
- Total per row: 1,208 bytes vs 128 bytes numpy (32 × 4) = 9.4× at steady state
- But during DataFrame construction pandas creates additional intermediate representations → measured 33× peak

**This is the smoking gun** for the current memory pressure. Even with `del df_for_save` in the save loop (DoD #2.6), the peak of 4,766 MB + raw_preds (3.9 GB) = **8.7 GB** occurs during the first `to_prediction_df()` call. The fragmentation then permanently inflates process RSS by ~2.3 GB.

---

## Root Cause Ranking (Pipeline-Core Side)

| Rank | Cause | Peak contribution | Recoverable? |
|------|-------|-------------------|--------------|
| 1 | `to_prediction_df()` list explosion (H6) | +4,766 MB peak; +2,265 MB permanent fragmentation | Partial — peak yes, fragmentation no |
| 2 | `from_prediction_frames()` coexistence (H3) | +3,138 MB (input + output coexist) | Yes — del+gc works |
| 3 | PF accumulation (H1) | +4,166 MB (13 seq × 2 tgt) | Yes — del+gc works |
| 4 | Un-freed input data X (H4) | +2,145 MB | Yes — del X in model repo |
| 5 | PyTorch caching allocator (H2, not tested) | Unknown — likely significant | Requires `torch.cuda.empty_cache()` in model repo |

---

## What H6 Means for `to_prediction_df()`

The `to_prediction_df()` method is called in the **save loop** (pipeline-core, DoD #2.6 fixed it to one-at-a-time). Even after the fix, the first call causes a 4.8 GB peak while raw_preds (3.9 GB) is still in memory = **8.7 GB combined**.

The fragmentation means this 4.8 GB peak is not fully reversible. Subsequent calls reuse the arenas, so they don't compound — but the process permanently holds ~2.3 GB of OS pages as a result of any single call.

**Implication**: `to_prediction_df()` is not just a convenience method — it is a significant memory liability. The list-in-cell format (required for the ensemble cross-repo contract) is fundamentally incompatible with large-scale prediction storage without a streaming/chunked write path that avoids materializing the full DataFrame in memory.

---

## Recommended Next Steps (Prioritized)

### Immediate (model-repo side)
1. **Verify HydraNet: torch memory management** — check for `torch.no_grad()` in inference loop, explicit `del tensor`, and `torch.cuda.empty_cache()` or `gc.collect()` after each sequence. This is H2 for PyTorch and is likely the largest real-world contributor.
2. **Verify HydraNet: del X after each sequence** — confirming H4 fix.

### Pipeline-core side
3. **Streaming write for `to_prediction_df()`** — instead of constructing a full in-memory DataFrame then writing to parquet, write row groups directly to parquet via pyarrow without materializing the Python list structure:
   ```python
   # Avoids [list(row) for row in pf.y_pred] entirely
   import pyarrow as pa
   import pyarrow.parquet as pq
   # Write pf.y_pred rows as a pyarrow array directly
   ```
   This eliminates both the 4.8 GB peak and the 2.3 GB fragmentation for each save.

4. **Phase 2 (streaming evaluation)** — per-sequence inference + save + evaluate reduces raw_preds from 13 × PF × targets to 1 × PF × targets. Combined with fix #3, peak would be ~1 GB per sequence.

---

## Scaling Projection (float32, theoretical)

```
Scenario                    PF size   raw_preds   EF peak/tgt   save transient   est total peak
─────────────────────────────────────────────────────────────────────────────────────────────────
current (32 samp, pgm)       0.1 GB       3.6 GB        3.6 GB          1.0 GB (theory)  ~8+ GB*
252 samp, pgm                1.1 GB      28.5 GB       28.5 GB          7.7 GB           ~64+ GB
32 samp, global grid         1.1 GB      28.9 GB       28.9 GB          7.8 GB           ~65+ GB
252 samp, global grid        8.8 GB     227.8 GB      227.8 GB         61.3 GB          ~516+ GB
```
*Actual peak at current scale measured at ~8.7 GB (pipeline-core only). HydraNet inference adds more.
