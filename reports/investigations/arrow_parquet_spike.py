"""
Arrow Parquet Spike: Memory-Safe Read/Write for List-in-Cell Parquet
=====================================================================

Proves that all three fixes (A, B, C) are viable on the current environment
(Python, Polars, pyarrow versions installed) BEFORE any production code is
modified.

Six blocks:
  1 — Baseline: pandas write + pandas read  (current broken path)
  2 — Fix A:    pyarrow direct write, no Python list materialisation
  3 — Fix B:    pl.read_parquet() at read time  (no pandas object explosion)
  4 — Fix B+:   pl.scan_parquet() with column pushdown
  5 — Fix C:    Arrow buffer extraction → numpy (no Python list intermediary)
  6 — End-to-end: M=2 models × K=2 targets, full ensemble simulation

Scale:  N=200_000 rows × S=64 samples
        numpy size = 200_000 × 64 × 4 bytes = 51.2 MB
        Python list explosion at this scale ≈ 3.2 GB

Success criteria (from the plan):
  Fix A write peak < 200 MB additional (< 4× numpy size)
  Fix B read  peak < 200 MB additional (< 4× numpy size)
  Fix C arrow→numpy peak < 100 MB      (near zero-copy)

Usage:
    conda run -n views_pipeline python reports/investigations/arrow_parquet_spike.py
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

# ── repo root ──────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── scale constants ────────────────────────────────────────────────────────────
N = 200_000      # rows (slightly above pgm per-origin scale of ~172k × 1 step)
S = 64           # posterior samples (HydraNet production)
NUMPY_MB = N * S * 4 / (1024 ** 2)  # 51.2 MB

TARGET_A = "pred_lr_sb_best"
TARGET_B = "pred_lr_ns_best"
TARGETS = [TARGET_A, TARGET_B]

# ── helpers ────────────────────────────────────────────────────────────────────

_proc = psutil.Process(os.getpid())


def _rss_mb() -> float:
    return _proc.memory_info().rss / (1024 ** 2)


def _gb(mb: float) -> str:
    return f"{mb / 1024:.2f} GB" if mb >= 512 else f"{mb:.0f} MB"


def _header(title: str, extra: str = "") -> None:
    print()
    print("─" * 76)
    print(f"  {title}")
    if extra:
        print(f"  {extra}")
    print("─" * 76)


def _result(label: str, delta_mb: float) -> None:
    sign = "+" if delta_mb >= 0 else ""
    print(f"  {label:<58s}  Δ {sign}{_gb(delta_mb)}")


def _check(cond: bool, msg: str) -> bool:
    marker = "✓" if cond else "✗ FAIL:"
    print(f"    {marker} {msg}")
    return cond


def _make_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic y_pred, times, units."""
    rng = np.random.default_rng(42)
    y_pred = rng.random((N, S)).astype(np.float32)
    times = np.arange(400, 400 + N, dtype=np.int64)
    units = np.arange(1, N + 1, dtype=np.int64)
    return y_pred, times, units


def _build_list_array(y_pred: np.ndarray) -> pa.ChunkedArray:
    """
    Build a pa.ListArray[Float32] from a (N, S) numpy array WITHOUT
    creating any Python list objects.

    Uses zero-copy: pa.array() on a contiguous float32 numpy view
    does not allocate new memory.
    """
    flat_values = pa.array(y_pred.reshape(-1), type=pa.float32())   # zero-copy view
    offsets = pa.array(np.arange(0, (N + 1) * S, S, dtype=np.int32))  # ~0.8 MB offsets
    return pa.ListArray.from_arrays(offsets, flat_values)


# ── Block 1: Baseline ──────────────────────────────────────────────────────────

def block1_baseline(
    tmpdir: Path,
    y_pred: np.ndarray,
    times: np.ndarray,
    units: np.ndarray,
) -> tuple[bool, float, float, Path]:
    """Write via pandas (list comprehension), read via pandas."""
    _header(
        "Block 1 — Baseline: pandas write + pandas read  (current broken path)",
        f"N={N:,}, S={S}, numpy={_gb(NUMPY_MB)}"
    )

    path = tmpdir / "block1_pandas.parquet"
    passed = True

    # ── write ──
    gc.collect()
    r0 = _rss_mb()
    df_write = pd.DataFrame({
        "time": times,
        "unit": units,
        TARGET_A: [list(row) for row in y_pred],   # THE LIST EXPLOSION
    })
    r1 = _rss_mb()
    write_delta = r1 - r0
    _result("after list comprehension + DataFrame creation  ← EXPLOSION", write_delta)

    df_write.to_parquet(path)
    r2 = _rss_mb()
    _result("after df.to_parquet()", r2 - r1)

    del df_write
    gc.collect()
    r3 = _rss_mb()
    _result("after del + gc.collect()", r3 - r0)

    # ── read ──
    gc.collect()
    r4 = _rss_mb()
    df_read = pd.read_parquet(path)
    r5 = _rss_mb()
    read_delta = r5 - r4
    _result("after pd.read_parquet()  ← EXPLOSION", read_delta)

    # correctness
    first_val = df_read[TARGET_A].iloc[0]
    # pandas reads list-in-cell parquet as ndarray (not list) — both are Python objects
    c1 = _check(isinstance(first_val, (list, np.ndarray)), "cell is Python list or ndarray (object dtype)")
    c2 = _check(len(first_val) == S, f"cell length == S ({S})")
    c3 = _check(abs(float(first_val[0]) - float(y_pred[0, 0])) < 1e-5, "first value correct")
    passed = c1 and c2 and c3

    del df_read
    gc.collect()
    r6 = _rss_mb()
    _result("after del + gc.collect()", r6 - r4)

    print(f"\n  Write peak: {_gb(write_delta)}  ({write_delta / NUMPY_MB:.1f}× numpy)")
    print(f"  Read  peak: {_gb(read_delta)}  ({read_delta / NUMPY_MB:.1f}× numpy)")
    print(f"  Status: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed, write_delta, read_delta, path


# ── Block 2: Fix A — pyarrow direct write ─────────────────────────────────────

def block2_fix_a_write(
    tmpdir: Path,
    y_pred: np.ndarray,
    times: np.ndarray,
    units: np.ndarray,
    pandas_path: Path,
) -> tuple[bool, float, Path]:
    """Write via pyarrow directly. No Python list objects created."""
    _header(
        "Block 2 — Fix A: pyarrow direct write (no Python list materialisation)",
        f"N={N:,}, S={S}  success criterion: write peak < 200 MB (< 4× numpy)"
    )

    path = tmpdir / "block2_arrow.parquet"
    passed = True

    gc.collect()
    r0 = _rss_mb()

    # Build the ListArray from flat numpy (zero-copy for the values buffer)
    list_array = _build_list_array(y_pred)
    r1 = _rss_mb()
    _result("after _build_list_array() — pa.ListArray, zero-copy", r1 - r0)

    # Build the full Arrow table
    table = pa.table({
        "time": times,
        "unit": units,
        TARGET_A: list_array,
        TARGET_B: list_array,   # second target — same data for simplicity
    })
    r2 = _rss_mb()
    _result("after pa.table() with 2 targets", r2 - r1)

    # Write to parquet
    pq.write_table(table, path)
    r3 = _rss_mb()
    _result("after pq.write_table()", r3 - r2)

    del table, list_array
    gc.collect()
    r4 = _rss_mb()
    _result("after del + gc.collect()", r4 - r0)

    write_peak = r2 - r0

    # ── FORMAT COMPATIBILITY CHECK: pandas must read same list-in-cell format ──
    print("\n  Format compatibility check (pandas reader):")
    df_pd = pd.read_parquet(path)
    cell_val = df_pd[TARGET_A].iloc[0]
    # pandas reads list-in-cell as ndarray (not list) — both are Python objects (explosion)
    c1 = _check(isinstance(cell_val, (list, np.ndarray)), "pandas reads cell as list/ndarray (object dtype)")
    c2 = _check(len(cell_val) == S, f"cell length == S ({S})")
    c3 = _check(abs(float(cell_val[0]) - float(y_pred[0, 0])) < 1e-5, "value matches original")
    del df_pd
    gc.collect()

    # Compare with pandas-written parquet: both should produce identical data
    df_pandas_written = pd.read_parquet(pandas_path)
    df_arrow_written = pd.read_parquet(path)
    try:
        arr_pandas = np.array(df_pandas_written[TARGET_A].tolist(), dtype=np.float32)
        arr_arrow  = np.array(df_arrow_written[TARGET_A].tolist(), dtype=np.float32)
        np.testing.assert_allclose(arr_pandas, arr_arrow, rtol=1e-5, atol=1e-6)
        c4 = _check(True, "pyarrow-written parquet matches pandas-written (round-trip parity)")
    except AssertionError as e:
        c4 = _check(False, f"round-trip parity FAILED: {e}")
    del df_pandas_written, df_arrow_written
    gc.collect()

    passed = c1 and c2 and c3 and c4

    print(f"\n  Write peak: {_gb(write_peak)}  ({write_peak / NUMPY_MB:.1f}× numpy)")
    criterion = write_peak < 200
    print(f"  Criterion (< 200 MB): {'PASS ✓' if criterion else 'FAIL ✗'}")
    print(f"  Status: {'PASS ✓' if passed and criterion else 'FAIL ✗'}")
    return passed and criterion, write_peak, path


# ── Block 3: Fix B — pl.read_parquet() ────────────────────────────────────────

def block3_fix_b_read(
    arrow_path: Path,
    y_pred: np.ndarray,
) -> tuple[bool, float, pl.DataFrame]:
    """Read the pyarrow-written parquet with pl.read_parquet()."""
    _header(
        "Block 3 — Fix B: pl.read_parquet() (no pandas object explosion)",
        f"N={N:,}, S={S}  success criterion: read peak < 200 MB (< 4× numpy)"
    )

    passed = True

    gc.collect()
    r0 = _rss_mb()
    df_pl = pl.read_parquet(arrow_path)
    r1 = _rss_mb()
    read_peak = r1 - r0
    _result("after pl.read_parquet()", read_peak)

    # Correctness: dtype must be List(Float32), not Object
    col_dtype = df_pl[TARGET_A].dtype
    c1 = _check(isinstance(col_dtype, pl.List), f"dtype is Polars List, not object (got: {col_dtype})")
    c2 = _check(col_dtype == pl.List(pl.Float32), f"inner dtype is Float32 (got: {col_dtype})")
    c3 = _check(df_pl.height == N, f"row count == N ({N:,})")

    # Spot-check first row value (no full Python materialisation)
    first_cell = df_pl[TARGET_A][0].to_list()
    c4 = _check(len(first_cell) == S, f"first cell length == S ({S})")
    c5 = _check(abs(first_cell[0] - float(y_pred[0, 0])) < 1e-5, "first value correct")

    passed = c1 and c2 and c3 and c4 and c5
    criterion = read_peak < 200

    print(f"\n  Read peak: {_gb(read_peak)}  ({read_peak / NUMPY_MB:.1f}× numpy)")
    print(f"  Criterion (< 200 MB): {'PASS ✓' if criterion else 'FAIL ✗'}")
    print(f"  Status: {'PASS ✓' if passed and criterion else 'FAIL ✗'}")
    return passed and criterion, read_peak, df_pl


# ── Block 4: Fix B+ — pl.scan_parquet() with column pushdown ──────────────────

def block4_fix_b_plus(arrow_path: Path) -> tuple[bool, float]:
    """Lazy scan with column selection — only reads the needed columns from disk."""
    _header(
        "Block 4 — Fix B+: pl.scan_parquet() with column pushdown",
        f"File has {len(TARGETS)} target columns; scan reads only 1"
    )

    passed = True

    gc.collect()
    r0 = _rss_mb()

    # Lazy scan: only materialise the one needed column
    df_lazy = (
        pl.scan_parquet(arrow_path)
        .select(["time", "unit", TARGET_A])
        .collect()
    )
    r1 = _rss_mb()
    scan_peak = r1 - r0
    _result(f"after scan_parquet().select(['time','unit','{TARGET_A}']).collect()", scan_peak)

    c1 = _check(TARGET_A in df_lazy.columns, f"'{TARGET_A}' column present")
    c2 = _check(TARGET_B not in df_lazy.columns, f"'{TARGET_B}' NOT materialised (column pushdown)")
    c3 = _check(df_lazy.height == N, f"row count == N ({N:,})")
    c4 = _check(isinstance(df_lazy[TARGET_A].dtype, pl.List), "dtype is List")

    del df_lazy
    gc.collect()

    # Bonus: multi-file lazy scan (simulates reading M=2 model files)
    print("\n  Bonus: pl.scan_parquet([file1, file2]) — parallel multi-file lazy read")
    r2 = _rss_mb()
    # Use the same file twice (simulates two models with identical structure)
    df_multi = (
        pl.scan_parquet([arrow_path, arrow_path])
        .select(["time", "unit", TARGET_A])
        .collect()
    )
    r3 = _rss_mb()
    multi_peak = r3 - r2
    _result("after scan_parquet([f1, f2]).select(...).collect()", multi_peak)
    c5 = _check(df_multi.height == 2 * N, f"stacked row count == 2*N ({2*N:,})")
    del df_multi
    gc.collect()

    passed = c1 and c2 and c3 and c4 and c5
    print(f"\n  Single-file scan peak: {_gb(scan_peak)}")
    print(f"  Multi-file scan peak:  {_gb(multi_peak)}")
    print(f"  Status: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed, scan_peak


# ── Block 5: Fix C — Arrow buffer extraction ───────────────────────────────────

def block5_fix_c_arrow_buffer(
    df_pl: pl.DataFrame,
    y_pred: np.ndarray,
) -> tuple[bool, float]:
    """
    Extract numpy array from Polars list column via Arrow buffers.
    No Python list intermediary — replaces .to_list() + np.asarray().
    """
    _header(
        "Block 5 — Fix C: Arrow buffer extraction → numpy (no Python list)",
        f"N={N:,}, S={S}  success criterion: peak < 100 MB (near zero-copy)"
    )

    passed = True

    # ── Baseline: current broken aggregator path ──
    print("  [Baseline] Current aggregator path (.to_list() + np.asarray()):")
    gc.collect()
    rb0 = _rss_mb()
    col_vals = df_pl[TARGET_A].to_list()   # Polars → Python lists  ← EXPLOSION
    rb1 = _rss_mb()
    _result("after df[col].to_list()  ← EXPLOSION", rb1 - rb0)
    arr_baseline = np.asarray(col_vals, dtype=np.float64)   # (N, S)
    rb2 = _rss_mb()
    _result("after np.asarray(col_vals, dtype=float64)", rb2 - rb1)
    del col_vals, arr_baseline
    gc.collect()
    rb3 = _rss_mb()
    _result("after del + gc.collect()", rb3 - rb0)
    baseline_peak = rb2 - rb0
    print(f"    baseline peak: {_gb(baseline_peak)}  ({baseline_peak / NUMPY_MB:.1f}× numpy)")

    # ── Fix C: via Arrow flatten + to_numpy ──
    # NOTE: Polars series.to_arrow() returns a single Arrow array (LargeListArray),
    # NOT a ChunkedArray — so no combine_chunks() is needed.
    print("\n  [Fix C] Arrow path: series.to_arrow().flatten().to_numpy():")
    gc.collect()
    r0 = _rss_mb()

    # Step 1: Convert Polars series to Arrow (zero-copy, backed by Polars Arrow pool)
    arrow_col = df_pl[TARGET_A].to_arrow()
    r1 = _rss_mb()
    arrow_type_name = type(arrow_col).__name__
    _result(f"after series.to_arrow()  [{arrow_type_name}]", r1 - r0)
    # Polars uses LargeListArray (int64 offsets) — document this for the codebase
    _check(
        isinstance(arrow_col, (pa.ListArray, pa.LargeListArray, pa.ChunkedArray)),
        f"result is Arrow array (got {arrow_type_name})"
    )

    # If it's a ChunkedArray (multiple chunks), consolidate first
    if isinstance(arrow_col, pa.ChunkedArray):
        arrow_col = arrow_col.combine_chunks()
        r1b = _rss_mb()
        _result("after combine_chunks() (was ChunkedArray)", r1b - r1)
        r1 = r1b

    # ── Inspect buffer layout ──
    print("\n  Buffer layout (documenting for the codebase):")
    all_buffers = arrow_col.buffers()
    print(f"    Arrow type:    {arrow_col.type}")
    print("    Offset dtype:  int64 (LargeList) vs int32 (List)")
    print(f"    Total buffers in depth-first listing: {len(all_buffers)}")
    expected_values_size = N * S * 4   # float32 values
    expected_offsets_int32 = (N + 1) * 4   # int32 offsets
    expected_offsets_int64 = (N + 1) * 8   # int64 offsets (LargeList)
    for i, buf in enumerate(all_buffers):
        if buf is None:
            print(f"    [{i}] = None (validity bitmap — no nulls)")
        else:
            tag = ""
            if buf.size == expected_offsets_int32:
                tag = "  ← OFFSETS (int32, List)"
            elif buf.size == expected_offsets_int64:
                tag = "  ← OFFSETS (int64, LargeList)"
            elif buf.size == expected_values_size:
                tag = "  ← VALUES (float32)"
            print(f"    [{i}] = Buffer(size={buf.size:,} bytes){tag}")

    # ── Determine flatten method ──
    # Both ListArray and LargeListArray support .flatten() in pyarrow >= 0.17
    r2 = r1

    # Step 2: Extract flat values via flatten (works for both List and LargeList)
    flat_arrow = arrow_col.flatten()   # pa.Float32Array of length N*S
    r3 = _rss_mb()
    _result("after arrow_col.flatten()", r3 - r2)
    _check(
        hasattr(flat_arrow, "type") and flat_arrow.type == pa.float32(),
        f"flat_arrow dtype is float32 (got type={getattr(flat_arrow, 'type', '?')})"
    )

    # Step 3: Zero-copy view into Arrow memory pool
    try:
        flat_np = flat_arrow.to_numpy(zero_copy_only=True)
        _check(True, "flat_arrow.to_numpy(zero_copy_only=True) — genuine zero-copy")
    except pa.lib.ArrowInvalid as e:
        print(f"    to_numpy(zero_copy_only=True) failed ({e}); falling back to copy")
        flat_np = flat_arrow.to_numpy()

    r4 = _rss_mb()
    _result("after flat_arrow.to_numpy(zero_copy_only=True)", r4 - r3)

    # Step 4: Reshape — also a view (no copy) if contiguous
    arr = flat_np.reshape(N, S)
    r5 = _rss_mb()
    _result("after flat_np.reshape(N, S)  — view, no copy", r5 - r4)

    fix_c_peak = r5 - r0

    # ── Also try the buffers() approach described in the plan ──
    print("\n  [Fix C alt] buffers() approach (scanning by size to find values):")
    print("    NOTE: The plan says buffers()[1] = values, but that's only true for")
    print("    specific Arrow versions/types. Polars emits LargeListArray (int64 offsets).")
    print("    Correct approach: use arrow_col.flatten().to_numpy() for portability.")
    values_buf_found = None
    values_buf_idx = None
    for i, buf in enumerate(all_buffers):
        if buf is not None and buf.size == expected_values_size:
            values_buf_found = buf
            values_buf_idx = i
            break

    if values_buf_found is not None:
        arr_from_buf = np.frombuffer(values_buf_found, dtype=np.float32).reshape(N, S)
        try:
            np.testing.assert_allclose(arr_from_buf, arr, rtol=1e-5, atol=0)
            _check(True, f"buffers()[{values_buf_idx}] approach produces identical result")
        except AssertionError as e:
            _check(False, f"buffers()[{values_buf_idx}] mismatch: {e}")
    else:
        print("    Values buffer not found by size scan — flatten().to_numpy() is the safe path.")

    # ── Correctness: compare against original y_pred ──
    print("\n  Correctness check (vs original y_pred, float32 tolerance):")
    try:
        np.testing.assert_allclose(arr, y_pred, rtol=1e-5, atol=1e-6)
        c_correct = _check(True, "arr matches original y_pred within float32 tolerance")
    except AssertionError as e:
        c_correct = _check(False, f"mismatch: {e}")

    c_dtype  = _check(arr.dtype == np.float32, f"dtype is float32 (got {arr.dtype})")
    c_shape  = _check(arr.shape == (N, S), f"shape is ({N}, {S})")
    passed = c_correct and c_dtype and c_shape

    criterion = fix_c_peak < 100
    print(f"\n  Fix C peak:     {_gb(fix_c_peak)}  ({fix_c_peak / NUMPY_MB:.1f}× numpy)")
    print(f"  Baseline peak:  {_gb(baseline_peak)}  ({baseline_peak / NUMPY_MB:.1f}× numpy)")
    print(f"  Speedup:        {baseline_peak / max(fix_c_peak, 0.1):.1f}× less memory")
    print(f"  Criterion (< 100 MB): {'PASS ✓' if criterion else 'FAIL ✗'}")
    print(f"  Status: {'PASS ✓' if passed and criterion else 'FAIL ✗'}")

    del arrow_col, flat_arrow, flat_np, arr
    gc.collect()
    return passed and criterion, fix_c_peak


# ── Block 6: End-to-end pipeline simulation ────────────────────────────────────

def block6_end_to_end(
    tmpdir: Path,
    y_pred: np.ndarray,
    times: np.ndarray,
    units: np.ndarray,
) -> tuple[bool, float]:
    """
    Full simulation: M=2 models × K=2 targets.
    Fix A → write, Fix B → read, Fix C → stack, aggregate → assert correctness.
    """
    _header(
        "Block 6 — End-to-end simulation: M=2 models × K=2 targets",
        f"N={N:,}, S={S}"
    )

    passed = True

    # ── Write M=2 models using Fix A ──
    print("  Phase 1: Write M=2 model parquets using pyarrow (Fix A)...")
    model_paths = []
    gc.collect()
    r0 = _rss_mb()

    for m_idx in range(2):
        # Slightly different data per model (add small offset)
        y_m = (y_pred + m_idx * 0.01).astype(np.float32)

        list_a = _build_list_array(y_m)
        table = pa.table({
            "time": times,
            "unit": units,
            TARGET_A: list_a,
            TARGET_B: list_a,
        })
        path = tmpdir / f"model_{m_idx}.parquet"
        pq.write_table(table, path)
        model_paths.append(path)
        del table, list_a, y_m
        gc.collect()

    r1 = _rss_mb()
    _result("after writing 2 model parquets (Fix A)", r1 - r0)

    # ── Read M=2 models using Fix B ──
    print("\n  Phase 2: Read M=2 model parquets using pl.read_parquet() (Fix B)...")
    dfs = []
    for m_idx, path in enumerate(model_paths):
        gc.collect()
        r_before = _rss_mb()
        df = pl.read_parquet(path)
        r_after = _rss_mb()
        _result(f"  model_{m_idx}: pl.read_parquet()", r_after - r_before)
        dfs.append(df)

    r2 = _rss_mb()
    _result("after reading both models (cumulative)", r2 - r0)

    # ── Extract via Fix C and stack ──
    print("\n  Phase 3: Arrow buffer → numpy per model (Fix C), stack to (N, M, S)...")
    gc.collect()
    r3 = _rss_mb()

    # Extract each model's data into numpy arrays
    # Polars .to_arrow() returns LargeListArray directly (no combine_chunks needed)
    model_arrays = []
    for m_idx, df in enumerate(dfs):
        arrow_col = df[TARGET_A].to_arrow()   # pa.LargeListArray or pa.ListArray
        if isinstance(arrow_col, pa.ChunkedArray):
            arrow_col = arrow_col.combine_chunks()
        flat_np = arrow_col.flatten().to_numpy(zero_copy_only=False)
        arr_m = flat_np.reshape(N, S).astype(np.float32)
        model_arrays.append(arr_m)
        del arrow_col, flat_np
        gc.collect()

    r4 = _rss_mb()
    _result("after Fix C extraction for both models", r4 - r3)

    # Stack: (N, S) × M → (N, M, S)
    model_stack = np.stack(model_arrays, axis=1)  # (N, M=2, S)
    r5 = _rss_mb()
    _result("after np.stack to (N, M, S)", r5 - r4)

    # Aggregate: mean across models → (N, S)
    ensemble_mean = model_stack.mean(axis=1)   # (N, S)
    r6 = _rss_mb()
    _result("after ensemble_mean = model_stack.mean(axis=1)", r6 - r5)

    # ── Correctness check ──
    print("\n  Correctness checks:")
    c1 = _check(model_stack.shape == (N, 2, S), f"model_stack.shape == ({N}, 2, {S})")
    c2 = _check(ensemble_mean.shape == (N, S), f"ensemble_mean.shape == ({N}, {S})")

    # The mean of models 0 and 1 should be y_pred + 0.005
    expected_mean = (y_pred + 0.005).astype(np.float32)
    try:
        np.testing.assert_allclose(ensemble_mean, expected_mean, rtol=1e-4, atol=1e-5)
        c3 = _check(True, "ensemble_mean values correct (mean of model_0 and model_1)")
    except AssertionError as e:
        c3 = _check(False, f"ensemble_mean mismatch: {e}")

    c4 = _check(ensemble_mean.dtype == np.float32, "output dtype is float32")

    passed = c1 and c2 and c3 and c4
    total_peak = r6 - r0

    del dfs, model_arrays, model_stack, ensemble_mean
    gc.collect()
    r7 = _rss_mb()
    _result("after del + gc.collect()", r7 - r0)

    print(f"\n  Total end-to-end peak: {_gb(total_peak)}")
    print(f"  Status: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed, total_peak


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    print()
    print("=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    print(f"  {'Block':<8}  {'Name':<46}  {'Status':<8}  {'Peak'}")
    print("  " + "-" * 72)
    for key, (passed, peak_mb, name) in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        peak_s = _gb(peak_mb) if peak_mb != float("inf") else "OOM"
        print(f"  {key:<8}  {name:<46}  {status:<8}  {peak_s}")

    print()
    print("  Interpretation:")
    print("  - Block 1: documents the current baseline explosion (~many× numpy)")
    print("  - Block 2 PASS → Fix A (pyarrow write) is safe: no list explosion at write time")
    print("  - Block 3 PASS → Fix B (pl.read_parquet) is safe: no list explosion at read time")
    print("  - Block 4 PASS → Fix B+ (lazy scan) adds free I/O win via column pushdown")
    print("  - Block 5 PASS → Fix C (Arrow buffers) is safe: near zero-copy list→numpy")
    print("  - Block 6 PASS → Full end-to-end fix works and produces correct aggregation")
    print()
    print("  See the plan document for integration guidance:")
    print("  reports/for_hydranet/  and  reports/investigations/hydranet_memory_investigation.md")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  views-pipeline-core · Arrow Parquet Spike                              ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  pyarrow version: {pa.__version__}")
    print(f"  polars  version: {pl.__version__}")
    print(f"  numpy   version: {np.__version__}")
    print(f"  Scale:  N={N:,} rows × S={S} samples")
    print(f"  numpy array size: {_gb(NUMPY_MB)}  ({NUMPY_MB:.1f} MB)")
    print(f"  Python list explosion expected: ~{NUMPY_MB * 60 / NUMPY_MB:.0f}× numpy at full pgm scale")
    print(f"  Process RSS at start: {_rss_mb():.0f} MB")
    print()

    tmpdir = Path(tempfile.mkdtemp(prefix="arrow_spike_"))
    results = {}

    try:
        # Prepare synthetic data once
        print("  Creating synthetic y_pred (N, S) float32...")
        gc.collect()
        r0 = _rss_mb()
        y_pred, times, units = _make_data()
        r1 = _rss_mb()
        print(f"  y_pred created: {_gb(r1 - r0)} RSS growth  (expected ~{_gb(NUMPY_MB)})")

        # ── Block 1 ──
        try:
            passed, write_pk, read_pk, pandas_path = block1_baseline(tmpdir, y_pred, times, units)
            results["1"] = (passed, max(write_pk, read_pk), "Baseline: pandas write+read (broken path)")
        except Exception as e:
            print(f"\n  ✗ Block 1 ERROR: {type(e).__name__}: {e}")
            results["1"] = (False, float("inf"), "Baseline: pandas write+read (ERROR)")
            pandas_path = None
            raise

        # ── Block 2 ──
        try:
            passed, write_pk, arrow_path = block2_fix_a_write(tmpdir, y_pred, times, units, pandas_path)
            results["2"] = (passed, write_pk, "Fix A: pyarrow direct write")
        except Exception as e:
            print(f"\n  ✗ Block 2 ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results["2"] = (False, float("inf"), "Fix A: pyarrow direct write (ERROR)")
            arrow_path = None

        # ── Block 3 ──
        df_pl = None
        if arrow_path is not None:
            try:
                passed, read_pk, df_pl = block3_fix_b_read(arrow_path, y_pred)
                results["3"] = (passed, read_pk, "Fix B: pl.read_parquet()")
            except Exception as e:
                print(f"\n  ✗ Block 3 ERROR: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results["3"] = (False, float("inf"), "Fix B: pl.read_parquet() (ERROR)")
        else:
            results["3"] = (False, float("inf"), "Fix B: pl.read_parquet() (SKIPPED — no arrow_path)")

        # ── Block 4 ──
        if arrow_path is not None:
            try:
                passed, scan_pk = block4_fix_b_plus(arrow_path)
                results["4"] = (passed, scan_pk, "Fix B+: pl.scan_parquet() column pushdown")
            except Exception as e:
                print(f"\n  ✗ Block 4 ERROR: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results["4"] = (False, float("inf"), "Fix B+: pl.scan_parquet() (ERROR)")
        else:
            results["4"] = (False, float("inf"), "Fix B+: pl.scan_parquet() (SKIPPED)")

        # ── Block 5 ──
        if df_pl is not None:
            try:
                passed, buf_pk = block5_fix_c_arrow_buffer(df_pl, y_pred)
                results["5"] = (passed, buf_pk, "Fix C: Arrow buffer → numpy")
            except Exception as e:
                print(f"\n  ✗ Block 5 ERROR: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results["5"] = (False, float("inf"), "Fix C: Arrow buffer → numpy (ERROR)")
        else:
            results["5"] = (False, float("inf"), "Fix C: Arrow buffer → numpy (SKIPPED — no df_pl)")

        # ── Block 6 ──
        try:
            passed, e2e_pk = block6_end_to_end(tmpdir, y_pred, times, units)
            results["6"] = (passed, e2e_pk, "End-to-end: M=2 models × K=2 targets")
        except Exception as e:
            print(f"\n  ✗ Block 6 ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results["6"] = (False, float("inf"), "End-to-end simulation (ERROR)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print_summary(results)

    all_passed = all(v[0] for v in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
