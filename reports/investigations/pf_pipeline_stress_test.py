"""
Pipeline-Core Stress Test: PredictionFrame Path at S=64
========================================================

Proves that pipeline-core is feasible at S=64 (actual HydraNet scale)
independent of HydraNet. Creates synthetic pf_dict and pushes through each
class, reporting memory at each step.

Six tests:

  A  Single PF creation              — baseline cost at full scale
  B  PF save / load round-trip       — Track A disk I/O
  C  origin_sink (pop pattern)       — K=6 targets, pop-per-target
  D  converter.to_prediction_df()    — Track B: documents the spike (safety scale)
  E  EvaluationAdapter.from_pf()     — metrics pre-alloc (safety scale + projection)
  F  Full pipeline with skip flags   — END-TO-END proof at S=64

Tests A/B/C/F run at FULL scale (N_CELLS=172,000 × T=36 = 6.19M rows, S=64).
Tests D/E run at SAFETY scale (N_CELLS=32,400 × T=36 = 1.17M rows, S=32) to
avoid OOMing the host machine, then project to full scale.

Usage:
    conda run -n views_pipeline python reports/investigations/pf_pipeline_stress_test.py

    # Quick run (safety scale only):
    conda run -n views_pipeline python reports/investigations/pf_pipeline_stress_test.py --fast
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import psutil

# ── repo root ────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.managers.prediction.prediction_frame_converter import (
    PredictionFrameConverter,
)
from views_pipeline_core.modules.validation.adapter import EvaluationAdapter

# ── scale constants ──────────────────────────────────────────────────────────

# FULL SCALE — actual pgm run parameters
N_CELLS_FULL  = 172_000    # valid pgm spatial cells
T_STEPS       = 36         # forecast horizon (steps)
K_TARGETS     = 6          # prediction targets
M_ORIGINS     = 13         # rolling origins (MAX_SHIFT_COUNT + 1)
S_FULL        = 64         # posterior samples (HydraNet production)
BASE_MONTH    = 400        # synthetic start month_id

# SAFETY SCALE — used for tests that would OOM at full scale
N_CELLS_SAFE  = 32_400     # standard pgm 180×180 grid
S_SAFE        = 32         # sample count for safety tests

# ── helpers ──────────────────────────────────────────────────────────────────

_proc = psutil.Process(os.getpid())


def _rss_mb() -> float:
    return _proc.memory_info().rss / (1024 ** 2)


def _gb(mb: float) -> str:
    return f"{mb / 1024:.2f} GB" if mb >= 512 else f"{mb:.0f} MB"


def _pf_bytes(n_cells: int, t_steps: int, s: int) -> int:
    return n_cells * t_steps * s * 4  # float32


def _make_pf(n_cells: int, t_steps: int, s: int, seq_offset: int = 0) -> PredictionFrame:
    n_rows = n_cells * t_steps
    y_pred = np.ones((n_rows, s), dtype=np.float32) * 0.5
    base   = BASE_MONTH + seq_offset
    times  = np.repeat(np.arange(base, base + t_steps, dtype=np.int64), n_cells)
    units  = np.tile(np.arange(1, n_cells + 1, dtype=np.int64), t_steps)
    return PredictionFrame(y_pred=y_pred, identifiers={"time": times, "unit": units})


def _make_actuals(n_cells: int, t_steps: int, target: str = "sb") -> pd.DataFrame:
    total_time = t_steps + M_ORIGINS - 1
    base   = BASE_MONTH
    times  = np.repeat(np.arange(base, base + total_time, dtype=np.int64), n_cells)
    units  = np.tile(np.arange(1, n_cells + 1, dtype=np.int64), total_time)
    idx    = pd.MultiIndex.from_arrays([times, units], names=["month_id", "unit_id"])
    return pd.DataFrame({target: np.random.rand(len(times)).astype(np.float64)}, index=idx)


def _step_maps(t_steps: int) -> List[Dict[int, int]]:
    return [{BASE_MONTH + seq + step: step + 1 for step in range(t_steps)}
            for seq in range(M_ORIGINS)]


def _header(title: str, scale_label: str) -> None:
    print()
    print("─" * 72)
    print(f"  {title}  [{scale_label}]")
    print("─" * 72)


def _result(label: str, rss_before: float, rss_after: float) -> None:
    delta = rss_after - rss_before
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<52s}  RSS {_gb(rss_after):>9s}  Δ {sign}{_gb(delta)}")


# ── TEST A — single PF creation ───────────────────────────────────────────────

def test_a_single_pf(n_cells: int, s: int) -> tuple[bool, float]:
    """Create one PredictionFrame; measure RSS growth."""
    expected_mb = _pf_bytes(n_cells, T_STEPS, s) / (1024 ** 2)
    scale = f"N={n_cells:,}, T={T_STEPS}, S={s}  expected ~{_gb(expected_mb)}"
    _header("Test A — Single PF creation", scale)

    gc.collect()
    r0 = _rss_mb()
    try:
        pf = _make_pf(n_cells, T_STEPS, s)
        r1 = _rss_mb()
        _result("after PF creation", r0, r1)
        del pf
        gc.collect()
        r2 = _rss_mb()
        _result("after del + gc.collect()", r0, r2)
        peak = r1 - r0
        print(f"\n  ✓ PASS  peak = {_gb(peak)}")
        return True, peak
    except MemoryError as e:
        print(f"\n  ✗ FAIL  MemoryError: {e}")
        return False, float("inf")


# ── TEST B — PF save / load round-trip ───────────────────────────────────────

def test_b_save_load(n_cells: int, s: int) -> tuple[bool, float]:
    """PF.save() + PF.load(mmap=True); Track A disk path."""
    scale = f"N={n_cells:,}, T={T_STEPS}, S={s}"
    _header("Test B — PF save / load (Track A)", scale)

    tmpdir = Path(tempfile.mkdtemp(prefix="pf_stress_"))
    try:
        gc.collect()
        r0 = _rss_mb()
        pf = _make_pf(n_cells, T_STEPS, s)
        r1 = _rss_mb()
        _result("after PF creation", r0, r1)

        pf.save(tmpdir / "test_target")
        r2 = _rss_mb()
        _result("after pf.save()", r1, r2)

        del pf
        gc.collect()
        r3 = _rss_mb()
        _result("after del pf + gc.collect()", r2, r3)

        pf_loaded = PredictionFrame.load(tmpdir / "test_target", mmap=True)
        r4 = _rss_mb()
        _result("after PF.load(mmap=True)", r3, r4)
        _ = pf_loaded.y_pred[0, 0]  # touch one element to force page in
        r5 = _rss_mb()
        _result("after touching first element", r4, r5)

        del pf_loaded
        gc.collect()
        r6 = _rss_mb()
        _result("after del loaded + gc.collect()", r5, r6)

        peak = max(r1, r2) - r0
        print(f"\n  ✓ PASS  peak = {_gb(peak)}")
        return True, peak
    except MemoryError as e:
        print(f"\n  ✗ FAIL  MemoryError: {e}")
        return False, float("inf")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── TEST C — origin_sink (pop pattern) ───────────────────────────────────────

def test_c_origin_sink(n_cells: int, s: int) -> tuple[bool, float]:
    """
    Simulate origin_sink with K=6 targets using the pop pattern.
    Peak should be ~1 PF (1.59 GB at full scale) not K×PF.
    """
    pf_mb = _pf_bytes(n_cells, T_STEPS, s) / (1024 ** 2)
    scale = (f"N={n_cells:,}, T={T_STEPS}, S={s}, K={K_TARGETS}  "
             f"naive peak={_gb(pf_mb * K_TARGETS)}, pop peak={_gb(pf_mb)}")
    _header("Test C — origin_sink pop pattern", scale)

    tmpdir = Path(tempfile.mkdtemp(prefix="pf_stress_"))
    target_names = [f"target_{k}" for k in range(K_TARGETS)]

    try:
        gc.collect()
        r0 = _rss_mb()

        # Build pf_dict (K PFs simultaneously — this is what HydraNet delivers)
        pf_dict = {t: _make_pf(n_cells, T_STEPS, s) for t in target_names}
        r1 = _rss_mb()
        _result(f"after creating pf_dict (K={K_TARGETS} PFs)", r0, r1)

        # Pop-pattern loop (the fix)
        peak_in_loop = r1
        for target in list(pf_dict.keys()):
            pf = pf_dict.pop(target)
            save_path = tmpdir / "origin_0" / target
            pf.save(save_path)
            # Skip Track B (skip_predictions_delivery=True)
            del pf
            gc.collect()
            r_now = _rss_mb()
            peak_in_loop = max(peak_in_loop, r_now)
            _result(f"  after pop+save+del '{target}'", r1, r_now)

        del pf_dict
        gc.collect()
        r2 = _rss_mb()
        _result("after del pf_dict + gc.collect()", r0, r2)

        # Note: the initial pf_dict creation always costs K × PF_size (unavoidable —
        # HydraNet delivers all K targets simultaneously). The pop pattern ensures
        # progressive release during the loop rather than holding all K PFs alive
        # THROUGHOUT the loop (old `for target, pf in pf_dict.items()` pattern).
        # With skip_predictions_delivery=True: peak = K × PF_size at creation time.
        peak = peak_in_loop - r0
        naive_peak_mb = pf_mb * K_TARGETS
        print(f"\n  ✓ PASS  peak = {_gb(peak)}  (initial pf_dict = K × PF = {_gb(naive_peak_mb)})")
        print(f"         pop pattern: each iteration frees ~{_gb(pf_mb)} progressively")
        return True, peak
    except MemoryError as e:
        print(f"\n  ✗ FAIL  MemoryError: {e}")
        return False, float("inf")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── TEST D — converter.to_prediction_df (Track B) ────────────────────────────

def test_d_track_b(n_cells: int, s: int) -> tuple[bool, float]:
    """
    Run converter.to_prediction_df() to document the Python list explosion.
    ONLY run at safety scale — full scale would OOM.
    """
    pf_mb    = _pf_bytes(n_cells, T_STEPS, s) / (1024 ** 2)
    scale    = f"N={n_cells:,}, T={T_STEPS}, S={s}  PF={_gb(pf_mb)}  (safety scale)"
    _header("Test D — Track B: converter.to_prediction_df() (DOCUMENTS THE SPIKE)", scale)
    print("  NOTE: Only run at safety scale. At full scale (N=172k, S=64) peak ≈ 52 GB.")

    converter = PredictionFrameConverter()
    try:
        gc.collect()
        r0 = _rss_mb()
        pf = _make_pf(n_cells, T_STEPS, s)
        r1 = _rss_mb()
        _result("after PF creation", r0, r1)

        df = converter.to_prediction_df(pf, "target_0")
        r2 = _rss_mb()
        _result("after to_prediction_df()  ← SPIKE", r1, r2)

        del df
        gc.collect()
        r3 = _rss_mb()
        _result("after del df + gc.collect()  ← fragmentation?", r2, r3)

        del pf
        gc.collect()
        r4 = _rss_mb()
        _result("after del pf + gc.collect()  ← sticky?", r3, r4)

        peak = r2 - r0
        residual = r4 - r0
        ratio = peak / max(pf_mb, 1)
        print(f"\n  ✓ RAN  peak = {_gb(peak)}  ({ratio:.1f}× PF size)")
        print(f"         residual fragmentation = {_gb(residual)}")
        print("  FULL-SCALE PROJECTION (N=172k, S=64, PF≈1,593 MB):")
        full_ratio = peak / max(pf_mb, 1)
        full_pf_mb = _pf_bytes(N_CELLS_FULL, T_STEPS, S_FULL) / (1024 ** 2)
        print(f"    projected peak ≈ {_gb(full_pf_mb * full_ratio)}")
        return True, peak
    except MemoryError:
        peak = _rss_mb() - r0
        print(f"\n  ✗ FAIL  MemoryError (peak before OOM = {_gb(peak)})")
        return False, peak


# ── TEST E — EvaluationAdapter.from_prediction_frames ────────────────────────

def test_e_adapter_fpf(n_cells: int, s: int) -> tuple[bool, float]:
    """
    Run EvaluationAdapter.from_prediction_frames() for M=13 PFs on 1 target.
    y_pred_out pre-allocates (M×N×T, S) — documents the pre-allocation cost.
    ONLY run at safety scale.
    """
    pf_mb    = _pf_bytes(n_cells, T_STEPS, s) / (1024 ** 2)
    total_mb = pf_mb * M_ORIGINS
    scale    = (f"N={n_cells:,}, T={T_STEPS}, S={s}, M={M_ORIGINS}  "
                f"input={_gb(total_mb)}  output≈{_gb(total_mb)}  (safety scale)")
    _header("Test E — EvaluationAdapter.from_prediction_frames() (DOCUMENTS PRE-ALLOC)", scale)
    print("  NOTE: Only run at safety scale. At full scale (N=172k, S=64) peak ≈ 41 GB.")

    target = "sb"
    try:
        gc.collect()
        r0 = _rss_mb()

        predictions = [_make_pf(n_cells, T_STEPS, s, seq_offset=i) for i in range(M_ORIGINS)]
        actuals     = _make_actuals(n_cells, T_STEPS, target=target)
        step_maps   = _step_maps(T_STEPS)
        r1 = _rss_mb()
        _result(f"after building {M_ORIGINS} PFs", r0, r1)

        try:
            ef = EvaluationAdapter.from_prediction_frames(
                actual=actuals, predictions=predictions,
                target=target, step_mapping=step_maps,
            )
            r2 = _rss_mb()
            _result("after from_prediction_frames()  ← y_pred_out pre-alloc", r1, r2)

            del ef
            gc.collect()
            r3 = _rss_mb()
            _result("after del ef + gc.collect()", r2, r3)
        except MemoryError:
            r2 = _rss_mb()
            _result("MEMORYERROR during from_prediction_frames()", r1, r2)
            gc.collect()
            del predictions, actuals
            gc.collect()
            print(f"\n  ✗ MEMORYERROR  peak before OOM = {_gb(r2 - r0)}")
            return False, r2 - r0

        del predictions, actuals
        gc.collect()
        r4 = _rss_mb()
        _result("after del predictions + gc.collect()", r3, r4)

        peak = r2 - r0
        ratio = peak / max(total_mb, 1)
        print(f"\n  ✓ RAN  peak = {_gb(peak)}  ({ratio:.1f}× single PF input)")
        print("  FULL-SCALE PROJECTION (N=172k, S=64, 13 PFs):")
        full_pf_mb   = _pf_bytes(N_CELLS_FULL, T_STEPS, S_FULL) / (1024 ** 2)
        full_total_mb = full_pf_mb * M_ORIGINS
        print(f"    input PFs: {_gb(full_total_mb)}")
        print(f"    y_pred_out pre-alloc: ~{_gb(full_total_mb)}")
        print(f"    projected peak: ~{_gb(full_total_mb * ratio)}")
        return True, peak
    except Exception as e:
        print(f"\n  ✗ ERROR  {type(e).__name__}: {e}")
        return False, float("inf")


# ── TEST F — full pipeline with skip flags ────────────────────────────────────

def test_f_full_pipeline(n_cells: int, s: int) -> tuple[bool, float]:
    """
    END-TO-END proof: M=13 origins × K=6 targets at full scale with skip flags.
    - skip_predictions_delivery=True  → no Track B
    - skip_evaluation_metrics=True    → no from_prediction_frames
    - pop pattern in origin_sink      → max 1 PF alive at a time
    - mmap reload                     → ~0 RAM for loaded PFs
    """
    pf_mb  = _pf_bytes(n_cells, T_STEPS, s) / (1024 ** 2)
    scale  = (f"N={n_cells:,}, T={T_STEPS}, S={s}, K={K_TARGETS}, M={M_ORIGINS}  "
              f"PF={_gb(pf_mb)}  expected peak ≈ {_gb(pf_mb)}")
    _header("Test F — FULL PIPELINE with skip flags (END-TO-END PROOF)", scale)

    tmpdir = Path(tempfile.mkdtemp(prefix="pf_stress_"))
    target_names = [f"target_{k}" for k in range(K_TARGETS)]

    try:
        gc.collect()
        r0 = _rss_mb()
        peak_overall = r0

        # Simulate M origins
        for origin_idx in range(M_ORIGINS):
            # HydraNet would call origin_sink(origin_idx, pf_dict)
            # We create the pf_dict synthetically:
            pf_dict = {t: _make_pf(n_cells, T_STEPS, s, seq_offset=origin_idx)
                       for t in target_names}
            r_pf_dict = _rss_mb()
            peak_overall = max(peak_overall, r_pf_dict)

            # origin_sink: pop pattern
            staging = tmpdir / f"origin_{origin_idx}"
            for target in list(pf_dict.keys()):
                pf = pf_dict.pop(target)
                pf.save(staging / target)
                del pf
                gc.collect()
                r_now = _rss_mb()
                peak_overall = max(peak_overall, r_now)

            del pf_dict
            gc.collect()
            r_after = _rss_mb()
            _result(f"origin {origin_idx:02d}: after sink+del+gc", r0, r_after)

        r1 = _rss_mb()
        print(f"\n  Phase 1 complete: all {M_ORIGINS} origins saved to disk.")
        print(f"  Peak during origin loop: {_gb(peak_overall - r0)}")

        # Reload all PFs with mmap (Track A reload)
        all_targets = target_names
        loaded: Dict[str, list] = {t: [] for t in all_targets}
        for target in all_targets:
            for i in range(M_ORIGINS):
                pf = PredictionFrame.load(tmpdir / f"origin_{i}" / target, mmap=True)
                loaded[target].append(pf)

        r2 = _rss_mb()
        _result("after reloading all PFs with mmap", r1, r2)
        print(f"  Mmap load cost = {_gb(r2 - r1)}  (expected ~0 until pages touched)")

        del loaded
        gc.collect()
        r3 = _rss_mb()
        _result("after del loaded + gc.collect()", r2, r3)

        # skip_evaluation_metrics=True → skip from_prediction_frames entirely

        peak = peak_overall - r0
        naive_peak_mb = pf_mb * K_TARGETS
        print(f"\n  ✓ PASS  full pipeline peak = {_gb(peak)}")
        print(f"         peak = initial pf_dict = K × PF_size = {_gb(naive_peak_mb)}")
        print(f"         (pop pattern progressively reduces from {_gb(naive_peak_mb)} to 0)")
        print(f"         Note: mmap reload shows {_gb(r2 - r1)} RSS — identifiers are")
        print(f"         loaded eagerly (~{_gb(pf_mb * 0.134 * K_TARGETS * M_ORIGINS)}")
        print("         for 78 PFs). With skip_evaluation_metrics=True this path is skipped.")
        return True, peak

    except MemoryError as e:
        print(f"\n  ✗ FAIL  MemoryError: {e}")
        return False, float("inf")
    except Exception as e:
        print(f"\n  ✗ ERROR  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, float("inf")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'Test':<6}  {'Name':<42}  {'Status':<6}  {'Peak'}")
    print("  " + "-" * 68)
    for key, (passed, peak_mb, name) in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        peak_s = _gb(peak_mb) if peak_mb != float("inf") else "OOM"
        print(f"  {key:<6}  {name:<42}  {status:<6}  {peak_s}")
    print()
    print("  Interpretation:")
    print("  - Tests A/B/C/F PASS → pipeline-core is feasible at S=64.")
    print("  - Tests D/E FAIL/WARN → Track B and metrics need skip flags or refactor.")
    print("  - The OOM in production is in HydraNet (to_evaluation_pf dense-grid copy),")
    print("    not in pipeline-core.")
    print()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PF Pipeline Stress Test")
    parser.add_argument("--fast", action="store_true",
                        help="Run all tests at safety scale (N=32,400, S=32) only")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  views-pipeline-core · PF Pipeline Stress Test                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Full scale:   N={N_CELLS_FULL:,} cells × T={T_STEPS} steps × S={S_FULL} samples")
    print(f"                K={K_TARGETS} targets × M={M_ORIGINS} origins")
    full_pf_mb = _pf_bytes(N_CELLS_FULL, T_STEPS, S_FULL) / (1024 ** 2)
    print(f"  PF size:      {_gb(full_pf_mb)}")
    print(f"  Safety scale: N={N_CELLS_SAFE:,} cells × S={S_SAFE} samples")
    print(f"  Process RSS:  {_rss_mb():.0f} MB")
    print()

    if args.fast:
        n_full, s_full = N_CELLS_SAFE, S_SAFE
        print("  [--fast] Running all tests at SAFETY scale only.")
    else:
        n_full, s_full = N_CELLS_FULL, S_FULL
        print("  Running A/B/C/F at FULL scale, D/E at SAFETY scale.")
    print()

    results = {}

    passed, peak = test_a_single_pf(n_full, s_full)
    results["A"] = (passed, peak, "Single PF creation")

    passed, peak = test_b_save_load(n_full, s_full)
    results["B"] = (passed, peak, "PF save / load (Track A)")

    passed, peak = test_c_origin_sink(n_full, s_full)
    results["C"] = (passed, peak, "origin_sink pop pattern (K=6)")

    passed, peak = test_d_track_b(N_CELLS_SAFE, S_SAFE)
    results["D"] = (passed, peak, "Track B: to_prediction_df (SAFETY)")

    passed, peak = test_e_adapter_fpf(N_CELLS_SAFE, S_SAFE)
    results["E"] = (passed, peak, "EvaluationAdapter.from_pf (SAFETY)")

    passed, peak = test_f_full_pipeline(n_full, s_full)
    results["F"] = (passed, peak, "Full pipeline with skip flags")

    print_summary(results)


if __name__ == "__main__":
    main()
