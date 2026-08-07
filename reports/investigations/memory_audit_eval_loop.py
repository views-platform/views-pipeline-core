"""
Memory Audit: Evaluation Loop Under the PredictionFrame Path
============================================================

Self-contained script — no HydraNet, no viewser data, no WandB.
Uses real pipeline-core classes (PredictionFrame, from_prediction_frames,
to_prediction_df) and synthetic data of the correct scale.

Usage:
    conda run -n views_pipeline python reports/investigations/memory_audit_eval_loop.py

    Optional: adjust the PARAMS block below to test different scales.

Hypotheses tested:
    H1  PFs accumulate linearly during inference (~150 MB/seq/target at 32 samples)
    H2  Large transient allocations (simulating inference tensors) don't fully release
    H3  from_prediction_frames() is the largest single peak (input + output coexist)
    H4  Un-freed input data (X) compounds the accumulation
    H5  Python allocator fragmentation: RSS stays high after gc.collect()
    H6  to_prediction_df() Python list explosion creates a ~7× transient spike
"""

from __future__ import annotations

import gc
import os
import sys
import tracemalloc
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psutil

# ─── Add repo root to path ───────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from views_pipeline_core.data.prediction_frame import PredictionFrame  # noqa: E402
from views_pipeline_core.modules.frames.prediction_frame_converter import (  # noqa: E402
    PredictionFrameConverter,
)
from views_pipeline_core.modules.validation.adapter import EvaluationAdapter  # noqa: E402

# ─── PARAMETERS ───────────────────────────────────────────────────────────────
# Adjust these to test different scales.
N_SEQUENCES  = 13        # rolling-origin sequences (MAX_SHIFT_COUNT + 1)
N_TARGETS    = 2         # number of prediction targets
N_CELLS      = 32_400    # pgm 180×180 grid cells
N_TIME_STEPS = 36        # forecast horizon
N_SAMPLES    = 32        # posterior samples (current HydraNet)
N_FEATURES   = 20        # approximate viewser feature count per cell per month
INFERENCE_MULTIPLIER = 5 # PyTorch activations ≈ N× output (for H2 simulation)

# ─── Helpers ──────────────────────────────────────────────────────────────────

_proc = psutil.Process(os.getpid())


def rss_mb() -> float:
    """OS-reported resident set size in MB."""
    return _proc.memory_info().rss / (1024 ** 2)


def heap_mb() -> float:
    """Python-heap current size in MB (tracemalloc)."""
    current, _ = tracemalloc.get_traced_memory()
    return current / (1024 ** 2)


class _Probe:
    """Lightweight measurement probe."""

    def __init__(self) -> None:
        self._last_rss = rss_mb()

    def snap(self, label: str) -> None:
        rss  = rss_mb()
        heap = heap_mb()
        delta = rss - self._last_rss
        sign  = "+" if delta >= 0 else ""
        print(
            f"  [{label:<50s}]  RSS {rss:7.1f} MB  heap {heap:7.1f} MB  Δ {sign}{delta:+.1f} MB"
        )
        self._last_rss = rss


def _separator(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def _pf_size_mb(n_cells: int = N_CELLS, n_time: int = N_TIME_STEPS,
                n_samples: int = N_SAMPLES) -> float:
    """Expected float32 PF size in MB: (n_cells × n_time, n_samples) × 4 B."""
    return n_cells * n_time * n_samples * 4 / (1024 ** 2)


def _make_pf(n_cells: int = N_CELLS, n_time: int = N_TIME_STEPS,
             n_samples: int = N_SAMPLES, seq_offset: int = 0) -> PredictionFrame:
    """Build a synthetic PredictionFrame at the correct scale."""
    n_rows = n_cells * n_time
    # float32 enforced by PredictionFrame.__init__ (Phase 1)
    y_pred = np.ones((n_rows, n_samples), dtype=np.float32) * 0.5
    base_month = 400 + seq_offset
    times = np.repeat(np.arange(base_month, base_month + n_time, dtype=np.int64), n_cells)
    units = np.tile(np.arange(1, n_cells + 1, dtype=np.int64), n_time)
    return PredictionFrame(
        y_pred=y_pred,
        identifiers={"time": times, "unit": units},
    )


def _make_actuals(n_cells: int = N_CELLS, n_time: int = N_TIME_STEPS,
                  target: str = "sb", seq_offset: int = 0) -> pd.DataFrame:
    """Build synthetic actuals DataFrame covering all sequences."""
    # Must cover all sequences (base + MAX_SHIFT_COUNT)
    total_time = n_time + N_SEQUENCES - 1
    base_month = 400 + seq_offset
    times = np.repeat(np.arange(base_month, base_month + total_time, dtype=np.int64), n_cells)
    units = np.tile(np.arange(1, n_cells + 1, dtype=np.int64), total_time)
    idx = pd.MultiIndex.from_arrays([times, units], names=["month_id", "unit_id"])
    return pd.DataFrame(
        {target: np.random.rand(len(times)).astype(np.float64)},
        index=idx,
    )


def _step_mapping(seq_idx: int, n_time: int = N_TIME_STEPS) -> Dict[int, int]:
    base_month = 400 + seq_idx
    return {base_month + step: step + 1 for step in range(n_time)}


# ─── EXPERIMENT 1: PF Accumulation (H1) ──────────────────────────────────────

def experiment_1_accumulation() -> None:
    _separator("EXPERIMENT 1 — PF accumulation during inference loop (H1)")
    print(
        f"  Params: {N_SEQUENCES} sequences × {N_TARGETS} targets × "
        f"{N_CELLS} cells × {N_TIME_STEPS} steps × {N_SAMPLES} samples"
    )
    expected_per_seq = _pf_size_mb() * N_TARGETS
    print(f"  Expected RSS growth per sequence: ~{expected_per_seq:.0f} MB")
    print()

    gc.collect()
    probe = _Probe()
    probe.snap("BASELINE before loop")

    raw_preds: Dict[str, List[Any]] = {f"target_{t}": [] for t in range(N_TARGETS)}

    for seq in range(N_SEQUENCES):
        for t_idx in range(N_TARGETS):
            pf = _make_pf(seq_offset=seq)
            raw_preds[f"target_{t_idx}"].append(pf)
        probe.snap(f"after seq {seq:02d} append  (expected Δ ≈ +{expected_per_seq:.0f} MB)")

    probe.snap("FINAL after all sequences accumulated")
    print(
        f"\n  H1 verdict: expected total = {N_SEQUENCES * expected_per_seq:.0f} MB  "
        f"(see Δ column above)"
    )

    # Clean up
    del raw_preds
    gc.collect()
    probe.snap("after del + gc.collect()")


# ─── EXPERIMENT 2: Inference tensor release (H2) ─────────────────────────────

def experiment_2_inference_tensor_release() -> None:
    _separator("EXPERIMENT 2 — Inference tensor non-release between sequences (H2)")
    pf_mb    = _pf_size_mb()
    total_mb = pf_mb * INFERENCE_MULTIPLIER
    print(
        f"  Simulating inference output {pf_mb:.0f} MB + activations ×{INFERENCE_MULTIPLIER}"
        f" = {total_mb:.0f} MB per sequence"
    )
    print("  Testing: does RSS recover to pre-inference baseline after each del?")
    print()

    gc.collect()
    probe = _Probe()
    probe.snap("BASELINE")

    raw_preds: Dict[str, List[Any]] = {f"target_{t}": [] for t in range(N_TARGETS)}
    baselines = []

    for seq in range(N_SEQUENCES):
        b = rss_mb()
        baselines.append(b)

        # Simulate PyTorch tensor (output + activations)
        n_rows = N_CELLS * N_TIME_STEPS
        torch_sim = np.ones(
            (n_rows * INFERENCE_MULTIPLIER, N_SAMPLES), dtype=np.float32
        )
        probe.snap(f"seq {seq:02d}: AFTER inference alloc     (expected +{total_mb:.0f} MB)")

        # Simulate conversion: take slice for each target
        for t_idx in range(N_TARGETS):
            pf = PredictionFrame(
                y_pred=torch_sim[:n_rows].copy(),
                identifiers={
                    "time": np.repeat(
                        np.arange(400 + seq, 400 + seq + N_TIME_STEPS, dtype=np.int64),
                        N_CELLS,
                    ),
                    "unit": np.tile(np.arange(1, N_CELLS + 1, dtype=np.int64), N_TIME_STEPS),
                },
            )
            raw_preds[f"target_{t_idx}"].append(pf)

        del torch_sim
        probe.snap(
            f"seq {seq:02d}: AFTER del torch_sim, no gc  "
            f"(recovered? baseline was {b:.0f} MB)"
        )
        gc.collect()
        probe.snap(f"seq {seq:02d}: AFTER gc.collect()")

    print(
        "\n  H2 check: if RSS does NOT return to baseline after each 'del + gc', "
        "memory is leaking/caching."
    )
    del raw_preds
    gc.collect()
    probe.snap("after del raw_preds + gc.collect()")


# ─── EXPERIMENT 3: from_prediction_frames peak (H3) ──────────────────────────

def experiment_3_from_prediction_frames_peak() -> None:
    _separator("EXPERIMENT 3 — from_prediction_frames() peak (H3)")
    target     = "sb"
    pf_mb      = _pf_size_mb()
    input_mb   = pf_mb * N_SEQUENCES
    output_mb  = pf_mb * N_SEQUENCES  # EF y_pred same size as all PF inputs combined
    print(
        f"  Input (13 PFs):   ~{input_mb:.0f} MB"
        f"  |  Output (EF):   ~{output_mb:.0f} MB"
        f"  |  Expected peak: ~{input_mb + output_mb:.0f} MB"
    )
    print()

    gc.collect()
    probe = _Probe()

    # Build the full raw_preds dict (all 13 sequences)
    predictions = [_make_pf(seq_offset=i) for i in range(N_SEQUENCES)]
    actuals     = _make_actuals(target=target)
    step_maps   = [_step_mapping(i) for i in range(N_SEQUENCES)]

    probe.snap(f"AFTER building raw_preds (13 PFs)   (expected +{input_mb:.0f} MB)")

    ef = EvaluationAdapter.from_prediction_frames(
        actual=actuals,
        predictions=predictions,
        target=target,
        step_mapping=step_maps,
    )
    probe.snap(
        f"AFTER from_prediction_frames()      "
        f"(input {input_mb:.0f} + output {output_mb:.0f} = {input_mb + output_mb:.0f} MB coexist)"
    )

    del predictions
    gc.collect()
    probe.snap("AFTER del predictions + gc.collect()  (input freed, EF remains)")

    del ef
    gc.collect()
    probe.snap("AFTER del ef + gc.collect()            (EF freed)")

    print(
        f"\n  H3 check: peak between 'after from_prediction_frames' and "
        f"'after del predictions' should be ~{input_mb + output_mb:.0f} MB above baseline."
    )
    del actuals


# ─── EXPERIMENT 4: Un-freed input data (H4) ──────────────────────────────────

def experiment_4_input_data_accumulation() -> None:
    _separator("EXPERIMENT 4 — Input data (X) accumulation: freed vs un-freed (H4)")
    x_mb = N_CELLS * N_TIME_STEPS * N_FEATURES * 8 / (1024 ** 2)  # float64 viewser features
    print(
        f"  Synthetic X: {N_CELLS} cells × {N_TIME_STEPS} months × {N_FEATURES} features"
        f" × 8 B = {x_mb:.0f} MB per sequence"
    )
    print(f"  13 sequences WITHOUT del X: expected +{13 * x_mb:.0f} MB accumulation")
    print()

    gc.collect()
    probe = _Probe()
    probe.snap("BASELINE")

    # --- 4A: WITHOUT del X (simulating model repo that doesn't clean up) ---
    print("  4A: WITHOUT del X between sequences:")
    held_xs = []  # intentionally not freeing
    raw_preds: Dict[str, List[Any]] = {f"target_{t}": [] for t in range(N_TARGETS)}
    for seq in range(N_SEQUENCES):
        X = np.ones((N_CELLS * N_TIME_STEPS, N_FEATURES), dtype=np.float64)
        held_xs.append(X)
        for t_idx in range(N_TARGETS):
            pf = _make_pf(seq_offset=seq)
            raw_preds[f"target_{t_idx}"].append(pf)
        probe.snap(
            f"  seq {seq:02d}: PF + X held  "
            f"(expected Δ ≈ +{_pf_size_mb() * N_TARGETS + x_mb:.0f} MB)"
        )

    probe.snap(f"AFTER 13 seqs WITHOUT del X  (total X: {13 * x_mb:.0f} MB extra)")
    del held_xs, raw_preds
    gc.collect()
    probe.snap("cleaned up 4A")

    print()
    print("  4B: WITH del X between sequences:")
    raw_preds2: Dict[str, List[Any]] = {f"target_{t}": [] for t in range(N_TARGETS)}
    for seq in range(N_SEQUENCES):
        X = np.ones((N_CELLS * N_TIME_STEPS, N_FEATURES), dtype=np.float64)
        for t_idx in range(N_TARGETS):
            pf = _make_pf(seq_offset=seq)
            raw_preds2[f"target_{t_idx}"].append(pf)
        del X
        gc.collect()
        probe.snap(
            f"  seq {seq:02d}: PF kept, X freed  "
            f"(expected Δ ≈ +{_pf_size_mb() * N_TARGETS:.0f} MB)"
        )

    probe.snap("AFTER 13 seqs WITH del X  (only PFs remain)")
    del raw_preds2
    gc.collect()
    probe.snap("cleaned up 4B")

    print(
        f"\n  H4 check: compare 4A vs 4B total Δ. Difference = X accumulation = "
        f"~{13 * x_mb:.0f} MB if H4 is true."
    )


# ─── EXPERIMENT 5: Save-loop list explosion (H6) ─────────────────────────────

def experiment_5_save_loop_transient() -> None:
    _separator("EXPERIMENT 5 — to_prediction_df() Python list explosion (H6)")
    pf_mb       = _pf_size_mb()
    expected_mb = pf_mb * 7  # float32 → Python float: ~7× overhead
    print(
        f"  PF size: {pf_mb:.0f} MB (float32)"
        f"  |  Expected list expansion: ~7× = {expected_mb:.0f} MB"
    )
    print()

    gc.collect()
    probe = _Probe()
    probe.snap("BASELINE")

    pf = _make_pf()
    probe.snap(f"AFTER creating 1 PF                 (expected +{pf_mb:.0f} MB)")

    converter = PredictionFrameConverter()
    df_for_save = converter.to_prediction_df(pf, "sb")
    probe.snap(
        f"AFTER to_prediction_df()            "
        f"(Python list expansion; expected peak +{expected_mb:.0f} MB above PF baseline)"
    )

    del df_for_save
    gc.collect()
    probe.snap("AFTER del df_for_save + gc.collect() (transient freed?)")

    del pf
    gc.collect()
    probe.snap("AFTER del pf + gc.collect()          (back to baseline?)")

    print(
        f"\n  H6 check: peak at 'after to_prediction_df' should be ~{expected_mb:.0f} MB"
        f" above PF baseline. After del+gc it should recover."
    )


# ─── SCALING PROJECTION TABLE ─────────────────────────────────────────────────

def print_scaling_table() -> None:
    _separator("SCALING PROJECTION TABLE (theoretical, float32)")
    print()
    scenarios = [
        ("current (32 samp, pgm)",    32,  32_400,  N_TARGETS, N_SEQUENCES, N_TIME_STEPS),
        ("252 samp, pgm",            252,  32_400,  N_TARGETS, N_SEQUENCES, N_TIME_STEPS),
        ("32 samp, global grid",      32, 259_200,  N_TARGETS, N_SEQUENCES, N_TIME_STEPS),
        ("252 samp, global grid",    252, 259_200,  N_TARGETS, N_SEQUENCES, N_TIME_STEPS),
    ]

    def _gb(mb: float) -> str:
        return f"{mb / 1024:.1f} GB"

    print(
        f"  {'Scenario':<30s}  {'PF size':>8s}  {'raw_preds':>10s}  "
        f"{'EF peak/tgt':>12s}  {'save transient':>15s}  {'est total peak':>14s}"
    )
    print("  " + "-" * 95)
    for name, n_s, n_c, n_t, n_seq, n_step in scenarios:
        pf_mb       = n_c * n_step * n_s * 4 / (1024 ** 2)
        raw_gb      = pf_mb * n_seq * n_t / 1024
        ef_peak_gb  = (pf_mb * n_seq + pf_mb * n_seq) / 1024  # input + output per target
        save_trans  = pf_mb * 7 / 1024
        total_est   = raw_gb + ef_peak_gb + save_trans
        print(
            f"  {name:<30s}  {_gb(pf_mb):>8s}  {raw_gb:>9.1f}G  "
            f"{ef_peak_gb:>11.1f}G  {save_trans:>14.1f}G  {total_est:>13.1f}G"
        )
    print()
    print(
        "  Note: 'est total peak' = raw_preds (all seqs) + EF construction peak (1 target)\n"
        "  + save transient (1 PF). HydraNet inference tensors NOT included."
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║       views-pipeline-core  ·  Memory Audit: Evaluation Loop             ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Scale: {N_SEQUENCES} seqs × {N_TARGETS} targets × {N_CELLS:,} cells")
    print(f"         × {N_TIME_STEPS} steps × {N_SAMPLES} samples  (float32)")
    pf_mb = _pf_size_mb()
    print(f"  PF size at this scale: {pf_mb:.1f} MB")
    print(f"  Process RSS at start:  {rss_mb():.1f} MB")
    print()

    tracemalloc.start()

    experiment_1_accumulation()
    experiment_2_inference_tensor_release()
    experiment_3_from_prediction_frames_peak()
    experiment_4_input_data_accumulation()
    experiment_5_save_loop_transient()
    print_scaling_table()

    tracemalloc.stop()
    print()
    print("  Audit complete. Interpret results against hypotheses H1–H6 in plan file.")
    print()


if __name__ == "__main__":
    main()
