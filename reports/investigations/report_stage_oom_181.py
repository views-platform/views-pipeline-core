"""Synthetic micro-benchmark for GitHub issue #181 — line-isolate the report-stage
host-RAM amplification chain WITHOUT a GPU/model/full repro.

#181: `main.py -r calibration -t -e -re` OOM-killed (~16-20 GB) in the post-eval
report tail; drops ~7x without `-re`; scales with n_posterior_samples (S). Model
side ruled out. Suspected amplifications, each an object-dtype / dense blow-up:

  FORECAST side (carries S):
    A  PF.y_pred (N,S) float32  --to_prediction_df-->  list-in-cell object DataFrame
    B  PGMDataset(pred_df)       densify + per-cell np.array(float32)
    C  to_tensor()              float64 tensor (+ np.stack second copy), then MAP sort
  HISTORICAL side (S=1, but ~324 calibration months x 32_400 cells ~= 10.5M cells):
    H  read_dataframe(raw) scalar df --PGMDataset(scalar_df, broadcast_features=False)-->
       every scalar cell becomes a size-1 np.array (object dtype)

This box has limited RAM (~6 GB free), so we MEASURE at reduced scale (cells=8100,
quarter-PGM) and EXTRAPOLATE per-row to full PGM forecast (32_400 x 36) and
historical (32_400 x 324). tracemalloc captures Python-heap (object cells / lists);
RSS captures numpy tensors. calculate_map is intentionally NOT run (it is a sort
over C's tensor -> +~1x transient; reasoned, not measured, to keep runtime bounded).

Run: conda run -n views_pipeline python -u reports/investigations/report_stage_oom_181.py
"""
from __future__ import annotations

import gc
import os
import tracemalloc

import numpy as np
import pandas as pd
import psutil

from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.modules.frames.prediction_frame_converter import (
    PredictionFrameConverter,
)
from views_pipeline_core.data.handlers import PGMDataset

_proc = psutil.Process(os.getpid())

PGM_CELLS = 180 * 180          # 32_400
FORECAST_N = PGM_CELLS * 36    # forecast volume rows
HIST_N = PGM_CELLS * 324       # full calibration historical rows (~10.5M)

CELLS = 8100                   # quarter-PGM measured scale (linear-extrapolates)
MONTHS = 12


def rss_mb() -> float:
    return _proc.memory_info().rss / (1024 ** 2)


def _grid(n_units: int, months: int):
    units = np.arange(n_units, dtype=np.int64)
    times = np.arange(1, months + 1, dtype=np.int64)
    return np.repeat(times, n_units), np.tile(units, months), n_units * months


def _stage(label, fn):
    gc.collect()
    tracemalloc.start()
    r0 = rss_mb()
    out = fn()
    heap_peak = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
    tracemalloc.stop()
    gc.collect()
    r1 = rss_mb()
    print(f"    {label:<38} RSS+={r1 - r0:7.1f}MB  heap-peak={heap_peak:7.1f}MB  RSS={r1:7.0f}MB",
          flush=True)
    return out, max(r1 - r0, 0.0), heap_peak


def forecast_chain(S: int):
    print(f"\n  [FORECAST] cells={CELLS} months={MONTHS} S={S}  N={CELLS*MONTHS:,}", flush=True)
    t, u, n = _grid(CELLS, MONTHS)
    y = np.random.default_rng(0).random((n, S), dtype=np.float32)
    pf = PredictionFrame(y_pred=y, identifiers={"time": t, "unit": u})
    print(f"    dense y_pred (N,S) float32 = {y.nbytes/1024**2:7.1f}MB"
          f"  ({y.nbytes/n:.1f} B/row)", flush=True)
    conv = PredictionFrameConverter()

    df, _, a_heap = _stage("A to_prediction_df (list-in-cell)", lambda: conv.to_prediction_df(pf, "t0"))
    df.index = df.index.set_names(["month_id", "priogrid_id"])
    ds, b_rss, b_heap = _stage("B PGMDataset(pred) densify+arrays", lambda: PGMDataset(df))
    tensor, c_rss, _ = _stage("C to_tensor (float64)", ds.to_tensor)
    if isinstance(tensor, np.ndarray):
        print(f"      tensor {tensor.shape} {tensor.dtype} = {tensor.nbytes/1024**2:.1f}MB"
              f"  (float32 -> {tensor.nbytes/2/1024**2:.1f}MB)", flush=True)
    res = {"S": S, "n": n,
           "A_listcell_Bpr": a_heap * 1024**2 / n,
           "B_obj_Bpr": max(b_heap, b_rss) * 1024**2 / n,
           "C_tensor_Bpr": c_rss * 1024**2 / n}
    gc.collect()
    return res


def historical_object_ification():
    """Scalar (actuals) df -> PGMDataset(broadcast_features=False): per-cell np.array bloat."""
    print(f"\n  [HISTORICAL] scalar actuals, cells={CELLS} months={MONTHS}  N={CELLS*MONTHS:,}", flush=True)
    t, u, n = _grid(CELLS, MONTHS)
    idx = pd.MultiIndex.from_arrays([t, u], names=["month_id", "priogrid_id"])
    sdf = pd.DataFrame({"sb_best": np.random.default_rng(1).random(n).astype(np.float64)}, index=idx)
    print(f"    dense scalar df = {sdf.memory_usage(deep=True).sum()/1024**2:7.1f}MB"
          f"  ({sdf.memory_usage(deep=True).sum()/n:.1f} B/row)", flush=True)
    ds, h_rss, h_heap = _stage("H PGMDataset(scalar) obj-ify+densify",
                               lambda: PGMDataset(sdf, targets=["sb_best"]))
    res = {"H_obj_Bpr": max(h_heap, h_rss) * 1024**2 / n}
    gc.collect()
    return res


def main():
    print(f"RAM available: {psutil.virtual_memory().available/1024**3:.1f}GB  RSS={rss_mb():.0f}MB",
          flush=True)
    fc = [forecast_chain(S) for S in (1, 3, 8)]
    hist = historical_object_ification()

    print("\n" + "=" * 86)
    print("PER-ROW BYTES (measured @ quarter-PGM)  ->  EXTRAPOLATED to full production scale")
    print("=" * 86)
    print(f"FORECAST volume  N = 32400 x 36  = {FORECAST_N:,} rows")
    print(f"HISTORICAL volume N = 32400 x 324 = {HIST_N:,} rows (full calibration span)\n")
    print(f"{'stage (1 target)':<34}{'B/row':>9}{'forecast GB':>14}{'historical GB':>16}")
    for r in fc:
        for k, name in (("A_listcell_Bpr", f"A list-in-cell  S={r['S']}"),
                        ("B_obj_Bpr",      f"B densify+arr  S={r['S']}"),
                        ("C_tensor_Bpr",   f"C tensor f64   S={r['S']}")):
            bpr = r[k]
            print(f"{name:<34}{bpr:>9.0f}{bpr*FORECAST_N/1024**3:>14.2f}{bpr*HIST_N/1024**3:>16.2f}")
    bpr = hist["H_obj_Bpr"]
    print(f"{'H hist obj-ify (S=1)':<34}{bpr:>9.0f}{bpr*FORECAST_N/1024**3:>14.2f}{bpr*HIST_N/1024**3:>16.2f}")
    print("\nNotes: forecast stages carry S (scale with n_posterior_samples); H is S-independent")
    print("and applies to the ~10.5M-row historical frame -> the S-independent base.")
    print("calculate_map adds ~1x transient over C (np.sort copy). Full repro stays on the 32GB box.")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
