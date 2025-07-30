#!/usr/bin/env python3
"""
FPGA‑based MNIST Inference (INT16 Q6.10) — **Batched mode + idle power capture**
--------------------------------------------------------------------------
* Measures idle INA260 baseline (3s) and saves it to `idle_power.npy`.
* Then runs batched inference with a fresh logger and saves
  `power_trace.npy` + `power_bounds.npy` for validation.
"""
from __future__ import annotations

import argparse, os, time
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

from axi_stream_driver import NeuralNetworkOverlay
from mnist_utils          import load_and_quantize_mnist
from utils.power_monitor  import PowerMonitor

OUTPUT_DIM = 10
IDLE_SECONDS = 10.0          # record 10 s of idle samples
SAMPLE_DT   = 0.001          # INA260 sampling interval (10 ms)

# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-b", "--bitstream", required=True, help="Compiled .bit file path")
    p.add_argument("-m", "--metrics-dir", default="metrics/", help="Where to save .npy outputs")
    p.add_argument("--batch-size", type=int, default=100, help="Batch size for inference")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    return p.parse_args()

# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------

def allocate_overlay(bitstream: str, feat_dim: int) -> NeuralNetworkOverlay:
    return NeuralNetworkOverlay(bitstream,
                                x_shape=(feat_dim,),
                                y_shape=(OUTPUT_DIM,),
                                dtype=np.int16)

def run_batched(nn: NeuralNetworkOverlay, X: np.ndarray, batch: int, quiet: bool):
    N = X.shape[0]
    y_pred = np.empty((N, OUTPUT_DIM), np.float32)
    lat_c, lat_i = np.empty(N), np.empty(N)
    bounds: list[Tuple[float,float]] = []

    rng = tqdm(range(0, N, batch), disable=quiet, desc="Inference", ncols=90)
    for st in rng:
        ed = min(st+batch, N)
        y_b, lc, li, bnd = nn.predict_batch(X[st:ed], profile=True)
        y_pred[st:ed] = y_b; lat_c[st:ed] = lc; lat_i[st:ed] = li; bounds.extend(bnd)

    thr_c = 1/lat_c; thr_i = 1/lat_i
    return y_pred, lat_c, thr_c, lat_i, thr_i, bounds

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    args = parse_args()
    mdir = Path(args.metrics_dir); mdir.mkdir(parents=True, exist_ok=True)

    # 0. Idle baseline ---------------------------------------------------
    print(f"0. Measuring idle baseline for {IDLE_SECONDS}s …")
    idle_mon = PowerMonitor(interval=SAMPLE_DT)
    idle_mon.start(); time.sleep(IDLE_SECONDS); idle_mon.stop()
    idle_trace = np.array([p for _, p in idle_mon.get_trace()], dtype=float)
    np.save(mdir/"idle_power.npy", idle_trace)
    print(f"   Saved {len(idle_trace)} idle samples → {mdir/'idle_power.npy'}")

    # 1. Dataset ---------------------------------------------------------
    print("1. Loading and quantizing MNIST test set…")
    X_i16, y_int = load_and_quantize_mnist(); N, F = X_i16.shape
    print(f"   Samples: {N}  Features: {F}")

    # 2. FPGA bitstream --------------------------------------------------
    print("2. Programming FPGA bitstream…")
    nn = allocate_overlay(args.bitstream, F)

    # 3. Inference with power logging -----------------------------------
    print("3. Starting batched inference with power monitoring…")
    run_mon = PowerMonitor(interval=SAMPLE_DT)
    run_mon.start()
    y_hw_f32, lat_c, thr_c, lat_i, thr_i, bounds = run_batched(
        nn, X_i16, args.batch_size, args.no_progress)
    run_mon.stop()
    run_trace = run_mon.get_trace()
    print(f"   Collected {len(run_trace)} INA260 samples during inference")

    # 4. Accuracy --------------------------------------------------------
    acc = (y_hw_f32.argmax(1) == y_int).mean()*100
    print(f"4. Accuracy  : {acc:.2f}%")

    # 5. Save metrics ----------------------------------------------------
    np.save(mdir/"y_hw.npy",          y_hw_f32)
    np.save(mdir/"latency1.npy",      lat_c)
    np.save(mdir/"throughput1.npy",   thr_c)
    np.save(mdir/"latency2.npy",      lat_i)
    np.save(mdir/"throughput2.npy",   thr_i)
    np.save(mdir/"power_trace.npy",   np.array(run_trace, dtype=object))
    np.save(mdir/"power_bounds.npy",  np.array(bounds,     dtype=object))

    print("5. Metrics saved →", mdir)

if __name__ == "__main__":
    main()
