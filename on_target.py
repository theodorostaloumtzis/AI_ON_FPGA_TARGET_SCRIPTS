#!/usr/bin/env python3
"""
FPGA inference runner (INT16 Q6.10) + Stable Power Monitoring
-------------------------------------------------------------
Supports MNIST (28x28x1) and SVHN (32x32x3). Loads, quantises, runs the
bitstream, aligns power sampling windows with work, integrates energy,
and reports dynamic mW.

Outputs (saved to --metrics-dir):
  - y_hw.npy
  - latency_comm.npy, throughput_comm.npy
  - latency_inf.npy,  throughput_inf.npy
  - power_abs.npy      : concatenated trace    [t, mW]
  - power_dyn.npy      : concatenated dyn trace[t, mW]
  - power_windows.npy  : per-window stats      [t_start, t_end, mean_active_mW, dynamic_mW, energy_mJ, n]
  - cycles_core.npy (optional), cycles_e2e.npy (optional)
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

# Dataset loaders (both expose load_and_quantize_*)
from utils.mnist_utils import load_and_quantize_mnist
from utils.svhn_utils import load_and_quantize_svhn

from utils.power_monitor import PowerMonitor, find_power_inputs
from utils.driver import allocate_overlay
from utils.packing import pack4
from utils.runtime import run_windowed
from utils.io import save_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-b", "--bitstream", required=True, help="Compiled .bit file path")
    p.add_argument("-m", "--metrics-dir", default="metrics/", help="Where to save .npy outputs")

    # Dataset
    p.add_argument("--dataset", choices=["mnist", "svhn"], default=os.environ.get("DATASET", "mnist"),
                   help="Choose which dataset to load and quantise")

    # Progress + packing
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    p.add_argument("--package-data", "-pd", action="store_true",
                   help="Pack 4 16-bit samples into one uint64 word before inference")

    # Cycles
    p.add_argument("--cycles", choices=["auto", "on", "off"], default="auto",
                   help="Cycle counter enable: 'auto'(if present), 'on'(require), 'off'(disable)")
    p.add_argument("--cycle-type", choices=["core", "e2e", "both"], default="core",
                   help="Which counter(s) to read from AXI-GPIO")

    # Power
    p.add_argument("--power-off", action="store_true", help="Disable power monitoring altogether")
    p.add_argument("--power-poll", type=float, default=0.07, help="Power polling period (seconds)")
    p.add_argument("--power-frames", type=int, default=128,
                   help="Number of inferences per power window (↑ to get ≥15 samples)")
    p.add_argument("--power-rail", action="append", default=None,
                   help="Explicit power*_input path(s), repeatable. If omitted, auto-discovers INA/PMBus rails.")
    p.add_argument("--idle-seconds", type=float, default=1.0, help="Idle baseline duration (seconds)")
    return p.parse_args()


def configure_power_monitor(args) -> PowerMonitor | None:
    if args.power_off:
        return None
    rail_paths = args.power_rail if args.power_rail else find_power_inputs(only_ina=True)
    if not rail_paths:
        print("⚠️  No hwmon power rails found; continuing without power monitoring.")
        return None
    print("Power rails:", *[f"  - {p}" for p in rail_paths], sep="\n")
    return PowerMonitor(paths=rail_paths, poll_s=args.power_poll)


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X_i16 : (N, F) int16 in Q6.10
      y_int : (N,)  uint8 labels
    """
    if name == "mnist":
        print("1. Loading and quantising MNIST test set…")
        X_i16, y_int = load_and_quantize_mnist()
    elif name == "svhn":
        print("1. Loading and quantising SVHN test set…")
        # SVHN helper defaults to split='test' and flatten=True → (N, 3072)
        X_i16, y_int = load_and_quantize_svhn()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X_i16, y_int


def main() -> None:
    args = parse_args()
    mdir = Path(args.metrics_dir)
    mdir.mkdir(parents=True, exist_ok=True)

    # 1) Data
    X_i16, y_int = load_dataset(args.dataset)
    N, F = X_i16.shape
    print(f"   Dataset: {args.dataset}  Samples: {N}  Features: {F}")

    if args.package_data:
        print("   Packing data to uint64 (4× int16 per word)…")
        # Expects flattened features; both loaders return (N, F)
        X = pack4(X_i16, batch_size=1_000)
        dtype = np.uint64
    else:
        X = X_i16
        # Stream uses 16-bit words; negatives are two's complement
        dtype = np.uint16

    # 2) FPGA bitstream / overlay
    print("2. Programming FPGA bitstream…")
    nn = allocate_overlay(args.bitstream, feat_dim=X.shape[1], dtype=dtype, enable_cycles=args.cycles)

    # 3) Power monitor (optional)
    mon = configure_power_monitor(args)

    # 4) Inference + Power (windowed) with a single top-level progress bar
    print("3. Running inference…")
    with tqdm(total=X.shape[0], desc="Total inferences", unit="sample",
              disable=args.no_progress) as pbar:
        (y_hw, cyc_core, cyc_e2e,
         lat_c, thr_c, lat_i, thr_i,
         p_abs, p_dyn, p_win) = run_windowed(
            nn, X,
            cycle_type=args.cycle_type,
            power_monitor=mon,
            idle_seconds=args.idle_seconds,
            power_frames=max(1, args.power_frames),
            quiet=True,              # hide internal bar; we drive pbar
            progress_cb=pbar.update  # receive per-window increments
        )

    # 5) Accuracy
    acc = (y_hw.argmax(1) == y_int).mean() * 100
    print(f"4. Accuracy  : {acc:.2f}%")

    # 6) Save everything
    save_metrics(
        mdir,
        y_hw=y_hw,
        latency_comm=lat_c, throughput_comm=thr_c,
        latency_inf=lat_i,  throughput_inf=thr_i,
        power_abs=p_abs, power_dyn=p_dyn, power_windows=p_win,
        cycles_core=cyc_core, cycles_e2e=cyc_e2e
    )

    print("5. Metrics saved →", mdir.resolve())
    if nn.cycles_enabled:
        if cyc_core is not None:
            print("   (core cycles stored in cycles_core.npy)")
        if cyc_e2e is not None:
            print("   (e2e  cycles stored in cycles_e2e.npy)")
    if p_win.size:
        dyn_means = p_win[:, 3]
        print(f"   (dynamic power, per-window)  mean={dyn_means.mean():.2f} mW, P90={np.percentile(dyn_means,90):.2f} mW")
        print("   (absolute power trace stored in power_abs.npy)")
        print("   (dynamic-only power trace stored in power_dyn.npy)")
        print("   (per-window stats stored in power_windows.npy)")


if __name__ == "__main__":
    main()
