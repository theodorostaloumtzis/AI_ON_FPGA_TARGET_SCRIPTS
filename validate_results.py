#!/usr/bin/env python3
"""
validate_results.py — metrics, plots & power analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Validates FPGA output (y_hw.npy) against MNIST ground-truth labels and an
(optional) golden reference; prints robust latency / throughput statistics,
cycle counts (core / end-to-end), and power-usage stats if available.

Also saves:
 - confusion matrix PNG
 - multi-class ROC PNG
 - power trace PNG (if power data present)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.mnist_utils import get_mnist_test_labels  # ground-truth labels

# utils
from utils.arrays import ensure_2d, align_cols, safe_load, trim_outliers
from utils.metrics import (
    print_latency_thr_stats,
    print_cycle_stats,
    print_power_stats,
)
from utils.plotting import (
    plot_confusion,
    plot_multi_roc,
    plot_power_trace,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("-m", "--metrics-dir", default="metrics/",
                   help="Directory containing *.npy results")
    p.add_argument("--y-hw",      default="y_hw.npy",
                   help="FPGA logits file")
    p.add_argument("--golden",    default="golden_preds.npy",
                   help="Optional golden-reference logits file")
    # timing / throughput arrays
    p.add_argument("--latency-comm",    default="latency_comm.npy")
    p.add_argument("--throughput-comm", default="throughput_comm.npy")
    p.add_argument("--latency-inf",     default="latency_inf.npy")
    p.add_argument("--throughput-inf",  default="throughput_inf.npy")
    # raw cycles
    p.add_argument("--cycles-core", default="cycles_core.npy")
    p.add_argument("--cycles-e2e",  default="cycles_e2e.npy")
    p.add_argument("--clk-mhz",   type=float, default=None,
                   help="FPGA clock in MHz for cycles→µs conversion")
    # power monitoring
    p.add_argument("--power-abs", default="power_abs.npy",
                   help="Absolute board power trace (mW)")
    p.add_argument("--power-dyn", default="power_dyn.npy",
                   help="Idle-subtracted dynamic power trace (mW)")
    p.add_argument("--power-trace-name", default="power_trace.png",
                   help="Output power trace plot file")
    # outlier trimming & plotting
    p.add_argument("--exclude-pct", type=float, default=0.0,
                   help="Trim slowest P%% latency samples (0..100)")
    p.add_argument("--cm-name",  default="confusion_matrix.png")
    p.add_argument("--roc-name", default="roc_curve.png")
    p.add_argument("--no-show",  action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    mdir = Path(args.metrics_dir)

    # ── predictions ────────────────────────────────────────────────────
    y_hw_path = mdir / args.y_hw
    if not y_hw_path.exists():
        raise FileNotFoundError(f"Missing logits file: {y_hw_path}")
    y_hw   = ensure_2d(np.load(y_hw_path))
    y_true = ensure_2d(get_mnist_test_labels("mnist"))
    y_true, y_hw = align_cols(y_true, y_hw)

    y_ref = safe_load(mdir / args.golden)
    if y_ref is not None:
        y_ref = ensure_2d(y_ref)
        y_true, y_ref = align_cols(y_true, y_ref)

    print(f"HW vs GT Accuracy    : "
          f"{accuracy_score(y_true.argmax(1), y_hw.argmax(1)) * 100:.6f}%")
    if y_ref is not None:
        print(f"HW vs Golden Accuracy: "
              f"{accuracy_score(y_ref.argmax(1), y_hw.argmax(1)) * 100:.6f}%")

    # ── load timing arrays ─────────────────────────────────────────────
    arrays = {
        "COMMS_lat": safe_load(mdir / args.latency_comm),
        "COMMS_thr": safe_load(mdir / args.throughput_comm),
        "INF_lat":   safe_load(mdir / args.latency_inf),
        "INF_thr":   safe_load(mdir / args.throughput_inf),
    }

    # ── outlier trimming & masks ───────────────────────────────────────
    mask_inf = None
    if arrays["INF_lat"] is not None and 0.0 < args.exclude_pct < 100.0:
        arrays["INF_lat"], mask_inf = trim_outliers(arrays["INF_lat"], args.exclude_pct)
        if arrays["INF_thr"] is not None and mask_inf is not None:
            arrays["INF_thr"] = arrays["INF_thr"][mask_inf]

    if arrays["COMMS_lat"] is not None and 0.0 < args.exclude_pct < 100.0:
        arrays["COMMS_lat"], mask_comm = trim_outliers(arrays["COMMS_lat"], args.exclude_pct)
        if arrays["COMMS_thr"] is not None and mask_comm is not None:
            arrays["COMMS_thr"] = arrays["COMMS_thr"][mask_comm]

    # ── latency / throughput stats ─────────────────────────────────────
    for tag in ("COMMS", "INF"):
        lat, thr = arrays[f"{tag}_lat"], arrays[f"{tag}_thr"]
        if lat is not None and thr is not None:
            print_latency_thr_stats(tag, lat, thr)

    # ── cycle counts ───────────────────────────────────────────────────
    core = safe_load(mdir / args.cycles_core)
    e2e  = safe_load(mdir / args.cycles_e2e)

    if core is not None and mask_inf is not None and core.size == mask_inf.size:
        core = core[mask_inf]
    if e2e is not None and mask_inf is not None and e2e.size == mask_inf.size:
        e2e = e2e[mask_inf]

    if core is not None:
        print_cycle_stats("CYCLES core", core.astype(np.uint64), args.clk_mhz)
    if e2e is not None:
        print_cycle_stats("CYCLES e2e",  e2e.astype(np.uint64),  args.clk_mhz)
    if core is not None and e2e is not None and len(core) == len(e2e):
        overhead = e2e.astype(np.int64) - core.astype(np.int64)
        print_cycle_stats("CYCLES overhead (e2e - core)", overhead, args.clk_mhz)

    # ── power stats & plot ─────────────────────────────────────────────
    p_abs = safe_load(mdir / args.power_abs)
    p_dyn = safe_load(mdir / args.power_dyn)

    if p_abs is not None:
        print_power_stats("POWER absolute", p_abs)
    if p_dyn is not None:
        print_power_stats("POWER dynamic", p_dyn)

    if p_abs is not None or p_dyn is not None:
        out_path = mdir / args.power_trace_name
        plot_power_trace(p_abs, p_dyn, out_path)
        print(f"Power trace saved → {out_path}")

    # ── plots ─────────────────────────────────────────────────────────
    cm = confusion_matrix(y_true.argmax(1), y_hw.argmax(1))
    plot_confusion(cm, mdir / args.cm_name)
    plot_multi_roc(y_true, y_hw, mdir / args.roc_name)
    print(f"Confusion matrix saved → {mdir / args.cm_name}")
    print(f"ROC curve saved        → {mdir / args.roc_name}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
