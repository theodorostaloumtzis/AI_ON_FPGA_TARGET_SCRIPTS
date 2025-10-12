#!/usr/bin/env python3
"""
validate_results.py — metrics, plots & power analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Validates FPGA output (y_hw.npy) against ground-truth labels (MNIST or SVHN)
and an optional golden reference. Prints robust latency/throughput statistics,
cycle counts (core / end-to-end), and power-usage stats if available.

Adds:
 - Accuracy comparison (Golden vs GT, HW vs GT) and accuracy drop: (HW − Golden) in percentage points
 - Precision / Recall / F1 for the HW model (and Golden if provided)
 - Saves per-class PRF CSVs (always inside the run's metrics folder)
 - Saves a machine-readable summary of ALL metrics:
     * metrics_summary.json  (nested, detailed)
     * metrics_summary.csv   (flat, one row)
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from math import isnan

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Dataset label helpers
from utils.mnist_utils import get_mnist_test_labels
from utils.svhn_utils import get_svhn_test_labels

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
                   help="Optional golden-reference logits file (same shape as y_hw)")

    # dataset
    p.add_argument("--dataset", choices=["mnist", "svhn"],
                   default=os.environ.get("DATASET", "mnist"),
                   help="Choose ground-truth label source")

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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_ground_truth(dataset: str) -> np.ndarray:
    if dataset == "mnist":
        return ensure_2d(get_mnist_test_labels("mnist"))
    elif dataset == "svhn":
        return ensure_2d(get_svhn_test_labels("svhn"))
    raise ValueError(f"Unknown dataset: {dataset}")

def prf_summary(y_true_cls: np.ndarray, y_pred_cls: np.ndarray, n_classes: int):
    # Per-class
    p, r, f1, supp = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, labels=np.arange(n_classes), zero_division=0
    )
    # Aggregates
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, average="weighted", zero_division=0
    )
    return {
        "per_class": (p, r, f1, supp),
        "micro": (p_micro, r_micro, f1_micro),
        "macro": (p_macro, r_macro, f1_macro),
        "weighted": (p_weight, r_weight, f1_weight),
    }

def save_prf_csv(out_path: Path, p: np.ndarray, r: np.ndarray, f1: np.ndarray, supp: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True
    )
    header = "class,precision,recall,f1, support"
    classes = np.arange(len(p))
    rows = np.column_stack([classes, p, r, f1, supp])
    np.savetxt(out_path, rows, fmt=["%d","%.6f","%.6f","%.6f","%d"], delimiter=",", header=header, comments="")

def _mean(arr: np.ndarray | None) -> float | None:
    if arr is None or arr.size == 0:
        return None
    return float(np.mean(arr))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    mdir = Path(args.metrics_dir)
    mdir.mkdir(parents=True, exist_ok=True)

    # ── predictions ────────────────────────────────────────────────────
    y_hw_path = mdir / args.y_hw
    if not y_hw_path.exists():
        raise FileNotFoundError(f"Missing logits file: {y_hw_path}")
    y_hw   = ensure_2d(np.load(y_hw_path))
    y_true = load_ground_truth(args.dataset)
    y_true, y_hw = align_cols(y_true, y_hw)

    y_ref = safe_load(mdir / args.golden)
    if y_ref is not None:
        y_ref = ensure_2d(y_ref)
        y_true, y_ref = align_cols(y_true, y_ref)

    # Class indices
    y_true_cls = y_true.argmax(1)
    y_hw_cls   = y_hw.argmax(1)
    n_classes  = y_true.shape[1]

    # Accuracies and accuracy drop
    acc_hw_gt = accuracy_score(y_true_cls, y_hw_cls) * 100.0
    print(f"HW vs GT Accuracy        : {acc_hw_gt:.6f}%")
    acc_ref_gt = None
    if y_ref is not None:
        y_ref_cls = y_ref.argmax(1)
        acc_ref_gt = accuracy_score(y_true_cls, y_ref_cls) * 100.0
        print(f"Golden vs GT Accuracy    : {acc_ref_gt:.6f}%")
        drop_pp = acc_hw_gt - acc_ref_gt  # +ve → HW better; -ve → HW drop vs Golden
        sign = "+" if drop_pp >= 0 else ""
        print(f"Accuracy Δ (HW − Golden) : {sign}{drop_pp:.6f} percentage points")

    # ── timing arrays ──────────────────────────────────────────────────
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

    # ── latency / throughput stats (prints) ────────────────────────────
    for tag in ("COMMS", "INF"):
        lat, thr = arrays[f"{tag}_lat"], arrays[f"{tag}_thr"]
        if lat is not None and thr is not None:
            print_latency_thr_stats(tag, lat, thr)

    # ── cycle counts (prints) ──────────────────────────────────────────
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

    # ── power stats & plot (prints) ────────────────────────────────────
    p_abs = safe_load(mdir / args.power_abs)
    p_dyn = None

    p_abs_mean = None
    p_dyn_mean = None
    if p_abs is not None:
        p_abs_mean = print_power_stats("POWER absolute", p_abs)
    if p_dyn is not None:
        p_dyn_mean = print_power_stats("POWER dynamic", p_dyn)

    if p_abs is not None or p_dyn is not None:
        out_path = mdir / args.power_trace_name
        plot_power_trace(p_abs, p_dyn, out_path)
        print(f"Power trace saved → {out_path}")

    # ── PRF metrics (HW; Golden optional) — CSVs saved in the run's metrics folder ──
    # HW
    prf_hw = prf_summary(y_true_cls, y_hw_cls, n_classes)
    p_hw, r_hw, f1_hw, supp_hw = prf_hw["per_class"]
    out_hw_csv = mdir / "prf_hw.csv"
    save_prf_csv(out_hw_csv, p_hw, r_hw, f1_hw, supp_hw)
    print("HW Precision/Recall/F1:")
    print(f"  micro   P/R/F1 = {prf_hw['micro'][0]:.4f} / {prf_hw['micro'][1]:.4f} / {prf_hw['micro'][2]:.4f}")
    print(f"  macro   P/R/F1 = {prf_hw['macro'][0]:.4f} / {prf_hw['macro'][1]:.4f} / {prf_hw['macro'][2]:.4f}")
    print(f"  weightedP/R/F1 = {prf_hw['weighted'][0]:.4f} / {prf_hw['weighted'][1]:.4f} / {prf_hw['weighted'][2]:.4f}")
    print(f"  (per-class PRF saved → {out_hw_csv})")

    # Golden (if provided)
    if y_ref is not None:
        prf_ref = prf_summary(y_true_cls, y_ref_cls, n_classes)
        p_ref, r_ref, f1_ref, supp_ref = prf_ref["per_class"]
        out_ref_csv = mdir / "prf_golden.csv"
        save_prf_csv(out_ref_csv, p_ref, r_ref, f1_ref, supp_ref)
        print("Golden Precision/Recall/F1:")
        print(f"  micro   P/R/F1 = {prf_ref['micro'][0]:.4f} / {prf_ref['micro'][1]:.4f} / {prf_ref['micro'][2]:.4f}")
        print(f"  macro   P/R/F1 = {prf_ref['macro'][0]:.4f} / {prf_ref['macro'][1]:.4f} / {prf_ref['macro'][2]:.4f}")
        print(f"  weightedP/R/F1 = {prf_ref['weighted'][0]:.4f} / {prf_ref['weighted'][1]:.4f} / {prf_ref['weighted'][2]:.4f}")
        print(f"  (per-class PRF saved → {out_ref_csv})")

    # ── plots ─────────────────────────────────────────────────────────
    cm = confusion_matrix(y_true_cls, y_hw_cls, labels=np.arange(n_classes))
    plot_confusion(cm, mdir / args.cm_name)
    plot_multi_roc(y_true, y_hw, mdir / args.roc_name)
    print(f"Confusion matrix saved → {mdir / args.cm_name}")
    print(f"ROC curve saved        → {mdir / args.roc_name}")

    # ─────────────────────────────────────────────────────────────────────────
    # Save a complete, machine-readable metrics summary (JSON + CSV)
    # ─────────────────────────────────────────────────────────────────────────
    import json
    import pandas as pd

    # Averages for COMMS / INF (convert sec → ms)
    comms_lat_mean_ms = None if arrays["COMMS_lat"] is None else _mean(arrays["COMMS_lat"]) * 1000.0
    comms_thr_mean_ips = _mean(arrays["COMMS_thr"])

    inf_lat_mean_ms = None if arrays["INF_lat"] is None else _mean(arrays["INF_lat"]) * 1000.0
    inf_thr_mean_ips = _mean(arrays["INF_thr"])

    # Cycles means + mean time (µs)
    def cycles_mean_and_time_us(arr: np.ndarray | None):
        if arr is None or arr.size == 0:
            return None, None
        mean_cycles = int(np.mean(arr))
        mean_time_us = (mean_cycles / args.clk_mhz) if (args.clk_mhz and not isnan(args.clk_mhz)) else None
        return mean_cycles, mean_time_us

    core_cycles_mean, core_time_us = cycles_mean_and_time_us(core)
    e2e_cycles_mean,  e2e_time_us  = cycles_mean_and_time_us(e2e)

    overhead_cycles_mean, overhead_time_us = (None, None)
    if core is not None and e2e is not None and len(core) == len(e2e):
        overhead = e2e.astype(np.int64) - core.astype(np.int64)
        overhead_cycles_mean, overhead_time_us = cycles_mean_and_time_us(overhead)

    # Power sample counts & means (W)
    def power_stats_summary(arr: np.ndarray | None):
        if arr is None or arr.size == 0:
            return None, None
        n = int(arr.size)
        mean_w = arr.mean() 
        return n, mean_w

    power_abs_mean_w = p_abs_mean
    power_dyn_mean_w = p_dyn_mean 

    # HW and Golden PRF (micro aggregates)
    hw_p, hw_r, hw_f1 = prf_hw["micro"]
    golden_p = golden_r = golden_f1 = None
    if y_ref is not None:
        golden_p, golden_r, golden_f1 = prf_ref["micro"]

    summary = {
        "metrics_dir": str(mdir),
        "clk_mhz": args.clk_mhz,

        "accuracy": {
            "hw_vs_gt_pct": acc_hw_gt,
            "golden_vs_gt_pct": acc_ref_gt,
            "delta_hw_minus_golden_pp": (None if acc_ref_gt is None else (acc_hw_gt - acc_ref_gt)),
        },

        "comms": {
            "latency_mean_ms": comms_lat_mean_ms,
            "throughput_mean_inf_per_s": comms_thr_mean_ips,
        },
        "inf": {
            "latency_mean_ms": inf_lat_mean_ms,
            "throughput_mean_inf_per_s": inf_thr_mean_ips,
        },

        "cycles": {
            "core":     {"mean_cycles": core_cycles_mean,   "mean_time_us": core_time_us},
            "e2e":      {"mean_cycles": e2e_cycles_mean,    "mean_time_us": e2e_time_us},
            "overhead": {"mean_cycles": overhead_cycles_mean, "mean_time_us": overhead_time_us},
        },

        "power": {
            "absolute": {"mean_w": power_abs_mean_w},
            "dynamic":  {"mean_w": power_dyn_mean_w},
        },

        "hw_prf_micro": {
            "precision": float(hw_p) if hw_p is not None else None,
            "recall":    float(hw_r) if hw_r is not None else None,
            "f1":        float(hw_f1) if hw_f1 is not None else None,
        },
        "golden_prf_micro": {
            "precision": (float(golden_p) if golden_p is not None else None),
            "recall":    (float(golden_r) if golden_r is not None else None),
            "f1":        (float(golden_f1) if golden_f1 is not None else None),
        },
    }

    # Write JSON
    json_path = mdir / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary JSON → {json_path}")

    # Also write a flat CSV (single row) for quick aggregation
    flat = {
        "run": mdir.name,
        "clk_mhz": args.clk_mhz,
        "acc_hw_vs_gt_pct": acc_hw_gt,
        "acc_golden_vs_gt_pct": acc_ref_gt,
        "acc_delta_hw_minus_golden_pp": (None if acc_ref_gt is None else (acc_hw_gt - acc_ref_gt)),

        "comms_lat_ms_mean": comms_lat_mean_ms,
        "comms_thr_ips_mean": comms_thr_mean_ips,
        "inf_lat_ms_mean":   inf_lat_mean_ms,
        "inf_thr_ips_mean":  inf_thr_mean_ips,

        "core_cycles_mean": core_cycles_mean,
        "core_time_us_mean": core_time_us,
        "e2e_cycles_mean":  e2e_cycles_mean,
        "e2e_time_us_mean": e2e_time_us,
        "overhead_cycles_mean": overhead_cycles_mean,
        "overhead_time_us_mean": overhead_time_us,


        "power_abs_w_mean": power_abs_mean_w,
        "power_dyn_w_mean": power_dyn_mean_w,

        "hw_precision_micro": float(hw_p) if hw_p is not None else None,
        "hw_recall_micro":    float(hw_r) if hw_r is not None else None,
        "hw_f1_micro":        float(hw_f1) if hw_f1 is not None else None,
        "golden_precision_micro": (float(golden_p) if golden_p is not None else None),
        "golden_recall_micro":    (float(golden_r) if golden_r is not None else None),
        "golden_f1_micro":        (float(golden_f1) if golden_f1 is not None else None),
    }

    import pandas as pd  # local import to avoid hard dep if not needed earlier
    csv_path = mdir / "metrics_summary.csv"
    pd.DataFrame([flat]).to_csv(csv_path, index=False)
    print(f"Saved metrics summary CSV  → {csv_path}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
