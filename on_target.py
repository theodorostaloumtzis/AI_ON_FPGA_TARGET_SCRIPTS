#!/usr/bin/env python3
"""
On‑target inference for MNIST (INT16 Q6.10) — **one‑by‑one mode**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Runs each quantised MNIST test image through the FPGA accelerator
**individually** (no batching) and dumps logits + simple timing
metrics. A live **tqdm** bar shows progress.

Example:
    python3 on_target.py --bitstream bitstreams/quant50/quant50_cnn.bit

Optional:
    python3 on_target.py -b path/to/bit.bit \
            --metrics-dir runs/2025‑07‑23/ \
            --no-progress
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Tuple

import numpy as np
from tqdm import tqdm

from axi_stream_driver import NeuralNetworkOverlay
from mnist_utils import load_and_quantize_mnist, decode_arr

# ───────────────────────────────────────────────────────────────────────────
OUTPUT_DIM = 10  # MNIST classes 0‑9

# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FPGA MNIST inference (INT16, image‑by‑image)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-b", "--bitstream", required=True, help="Path to the compiled .bit file")
    p.add_argument("-m", "--metrics-dir", default="metrics/", help="Destination folder for .npy metrics")
    p.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar")
    return p.parse_args()

# ───────────────────────────────────────────────────────────────────────────
# Overlay helper
# ───────────────────────────────────────────────────────────────────────────

def allocate_overlay(bitstream: str, feature_dim: int) -> NeuralNetworkOverlay:
    """Instantiate the overlay with per‑sample buffer shapes."""
    return NeuralNetworkOverlay(
        bitstream,
        x_shape=(feature_dim,),
        y_shape=(OUTPUT_DIM,),
        dtype=np.int16,
    )

# ───────────────────────────────────────────────────────────────────────────
# Inference loop
# ───────────────────────────────────────────────────────────────────────────

def run_per_sample(nn: NeuralNetworkOverlay, X: np.ndarray, show_bar: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run *X* through *nn* one sample at a time."""
    n_samples, _ = X.shape
    y_pred_f32 = np.empty((n_samples, OUTPUT_DIM), dtype=np.float32)
    latency_s = np.empty(n_samples, dtype=np.float32)
    throughput = np.empty(n_samples, dtype=np.float32)

    iterator = tqdm(range(n_samples), unit="sample", disable=show_bar is False and False, desc="Inference", dynamic_ncols=True) if not show_bar else range(n_samples)

    for idx in iterator:
        sample_i16 = X[idx]
        raw_pred_i16, dt, rate = nn.predict(
            sample_i16,
            profile=True,
            encode=None,  # already quantised
            decode=None,
        )
        y_pred_f32[idx] = decode_arr(raw_pred_i16.copy())
        latency_s[idx] = dt
        throughput[idx] = rate

    if isinstance(iterator, tqdm):
        iterator.close()

    return y_pred_f32, latency_s, throughput

# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Dataset ────────────────────────────────────────────────────────────
    print("[INFO] Loading & quantising dataset …")
    X_test_i16, y_test_int = load_and_quantize_mnist()
    n_samples, feat_dim = X_test_i16.shape
    print(f"        samples: {n_samples}   feature dim: {feat_dim}")

    # ── Overlay ────────────────────────────────────────────────────────────
    print("[INFO] Programming FPGA with bitfile …")
    nn = allocate_overlay(args.bitstream, feat_dim)

    # ── Inference ──────────────────────────────────────────────────────────
    print("[INFO] Running inference …")
    y_hw_f32, latency_s, throughput = run_per_sample(nn, X_test_i16, args.no_progress)

    # ── Metrics ────────────────────────────────────────────────────────────
    print("[INFO] Computing accuracy …")
    pred = np.argmax(y_hw_f32, axis=1)
    acc = (pred == y_test_int).mean()
    print(f"[RESULT] HW accuracy: {acc * 100:.2f}%")

    # ── Persist ────────────────────────────────────────────────────────────
    print("[INFO] Saving metrics …")
    os.makedirs(args.metrics_dir, exist_ok=True)
    np.save(os.path.join(args.metrics_dir, "y_hw.npy"), y_hw_f32)
    np.save(os.path.join(args.metrics_dir, "latency.npy"), latency_s)
    np.save(os.path.join(args.metrics_dir, "throughput.npy"), throughput)
    print("[OK] Files written to", args.metrics_dir)


if __name__ == "__main__":
    main()
