#!/usr/bin/env python3
"""
FPGA-based MNIST Inference (INT16 Q6.10) — Single Image Mode
------------------------------------------------------------
Processes each quantized MNIST test image individually using the FPGA accelerator.
Displays logits and timing metrics, with optional progress bar.

Usage:
    python on_target.py --bitstream bitstreams/quant50/quant50_cnn.bit

Optional arguments:
    python on_target.py -b path/to/bit.bit \
            --metrics-dir path/to/metrics/ \
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

OUTPUT_DIM = 10  # Number of MNIST classes

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FPGA MNIST inference (INT16, single image mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--bitstream", required=True, help="Path to the compiled .bit file")
    parser.add_argument("-m", "--metrics-dir", default="metrics/", help="Directory to save .npy metrics")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    return parser.parse_args()

def allocate_overlay(bitstream: str, feature_dim: int) -> NeuralNetworkOverlay:
    """Initializes the FPGA overlay with buffer shapes for single sample inference."""
    return NeuralNetworkOverlay(
        bitstream,
        x_shape=(feature_dim,),
        y_shape=(OUTPUT_DIM,),
        dtype=np.int16,
    )

def run_per_sample(nn: NeuralNetworkOverlay, X: np.ndarray, show_bar: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs inference on each sample in X using the FPGA overlay."""
    n_samples, _ = X.shape
    y_pred_f32 = np.empty((n_samples, OUTPUT_DIM), dtype=np.float32)
    latency_s = np.empty(n_samples, dtype=np.float32)
    throughput = np.empty(n_samples, dtype=np.float32)

    iterator = tqdm(range(n_samples), unit="sample", disable=show_bar, desc="Inference", dynamic_ncols=True) if not show_bar else range(n_samples)

    for idx in iterator:
        sample_i16 = X[idx]
        raw_pred_i16, dt, rate = nn.predict(
            sample_i16,
            profile=True,
            encode=None,  # Already quantized
            decode=None,
        )
        y_pred_f32[idx] = decode_arr(raw_pred_i16.copy())
        latency_s[idx] = dt
        throughput[idx] = rate

    if isinstance(iterator, tqdm):
        iterator.close()

    return y_pred_f32, latency_s, throughput

def main():
    args = parse_args()

    print("1. Loading and quantizing MNIST test set...")
    X_test_i16, y_test_int = load_and_quantize_mnist()
    n_samples, feat_dim = X_test_i16.shape
    print(f"        Samples: {n_samples}   Feature dimension: {feat_dim}")

    print("2. Programming FPGA with bitstream...")
    nn = allocate_overlay(args.bitstream, feat_dim)

    print("3. Starting inference...")
    y_hw_f32, latency_s, throughput = run_per_sample(nn, X_test_i16, args.no_progress)

    print("4. Calculating accuracy...")
    pred = np.argmax(y_hw_f32, axis=1)
    acc = (pred == y_test_int).mean()
    print(f"[RESULT] Hardware accuracy: {acc * 100:.2f}%")

    print("5. Saving results...")
    os.makedirs(args.metrics_dir, exist_ok=True)
    np.save(os.path.join(args.metrics_dir, "y_hw.npy"), y_hw_f32)
    np.save(os.path.join(args.metrics_dir, "latency.npy"), latency_s)
    np.save(os.path.join(args.metrics_dir, "throughput.npy"), throughput)
    print("Metrics saved to", args.metrics_dir)

if __name__ == "__main__":
    main()
