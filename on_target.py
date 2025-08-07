#!/usr/bin/env python3
"""
FPGA-based MNIST inference (INT16 Q6.10)
---------------------------------------
Executes one-sample inference on an FPGA bitstream, collecting latency /
throughput metrics – and, when available, raw clock-cycle counts.

Cycle counting can be:
    •  auto  – use the counter only if the IP is present   (default)
    •  on    – require the IP, raise if missing
    •  off   – never read the counter
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from tqdm.auto import tqdm

from axi_stream_driver import NeuralNetworkOverlay
from mnist_utils       import load_and_quantize_mnist


OUTPUT_DIM = 10           # number of logits per MNIST sample
CYCLE_DTYP = np.uint32    # 32-bit counter width (adjust if you changed it)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("-b", "--bitstream", required=True,
                   help="Compiled .bit file path")
    p.add_argument("-m", "--metrics-dir", default="metrics/",
                   help="Where to save .npy outputs")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable tqdm progress bar")
    p.add_argument("--package-data", "-pd", action="store_true",
                   help="Pack 4 pixels into one uint64 word before inference")
    p.add_argument("--cycles", choices=["auto", "on", "off"], default="auto",
                   help=("Cycle counting: 'auto' (use if present), "
                         "'on' (require), 'off' (disable)"))
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def pack4(X_i16: np.ndarray, batch_size: int = 1_000) -> np.ndarray:
    """Packs 4×uint16 pixels into a single uint64 word."""
    N, F = X_i16.shape
    assert F % 4 == 0, "Feature dimension must be divisible by 4"
    X_u64 = np.empty((N, F // 4), dtype=np.uint64)

    shifts = np.array([0, 16, 32, 48], dtype=np.uint64).reshape(1, 1, 4)

    for i in tqdm(range(0, N, batch_size),
                  desc="Packing MNIST", unit="samples"):
        chunk  = X_i16[i:i + batch_size].astype(np.uint64)
        chunk  = chunk.reshape(-1, F // 4, 4)
        packed = ((chunk & 0xFFFF) << shifts).sum(axis=2, dtype=np.uint64)
        X_u64[i:i + batch_size] = packed

    return X_u64


def allocate_overlay(bitfile: str,
                     feat_dim: int,
                     dtype=np.uint16,
                     enable_cycles: str | bool = "auto"
                     ) -> NeuralNetworkOverlay:
    """Download bitstream and return a ready overlay."""
    return NeuralNetworkOverlay(
        bitfile,
        x_shape=(feat_dim,),
        y_shape=(OUTPUT_DIM,),
        dtype=dtype,
        enable_cycles=enable_cycles
    )

# ──────────────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────────────
def run(
    nn: "NeuralNetworkOverlay",
    X: np.ndarray,
    quiet: bool = False,
    *,
    encode=None,
    decode=None,
) -> Tuple[np.ndarray,
           Optional[np.ndarray],
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Runs inference one sample at a time.

    Returns
    -------
    y_pred       : (N, OUTPUT_DIM)
    cycles       : (N,)  or None      – raw cycle counts (if enabled)
    lat_c        : (N,)               – comms latency      [s]
    throughput_c : (N,)               – comms throughput   [samples/s]
    lat_i        : (N,)               – pure FPGA latency  [s]
    throughput_i : (N,)               – pure FPGA thrpt.   [samples/s]
    """
    N = X.shape[0]

    lat_c        = np.empty(N, dtype=np.float64)
    throughput_c = np.empty(N, dtype=np.float64)
    lat_i        = np.empty(N, dtype=np.float64)
    throughput_i = np.empty(N, dtype=np.float64)

    cycles_arr = (np.empty(N, dtype=CYCLE_DTYP)
                  if nn.cycles_enabled else None)

    iterator = tqdm(range(N),
                    desc="FPGA inference",
                    unit="sample",
                    disable=quiet)

    y_pred = None
    for i in iterator:
        ret = nn.predict(X[i],
                         profile=True,
                         return_cycles=nn.cycles_enabled,
                         encode=encode, decode=decode)

        # unpack depending on whether the counter is active
        if nn.cycles_enabled:
            (y, cyc,
             dts_coms, rate_coms,
             dts_inf,  rate_inf) = ret
            cycles_arr[i] = cyc
        else:
            (y,
             dts_coms, rate_coms,
             dts_inf,  rate_inf) = ret

        if y_pred is None:  # discover OUTPUT_DIM lazily
            y_pred = np.empty((N, *y.shape), dtype=y.dtype)

        y_pred[i]       = y
        lat_c[i]        = dts_coms
        throughput_c[i] = rate_coms
        lat_i[i]        = dts_inf
        throughput_i[i] = rate_inf

    return y_pred, cycles_arr, lat_c, throughput_c, lat_i, throughput_i


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    mdir = Path(args.metrics_dir)
    mdir.mkdir(parents=True, exist_ok=True)

    # 1. Load + quantise MNIST
    print("1. Loading and quantising MNIST test set…")
    X_i16, y_int = load_and_quantize_mnist()
    N, F         = X_i16.shape
    print(f"   Samples: {N}  Features: {F}")

    # 2. Optional packing to uint64
    if args.package_data:
        print("   Packing data to uint64 (4 pixels / word)…")
        X     = pack4(X_i16)
        dtype = np.uint64
    else:
        X     = X_i16
        dtype = np.uint16

    # 3. Bitstream → FPGA
    print("2. Programming FPGA bitstream…")
    nn = allocate_overlay(args.bitstream,
                          X.shape[1],
                          dtype=dtype,
                          enable_cycles=args.cycles)

    # 4. Inference
    print("3. Running inference…")
    y_hw, cycles_hw, lat_c, thr_c, lat_i, thr_i = run(
        nn, X, quiet=args.no_progress
    )

    # 5. Accuracy
    acc = (y_hw.argmax(1) == y_int).mean() * 100
    print(f"4. Accuracy  : {acc:.2f}%")

    # 6. Save metrics
    np.save(mdir / "y_hw.npy",            y_hw)
    np.save(mdir / "latency_comm.npy",    lat_c)
    np.save(mdir / "throughput_comm.npy", thr_c)
    np.save(mdir / "latency_inf.npy",     lat_i)
    np.save(mdir / "throughput_inf.npy",  thr_i)
    if cycles_hw is not None:
        np.save(mdir / "cycles_raw.npy", cycles_hw)

    print("5. Metrics saved →", mdir.resolve())
    if nn.cycles_enabled:
        print("   (cycle counts stored in cycles_raw.npy)")


if __name__ == "__main__":
    main()
