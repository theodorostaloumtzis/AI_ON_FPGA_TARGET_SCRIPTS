from __future__ import annotations
import numpy as np

def print_latency_thr_stats(tag: str, latency_s: np.ndarray, thr_ips: np.ndarray) -> None:
    lat_ms = latency_s * 1e3
    p50, p90, p99 = np.percentile(lat_ms, [50, 90, 99])
    print(f"\n[{tag}]")
    print(f"  Latency   : {lat_ms.mean():.4f} ms ± {lat_ms.std():.4f}  "
          f"(min={lat_ms.min():.4f}, max={lat_ms.max():.4f})")
    print(f"            : P50={p50:.4f}  P90={p90:.4f}  P99={p99:.4f}")
    print(f"  Throughput: {thr_ips.mean():.2f} inf/s ± {thr_ips.std():.2f}")

def print_cycle_stats(tag: str, cycles: np.ndarray, clk_mhz: float | None) -> None:
    if cycles.size == 0:
        return
    p50c, p90c, p99c = np.percentile(cycles, [50, 90, 99])
    print(f"\n[{tag}]   ({cycles.size} samples)")
    print(f"  Mean cycles : {cycles.mean():.0f} ± {cycles.std():.0f}  "
          f"(min={cycles.min():.0f}, max={cycles.max():.0f})")
    print(f"            : P50={p50c:.0f}  P90={p90c:.0f}  P99={p99c:.0f}")
    if clk_mhz:
        clk_hz = float(clk_mhz) * 1e6
        lat_us = cycles / clk_hz * 1e6
        mean_us = lat_us.mean()
        if mean_us > 0:
            print(f"  Mean time  : {mean_us:.3f} µs "
                  f"({1/mean_us*1e6:.2f} inf/s @ {clk_mhz:.0f} MHz)")

def print_power_stats(tag: str, ts_power_mw: np.ndarray) -> None:
    """
    ts_power_mw: Nx2 array (timestamp, power_mW)
    """
    if ts_power_mw.size == 0:
        return
    power = ts_power_mw[:, 1]
    print(f"\n[{tag}]   ({power.size} samples)")
    print(f"  Mean  : {power.mean():.2f} mW ± {power.std():.2f}")
    print(f"  Min   : {power.min():.2f} mW")
    print(f"  Max   : {power.max():.2f} mW")
    print(f"  Peak  : {np.percentile(power, 99):.2f} mW (P99)")
