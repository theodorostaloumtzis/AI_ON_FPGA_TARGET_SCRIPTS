from __future__ import annotations
import time
from typing import Optional, Tuple, List, Callable

import numpy as np
from tqdm.auto import tqdm

from utils.power_monitor import measure_idle_baseline, summarize_window, PowerMonitor

CYCLE_DTYP = np.uint32


def run_windowed(
    nn: "NeuralNetworkOverlay",
    X: np.ndarray,
    *,
    encode=None,
    decode=None,
    cycle_type: str = "core",
    power_monitor: Optional[PowerMonitor] = None,
    idle_seconds: float = 1.0,
    power_frames: int = 128,
    quiet: bool = False,
    progress_cb: Optional[Callable[[int], None]] = None,  # external progress updater
):
    """
    Execute inference in windows, optionally sampling power aligned to activity.

    Returns:
      y_pred, cycles_core, cycles_e2e, lat_c, thr_c, lat_i, thr_i, p_abs, p_dyn, p_win
    """
    N = X.shape[0]

    # --- Perf buffers ---
    lat_c = np.empty(N, dtype=np.float64)
    throughput_c = np.empty(N, dtype=np.float64)
    lat_i = np.empty(N, dtype=np.float64)
    throughput_i = np.empty(N, dtype=np.float64)

    if nn.cycles_enabled:
        cycles_core = np.empty(N, dtype=CYCLE_DTYP) if cycle_type in ("core", "both", True) else None
        cycles_e2e = np.empty(N, dtype=CYCLE_DTYP) if cycle_type in ("e2e", "both") else None
    else:
        cycles_core = cycles_e2e = None

    # --- Power baseline ---
    if power_monitor is None:
        idle_baseline = None
    else:
        idle_baseline = measure_idle_baseline(power_monitor, duration_s=idle_seconds)
        print(f"Idle baseline power: {idle_baseline:.1f} mW across {len(power_monitor.paths)} rail(s)")

    # Traces and window summaries
    abs_trace: List[Tuple[float, float]] = []
    dyn_trace: List[Tuple[float, float]] = []
    win_stats: List[Tuple[float, float, float, float, float, int]] = []

    # --- Inference windows ---
    y_pred = None
    step = max(1, power_frames)

    # If an external progress bar is provided, suppress this internal bar.
    progress = tqdm(
        range(0, N, step),
        desc="FPGA inference (windowed)",
        unit="window",
        disable=quiet or (progress_cb is not None),
    )

    for w_start in progress:
        w_end = min(N, w_start + step)
        indices = range(w_start, w_end)

        # Start power sampling (if enabled)
        if power_monitor is not None:
            power_monitor.start()
            time.sleep(power_monitor.poll_s)  # let first sample land

        t0 = time.perf_counter()
        for i in indices:
            ret = nn.predict(
                X[i],
                profile=True,
                return_cycles=(cycle_type if nn.cycles_enabled else False),
                encode=encode,
                decode=decode,
            )

            if nn.cycles_enabled and cycle_type:
                if cycle_type == "both":
                    (y, (cyc_core, cyc_e2e),
                     dts_coms, rate_coms,
                     dts_inf, rate_inf) = ret
                    if cycles_core is not None:
                        cycles_core[i] = cyc_core or 0
                    if cycles_e2e is not None:
                        cycles_e2e[i] = cyc_e2e or 0
                elif cycle_type == "e2e":
                    (y, cyc_e2e,
                     dts_coms, rate_coms,
                     dts_inf, rate_inf) = ret
                    if cycles_e2e is not None:
                        cycles_e2e[i] = cyc_e2e or 0
                else:  # "core"
                    (y, cyc_core,
                     dts_coms, rate_coms,
                     dts_inf, rate_inf) = ret
                    if cycles_core is not None:
                        cycles_core[i] = cyc_core or 0
            else:
                (y,
                 dts_coms, rate_coms,
                 dts_inf, rate_inf) = ret

            if y_pred is None:
                y_pred = np.empty((N, *y.shape), dtype=y.dtype)

            y_pred[i] = y
            lat_c[i] = dts_coms
            throughput_c[i] = rate_coms
            lat_i[i] = dts_inf
            throughput_i[i] = rate_inf

        t1 = time.perf_counter()

        # Stop & summarize power for the window
        if power_monitor is not None:
            power_monitor.stop()
            samples = power_monitor.drain()
            abs_trace.extend(samples)
            if samples and idle_baseline is not None:
                dyn_trace.extend([(t, max(0.0, p - idle_baseline)) for (t, p) in samples])

            stats = summarize_window(samples, t0, t1, idle_baseline or 0.0)
            win_stats.append((
                t0, t1,
                stats["mean_active_mW"],
                stats["dynamic_mW"],
                stats["energy_mJ"],
                stats["n_active"],
            ))

        # External progress update: how many samples processed this window
        if progress_cb is not None:
            progress_cb(w_end - w_start)

    power_abs_arr = np.array(abs_trace, dtype=np.float64) if abs_trace else np.empty((0, 2), dtype=np.float64)
    power_dyn_arr = np.array(dyn_trace, dtype=np.float64) if dyn_trace else np.empty((0, 2), dtype=np.float64)
    power_win_arr = np.array(win_stats, dtype=np.float64) if win_stats else np.empty((0, 6), dtype=np.float64)

    return (
        y_pred, cycles_core, cycles_e2e,
        lat_c, throughput_c, lat_i, throughput_i,
        power_abs_arr, power_dyn_arr, power_win_arr,
    )
