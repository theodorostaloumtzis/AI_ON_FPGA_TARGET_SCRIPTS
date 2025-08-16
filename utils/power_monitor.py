# power_mon/hwmon.py
from __future__ import annotations
import glob, os, time, threading
from typing import List, Tuple, Optional, Iterable
import numpy as np

def find_power_inputs(only_ina: bool = True) -> List[str]:
    """
    Discover hwmon power input files.
    Returns a list of paths like '/sys/class/hwmon/hwmonX/powerY_input'.
    If only_ina=True, limit to INA/PMBus rails (recommended).
    """
    paths: List[str] = []
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            with open(os.path.join(d, "name")) as f:
                name = f.read().strip().lower()
        except Exception:
            continue
        if only_ina and "ina" not in name and "pmbus" not in name:
            continue
        paths.extend(glob.glob(os.path.join(d, "power*_input")))
    # Stable order
    return sorted(set(paths))

def _read_paths_mw(paths: Iterable[str]) -> Optional[float]:
    total = 0.0
    have = False
    for p in paths:
        try:
            with open(p) as f:
                microw = int(f.read().strip())
            if microw > 0:
                total += microw / 1000.0  # µW → mW
                have = True
        except Exception:
            pass
    return total if have else None

class PowerMonitor:
    """
    Background sampler that polls hwmon power inputs at a fixed cadence.
    Samples are (t, mW) with t from time.perf_counter().
    """
    def __init__(self, paths: Optional[List[str]] = None, poll_s: float = 0.07):
        self.paths  = paths if paths else find_power_inputs(only_ina=True)
        if not self.paths:
            raise RuntimeError("No hwmon power inputs found.")
        self.poll_s = poll_s
        self._stop  = threading.Event()
        self._thr   = None
        self._buf: List[Tuple[float, float]] = []

    def start(self) -> None:
        if self._thr is not None:
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self) -> None:
        if self._thr is None:
            return
        self._stop.set()
        self._thr.join()
        self._thr = None

    def running(self):
        """Context manager: start/stop around a window."""
        class _Ctx:
            def __init__(self, mon: PowerMonitor): self.mon = mon
            def __enter__(self): self.mon.start(); return self.mon
            def __exit__(self, exc_type, exc, tb): self.mon.stop()
        return _Ctx(self)

    def drain(self) -> List[Tuple[float, float]]:
        """Return and clear collected samples."""
        out = self._buf
        self._buf = []
        return out

    def _loop(self):
        while not self._stop.is_set():
            p = _read_paths_mw(self.paths)
            if p is not None:
                self._buf.append((time.perf_counter(), p))
            time.sleep(self.poll_s)

def _trimmed_mean(arr: np.ndarray, trim: float = 0.1) -> float:
    if arr.size == 0:
        return float("nan")
    n = len(arr)
    k = max(0, int(n * trim))
    if 2 * k >= n:
        return float(np.mean(arr))
    arrs = np.sort(arr)
    return float(np.mean(arrs[k:n - k]))

def measure_idle_baseline(monitor: PowerMonitor, duration_s: float = 1.0) -> float:
    """Run the monitor briefly to compute an idle baseline (trimmed mean)."""
    with monitor.running():
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < duration_s:
            time.sleep(monitor.poll_s)
    samples = np.array(monitor.drain(), dtype=float)
    vals = samples[:, 1] if samples.size else np.array([])
    return _trimmed_mean(vals, 0.1) if vals.size else 0.0

def summarize_window(samples: List[Tuple[float, float]],
                     t_start: float,
                     t_end: float,
                     idle_baseline_mW: float) -> dict:
    """
    Compute mean dynamic power and energy for samples between [t_start, t_end].
    Returns dict(mean_active_mW, dynamic_mW, energy_mJ, n_active).
    """
    if not samples or t_end <= t_start:
        return dict(mean_active_mW=0.0, dynamic_mW=0.0, energy_mJ=0.0, n_active=0)

    arr = np.array(samples, dtype=float)
    t, p = arr[:, 0], arr[:, 1]

    mask = (t >= t_start) & (t <= t_end)
    t_w = t[mask]; p_w = p[mask]
    if t_w.size < 2:
        # not enough points, best-effort single-sample estimate
        mean_act = float(np.mean(p_w)) if t_w.size else 0.0
        dyn = max(0.0, mean_act - idle_baseline_mW)
        return dict(mean_active_mW=mean_act, dynamic_mW=dyn, energy_mJ=dyn * (t_end - t_start), n_active=int(t_w.size))

    # Energy via trapezoidal rule; dynamic = energy / duration
    energy_mJ = float(np.trapz(p_w - idle_baseline_mW, t_w))  # mW*s == mJ
    duration  = float(t_w[-1] - t_w[0])
    dynamic   = max(0.0, energy_mJ / duration) if duration > 0 else 0.0
    mean_act  = float(np.mean(p_w))
    return dict(mean_active_mW=mean_act, dynamic_mW=dynamic, energy_mJ=energy_mJ, n_active=int(t_w.size))
