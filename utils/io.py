from __future__ import annotations
from pathlib import Path
import numpy as np

def _maybe_save(path: Path, arr) -> None:
    if arr is None:
        return
    # For arrays: save only if non-empty (size>0). For scalars: save anyway.
    if hasattr(arr, "size"):
        if getattr(arr, "size", 0) == 0:
            return
    np.save(path, arr)

def save_metrics(
    mdir: Path,
    *,
    y_hw,
    latency_comm, throughput_comm,
    latency_inf,  throughput_inf,
    power_abs=None, power_dyn=None, power_windows=None,
    cycles_core=None, cycles_e2e=None
) -> None:
    mdir.mkdir(parents=True, exist_ok=True)

    np.save(mdir / "y_hw.npy",            y_hw)
    np.save(mdir / "latency_comm.npy",    latency_comm)
    np.save(mdir / "throughput_comm.npy", throughput_comm)
    np.save(mdir / "latency_inf.npy",     latency_inf)
    np.save(mdir / "throughput_inf.npy",  throughput_inf)

    _maybe_save(mdir / "power_abs.npy",     power_abs)
    _maybe_save(mdir / "power_dyn.npy",     power_dyn)
    _maybe_save(mdir / "power_windows.npy", power_windows)

    _maybe_save(mdir / "cycles_core.npy", cycles_core)
    _maybe_save(mdir / "cycles_e2e.npy",  cycles_e2e)
