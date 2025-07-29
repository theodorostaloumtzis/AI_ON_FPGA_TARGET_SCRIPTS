# utils/power_monitor.py
"""INA260‑only power logger.

Samples INA260 (I²C) power register in continuous 1‑sample mode.
Returns list of (timestamp, power_mw)."""
from __future__ import annotations
import time, threading
from typing import List, Tuple, Optional

import numpy as np
import smbus2

class PowerMonitor:
    """Simple INA260 power monitor.

    Args:
        interval: seconds between samples (default 10 ms).
        i2c_bus:  I²C bus number (default 1).
        addr:     INA260 address (default 0x40).

    Methods:
        start()  – begin background sampling
        stop()   – halt sampling thread
        get_trace() -> List[(ts, mW)]
        save(trace_path, bounds_path, bounds)
    """
    def __init__(self, interval: float = 0.01, *, i2c_bus: int = 1, addr: int = 0x40):
        self.interval = interval
        self.bus = smbus2.SMBus(i2c_bus)
        self.addr = addr
        self._log: List[Tuple[float, float]] = []
        self._active = False
        self._thr: Optional[threading.Thread] = None
        self._init_ina260()
        print(f"[PowerMonitor] INA260 @0x{addr:02X}, dt={interval}s")

    # ---------------- INA260 helpers ----------------
    @staticmethod
    def _swap(val: int) -> int:  # little ↔ big endian 16‑bit
        return ((val & 0xFF) << 8) | (val >> 8)

    def _init_ina260(self):
        # AVG=1 (bits9:7=000), VCT=140 µs (100), ICT=140 µs (100), MODE=111 (cont V+I)
        cfg = 0x4127
        self.bus.write_word_data(self.addr, 0x00, self._swap(cfg))

    def _read_power_mw(self) -> float:
        raw = self.bus.read_word_data(self.addr, 0x03)  # power register
        mw = self._swap(raw) * 1.25  # LSB = 1.25 mW
        return mw

    # ---------------- worker thread ----------------
    def _worker(self):
        self._log = []
        while self._active:
            ts = time.time()
            p = self._read_power_mw()
            self._log.append((ts, p))
            time.sleep(self.interval)

    # ---------------- public API ----------------
    def start(self):
        if self._active:
            return
        self._active = True
        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()
        time.sleep(self.interval*2)  # warm‑up

    def stop(self):
        self._active = False
        if self._thr:
            self._thr.join()

    def get_trace(self):
        return self._log

    def save(self, trace_path: str, bounds_path: str, bounds):
        np.save(trace_path, np.array(self._log, dtype=object))
        np.save(bounds_path, np.array(bounds, dtype=object))
