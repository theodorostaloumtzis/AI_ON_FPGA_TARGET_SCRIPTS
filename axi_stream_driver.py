from datetime import datetime
import numpy as np
import threading
import time
from pynq import Overlay, allocate
import smbus2


class NeuralNetworkOverlay(Overlay):
    def __init__(
        self, bitfile_name, x_shape, y_shape, dtype=np.float32, dtbo=None, download=True, ignore_version=False, device=None
    ):
        super().__init__(bitfile_name, dtbo=dtbo, download=download, ignore_version=ignore_version, device=device)
        self.sendchannel = self.hier_0.axi_dma_0.sendchannel
        self.recvchannel = self.hier_0.axi_dma_0.recvchannel
        self.input_buffer = allocate(shape=x_shape, dtype=dtype)
        self.output_buffer = allocate(shape=y_shape, dtype=dtype)

        # INA260 setup
        self.i2c_bus = smbus2.SMBus(1)
        self.ina260_addr = 0x40
        self._power_samples = []
        self._sampling_thread = None
        self._sampling_active = False

    def _read_power_mw(self):
        """Reads power in milliwatts from the INA260 sensor."""
        try:
            power_reg = 0x03
            raw = self.i2c_bus.read_word_data(self.ina260_addr, power_reg)
            raw = ((raw & 0xFF) << 8) | (raw >> 8)
            return raw * 1.25
        except Exception as e:
            print("INA260 read error:", e)
            return 0

    def _power_sampling_worker(self, interval=0.01):
        """Background thread to sample INA260 power."""
        self._power_samples = []
        while self._sampling_active:
            power = self._read_power_mw()
            timestamp = time.time()
            self._power_samples.append((timestamp, power))
            time.sleep(interval)

    def _start_power_monitoring(self):
        self._sampling_active = True
        self._sampling_thread = threading.Thread(target=self._power_sampling_worker, daemon=True)
        self._sampling_thread.start()

    def _stop_power_monitoring(self):
        self._sampling_active = False
        if self._sampling_thread:
            self._sampling_thread.join()

    def _compute_avg_power(self):
        if not self._power_samples:
            return 0, 0
        powers = [p for t, p in self._power_samples]
        times = [t for t, p in self._power_samples]
        duration = times[-1] - times[0] if len(times) > 1 else 0
        avg_power = sum(powers) / len(powers)
        return avg_power, duration

    def _print_dt(self, timea, timeb, timec, N):
        dt_coms = timec - timea
        dts_coms = dt_coms.seconds + dt_coms.microseconds * 10**-6
        rate_coms = N / dts_coms

        dt_inf = timec - timeb
        dts_inf = dt_inf.seconds + dt_inf.microseconds * 10**-6
        rate_inf = N / dts_inf

        return dts_coms, rate_coms, dts_inf, rate_inf

    def predict(self, X, debug=False, profile=False, power_profile=False, encode=None, decode=None):
        """
        Run inference with optional timing and threaded power monitoring.
        """
        if profile or power_profile:
            timea = datetime.now()
        if power_profile:
            self._start_power_monitoring()
            time.sleep(0.02)  # optional warm-up

        if encode is not None:
            X = encode(X)
        self.input_buffer[:] = X
        self.sendchannel.transfer(self.input_buffer)
        self.recvchannel.transfer(self.output_buffer)
        self.sendchannel.wait()

        if profile or power_profile:
            timeb = datetime.now()

        self.recvchannel.wait()

        if power_profile:
            self._stop_power_monitoring()

        if decode is not None:
            self.output_buffer = decode(self.output_buffer)

        if profile or power_profile:
            timec = datetime.now()

        # Results
        out = [self.output_buffer]
        if profile:
            dts1, rate1, dts2, rate2 = self._print_dt(timea, timeb, timec, 1)
            out.extend([dts1, rate1, dts2, rate2])
        if power_profile:
            avg_power, power_duration = self._compute_avg_power()
            out.extend([avg_power, power_duration])
        return tuple(out) if len(out) > 1 else out[0]
