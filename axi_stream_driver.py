from datetime import datetime
import numpy as np
import threading
import time
import glob
from pynq import Overlay, allocate

class NeuralNetworkOverlay(Overlay):
    def __init__(self, bitfile_name, x_shape, y_shape, dtype=np.uint16,
                 dtbo=None, download=True, ignore_version=False, device=None):
        super().__init__(bitfile_name, dtbo=dtbo, download=download,
                         ignore_version=ignore_version, device=device)
        self.sendchannel = self.hier_0.axi_dma_0.sendchannel
        self.recvchannel = self.hier_0.axi_dma_0.recvchannel
        self.input_buffer = allocate(shape=x_shape, dtype=dtype)
        self.output_buffer = allocate(shape=y_shape, dtype=np.uint16)

    def _print_dt(self, timea, timeb, timec, N):
        dt_coms = timec - timea
        dts_coms = dt_coms.total_seconds()
        rate_coms = N / dts_coms
        dt_inf = timec - timeb
        dts_inf = dt_inf.total_seconds()
        rate_inf = N / dts_inf
        return dts_coms, rate_coms, dts_inf, rate_inf

    def predict(self, X, debug=False, profile=False, encode=None, decode=None):
        if encode:
            X = encode(X)

        if profile:
            timea = datetime.now()

        self.input_buffer[:] = X
        self.sendchannel.transfer(self.input_buffer)
        self.recvchannel.transfer(self.output_buffer)
        self.sendchannel.wait()

        if profile:
            timeb = datetime.now()
        self.recvchannel.wait()

        if decode:
            self.output_buffer = decode(self.output_buffer)

        if profile:
            timec = datetime.now()

        out = [self.output_buffer.copy()]
        if profile:
            dts1, rate1, dts2, rate2 = self._print_dt(timea, timeb, timec, 1)
            out.extend([dts1, rate1, dts2, rate2])
        return tuple(out) if len(out) > 1 else out[0]