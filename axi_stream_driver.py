from datetime import datetime
import numpy as np
from pynq import Overlay, allocate
from pynq.lib import AxiGPIO        

class NeuralNetworkOverlay(Overlay):
    """
    Overlay wrapper for a streaming-CNN core.

    It will look for an AXI-GPIO (default instance name ``axi_gpio_0``) whose
    Channel-1 input is driven by a 32-bit cycle counter.  If the GPIO is not
    found the overlay falls back gracefully and all cycle-related calls are
    no-ops.

    Parameters
    ----------
    bitfile_name : str
    x_shape, y_shape : tuple[int]
        Shapes of input and output tensors.
    dtype : numpy dtype (default: uint16)
        Precision of the **input** buffer.
    gpio_name : str (default: "axi_gpio_0")
        Instance name of the GPIO that carries the cycle-counter bus.
    enable_cycles : bool | 'auto' (default: 'auto')
        * 'auto' → enable counting only if the GPIO is found
        * True   → raise if the GPIO is missing (you expect it!)
        * False  → never read cycles, even if the GPIO exists
    """

    def __init__(self, bitfile_name,
                 x_shape, y_shape,
                 dtype=np.uint16,
                 gpio_name='axi_gpio_0',
                 enable_cycles='auto',
                 dtbo=None, download=True,
                 ignore_version=False, device=None):

        super().__init__(bitfile_name, dtbo=dtbo,
                         download=download,
                         ignore_version=ignore_version,
                         device=device)

        # ─────────────────────────────  DMA  ────────────────────────────────── #
        self.sendchannel = self.hier_0.axi_dma_0.sendchannel
        self.recvchannel = self.hier_0.axi_dma_0.recvchannel

        # ────────────────────  Optional cycle counter  ──────────────────────── #
        self._cycle_ch1 = None
        if enable_cycles in (True, 'auto'):
            try:
                gpio = getattr(self, gpio_name)
                ch1 = gpio.channel1          # input channel
                ch1.setdirection('in')
                self._cycle_ch1 = ch1
            except AttributeError:
                if enable_cycles is True:
                    raise RuntimeError(f"Cycle-counter GPIO '{gpio_name}' "
                                       "not found in the bitstream.")
                # else: auto-disable silently

        self.cycles_enabled = self._cycle_ch1 is not None

        # ─────────────────────────  Buffers  ───────────────────────────────── #
        self.input_buffer  = allocate(shape=x_shape, dtype=dtype)
        self.output_buffer = allocate(shape=y_shape, dtype=np.uint16)

    # --------------------------------------------------------------------- #
    # helpers
    @staticmethod
    def _perf_stats(t0, t1, t2, n=1):
        dt_comm = (t2 - t0).total_seconds()
        dt_inf  = (t2 - t1).total_seconds()
        return (dt_comm, n/dt_comm,
                dt_inf,  n/dt_inf)

    # --------------------------------------------------------------------- #
    # public API
    def enable_cycle_counter(self, flag=True):
        """Enable / disable reading the hardware cycle counter on the fly."""
        self.cycles_enabled = bool(flag and self._cycle_ch1)

    def predict(self, X, *, encode=None, decode=None,
                profile=False, return_cycles=True):
        """
        Runs one inference and returns:

            output [, cycles] [, dt_comm, rate_comm, dt_inf, rate_inf]

        depending on *return_cycles* and *profile* flags.
        """
        if encode is not None:
            X = encode(X)

        if profile:
            t0 = datetime.now()

        # --- kick off DMA -------------------------------------------------- #
        self.input_buffer[:] = X
        self.sendchannel.transfer(self.input_buffer)
        self.recvchannel.transfer(self.output_buffer)
        self.sendchannel.wait()

        if profile:
            t1 = datetime.now()

        self.recvchannel.wait()

        # --- grab cycle count (if enabled) --------------------------------- #
        cycles = None
        if self.cycles_enabled and return_cycles:
            cycles = self._cycle_ch1.read()

        if decode is not None:
            self.output_buffer[:] = decode(self.output_buffer)

        if profile:
            t2 = datetime.now()

        # --- package results ---------------------------------------------- #
        result = [self.output_buffer.copy()]
        if return_cycles and self.cycles_enabled:
            result.append(cycles)
        if profile:
            result.extend(self._perf_stats(t0, t1, t2, 1))

        return tuple(result) if len(result) > 1 else result[0]
