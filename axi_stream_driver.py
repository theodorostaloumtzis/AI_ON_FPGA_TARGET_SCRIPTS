# axi_stream_driver.py
from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
from pynq import Overlay, allocate
from pynq.lib import AxiGPIO


class NeuralNetworkOverlay(Overlay):
    """
    Overlay wrapper for a streaming-CNN core with AXI DMA and optional
    cycle counters on AXI GPIOs.

    Expected default BD (names can be overridden):
      - hier_0/axi_dma_0            : DMA (sendchannel/recvchannel)
      - axi_gpio_0 channel-1 (in)   : cycles_core_out[31:0]
      - axi_gpio_1 channel-1 (in)   : cycles_e2e_out[31:0]

    You can also map both counters to a *single* AXI-GPIO by choosing different
    channels (e.g., core=ch1, e2e=ch2).

    Parameters
    ----------
    bitfile_name : str
    x_shape, y_shape : tuple[int]
        DMA buffer shapes (samples/words).
    dtype : numpy dtype
        Input buffer dtype (uint16 or uint64 depending on packing).
    enable_cycles : {'auto','on','off', True, False}
        - 'auto'/True : try to bind counters; proceed even if missing
        - 'on'        : require at least one counter; raise if none found
        - 'off'/False : never bind/read counters
    gpio_core_name : str | None
        IP instance that carries the *core* cycles (default 'axi_gpio_0').
    gpio_core_channel : {1,2}
        AXI-GPIO channel index for the core counter (default 1).
    gpio_e2e_name : str | None
        IP instance that carries the *e2e* cycles (default 'axi_gpio_1').
    gpio_e2e_channel : {1,2}
        AXI-GPIO channel index for the e2e counter (default 1).
    """

    def __init__(self,
                 bitfile_name: str,
                 x_shape: Tuple[int, ...],
                 y_shape: Tuple[int, ...],
                 dtype: np.dtype = np.uint16,
                 enable_cycles: Union[str, bool] = "auto",
                 *,
                 gpio_core_name: Optional[str] = "axi_gpio_0",
                 gpio_core_channel: int = 1,
                 gpio_e2e_name: Optional[str] = "axi_gpio_1",
                 gpio_e2e_channel: int = 1,
                 dtbo=None, download=True,
                 ignore_version=False, device=None):

        super().__init__(bitfile_name, dtbo=dtbo,
                         download=download,
                         ignore_version=ignore_version,
                         device=device)

        # ─────────────────────────────  DMA  ─────────────────────────────── #
        self.dma = self.hier_0.axi_dma_0
        self.sendchannel = self.dma.sendchannel
        self.recvchannel = self.dma.recvchannel

        # ─────────────────────  Optional cycle counters  ─────────────────── #
        # normalize enable flag
        if enable_cycles is True:
            enable_mode = "auto"
        elif enable_cycles is False:
            enable_mode = "off"
        else:
            enable_mode = str(enable_cycles).lower()

        self._cycle_ch: Dict[str, Any] = {}

        def _bind(label: str,
                  ip_name: Optional[str],
                  ch_index: int):
            if not ip_name or enable_mode == "off":
                return
            try:
                gpio: AxiGPIO = getattr(self, ip_name)
            except AttributeError:
                return
            # choose channel
            ch = gpio.channel1 if ch_index == 1 else gpio.channel2
            ch.setdirection('in')
            self._cycle_ch[label] = ch

        _bind("core", gpio_core_name, gpio_core_channel)
        _bind("e2e",  gpio_e2e_name,  gpio_e2e_channel)

        if enable_mode == "on" and not self._cycle_ch:
            raise RuntimeError(
                "Cycle counters requested (enable_cycles='on') "
                "but no AXI-GPIO counter could be bound."
            )

        self.cycles_enabled = bool(self._cycle_ch)

        # ─────────────────────────  Buffers  ─────────────────────────────── #
        self.input_buffer  = allocate(shape=x_shape, dtype=dtype)
        self.output_buffer = allocate(shape=y_shape, dtype=np.uint16)

    # --------------------------------------------------------------------- #
    @staticmethod
    def _perf_stats(t0, t1, t2, n=1):
        dt_comm = (t2 - t0).total_seconds()
        dt_inf  = (t2 - t1).total_seconds()
        return (dt_comm, n/dt_comm, dt_inf, n/dt_inf)

    def _read_core(self) -> Optional[int]:
        ch = self._cycle_ch.get("core")
        return int(ch.read()) if ch is not None else None

    def _read_e2e(self) -> Optional[int]:
        ch = self._cycle_ch.get("e2e")
        return int(ch.read()) if ch is not None else None

    # --------------------------------------------------------------------- #
    def predict(self, X,
                *, encode=None, decode=None,
                profile: bool = False,
                return_cycles: Union[bool, str] = False):
        """
        Run one inference and return a tuple that matches the CLI expectations.

        Return layout:
          cycle_type == "both": (y, (cyc_core, cyc_e2e), dtc, rc, dti, ri)
          cycle_type == "e2e" : (y, cyc_e2e,               dtc, rc, dti, ri)
          cycle_type == "core": (y, cyc_core,              dtc, rc, dti, ri)
          cycle_type == False : (y,                        dtc, rc, dti, ri)
        """
        # optional encode
        if encode is not None:
            X = encode(X)

        if profile:
            t0 = datetime.now()

        # queue DMA in both directions
        self.input_buffer[:] = X
        self.sendchannel.transfer(self.input_buffer)
        self.recvchannel.transfer(self.output_buffer)
        self.sendchannel.wait()

        if profile:
            t1 = datetime.now()

        self.recvchannel.wait()

        # copy & optional decode
        y = self.output_buffer.copy()
        if decode is not None:
            y = decode(y)

        # default: don't include cycles
        cyc_ret: Any = None
        if self.cycles_enabled and return_cycles:
            mode = str(return_cycles).lower() if isinstance(return_cycles, str) else "both"
            if mode == "both":
                cyc_ret = (self._read_core(), self._read_e2e())
            elif mode == "e2e":
                cyc_ret = self._read_e2e()
            else:  # "core" or anything else → core
                cyc_ret = self._read_core()

        if profile:
            t2 = datetime.now()
            dtc, rc, dti, ri = self._perf_stats(t0, t1, t2, 1)
        else:
            # keep API shape identical even if profile=False was ever used
            dtc = rc = dti = ri = None

        # build return tuple in the exact order the script unpacks
        if self.cycles_enabled and return_cycles:
            # both/e2e/core already encoded in cyc_ret
            return (y, cyc_ret, dtc, rc, dti, ri)
        else:
            return (y, dtc, rc, dti, ri)
