from __future__ import annotations
from axi_stream_driver import NeuralNetworkOverlay

OUTPUT_DIM = 10

def allocate_overlay(bitfile: str, feat_dim: int, *, dtype, enable_cycles: str | bool = "auto"
                     ) -> NeuralNetworkOverlay:
    return NeuralNetworkOverlay(
        bitfile,
        x_shape=(feat_dim,),
        y_shape=(OUTPUT_DIM,),
        dtype=dtype,
        enable_cycles=enable_cycles
    )
