from axi_stream_driver import NeuralNetworkOverlay
import numpy as np

FRAC_BITS = 10
SCALE     = 1 << FRAC_BITS  # 2¹⁰ = 1024



ol = NeuralNetworkOverlay(
    "quant_cnn.bit",
    x_shape=(784,),          # one dimension of length 28*28
    y_shape=(10,),           # one dimension of length 10
    dtype=np.int16
)

# Prepare one sample as a flat float32 vector
dummy = np.zeros((784,), dtype=np.int16)

# No encoding needed—hardware wants floats
logits = ol.predict(dummy, debug=True)

print("raw int16:", logits[:10])
