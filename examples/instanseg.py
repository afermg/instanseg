"""
This example uses a server within the environment defined on `https://github.com/afermg/instanseg.git`.

Run `nix run github:afermg/instanseg/nahual-wrap -- ipc:///tmp/instanseg.ipc` from any directory,
or `nix run --impure .#default -- ipc:///tmp/instanseg.ipc` from the root of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

# `instanseg` isn't in nahual's built-in registry — pass the signature explicitly.
setup, process = dispatch_setup_process("instanseg", signature=("dict", "numpy"))
address = "ipc:///tmp/instanseg.ipc"

# %% Load model server-side. Defaults to the `brightfield_nuclei` pretrained
# checkpoint on cuda:0. Other supported model_id values include
# `fluorescence_nuclei_and_cells` and `single_channel_nuclei`.
parameters = {
    # "model_id": "brightfield_nuclei",
    # "device": 0,
}
response = setup(parameters, address=address)
print(response)
# Loaded model with parameters {'device': 'cuda:0', 'model_id': 'brightfield_nuclei',
#  'cells_and_nuclei': False, 'model_pixel_size': 0.5, 'expected_channels': 3}

# %% Define custom data — 5-D NCZYX. The Z dimension is squeezed server-side.
# brightfield_nuclei expects 3 channels; under-specified inputs are zero-padded.
tile_size = 256
numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, 1, 1, tile_size, tile_size)).astype("float32")
result = process(data, address=address)
print(f"Shape: {result.shape}, Max: {result.max()}")
# Expected: Shape: (1, 256, 256), Max: 0  (random data → no detected instances)
