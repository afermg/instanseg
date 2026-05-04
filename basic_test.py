"""Standalone smoke test for InstanSeg.

Loads the model the same way ``server.py`` does and runs a forward pass on
a small synthetic input. Does NOT spin up the IPC server.

Run from the repo root:
    nix develop --impure --command python basic_test.py

Expected output:
    setup info dict (with cuda:0)
    type and shape of process() result
"""

import importlib.util
import os
import sys

# Drop the repo root from sys.path BEFORE importing instanseg — otherwise
# the in-tree `instanseg/` source dir shadows the nix-built package.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) not in {_HERE, ""}]

# server.py reads sys.argv[1] at import time; inject a placeholder so
# importing it from this file doesn't crash.
if len(sys.argv) < 2:
    sys.argv.append("ipc:///tmp/instanseg_basic_test.ipc")

import numpy  # noqa: E402

# Load `server.py` directly via importlib so we never need to put _HERE back
# on sys.path (which would re-trigger the instanseg source-dir shadow).
_spec = importlib.util.spec_from_file_location("server", os.path.join(_HERE, "server.py"))
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)
setup = server.setup


def main() -> None:
    processor, info = setup()
    print(f"setup: {info}")
    assert "cuda" in info["device"], f"Not on GPU! info['device']={info['device']!r}"

    # Image segmentation model: 5-D NCZYX. setup() reports the model's
    # expected channel count; the server zero-pads under-specified inputs
    # (e.g. 1 -> 3 for brightfield_nuclei).
    numpy.random.seed(0)
    data = numpy.random.random_sample((1, 1, 1, 256, 256)).astype(numpy.float32)
    out = processor(data)

    arr = out.cpu().numpy() if hasattr(out, "cpu") else out
    print(f"process: {type(arr).__name__} shape={arr.shape} dtype={arr.dtype} max={int(arr.max())}")


if __name__ == "__main__":
    main()
