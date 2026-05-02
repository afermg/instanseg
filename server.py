"""Nahual server for InstanSeg.

Loads an InstanSeg pretrained model (default: ``brightfield_nuclei``) via
``instanseg.InstanSeg`` and serves instance segmentation over the standard
Nahual IPC contract.

`process()` accepts a 5-D NCZYX float numpy array, squeezes Z, runs
``InstanSeg.eval_small_image`` per item in the batch, and returns instance
label maps shaped ``(N, H, W)`` as a numpy array.

Run with:
    nix run --impure . -- ipc:///tmp/instanseg.ipc
or:
    python server.py ipc:///tmp/instanseg.ipc
"""

import sys
from functools import partial
from typing import Callable

import numpy
import pynng
import torch
import trio
from nahual.server import responder

from instanseg import InstanSeg

# server.py captures argv[1] at import time; basic_test.py injects a
# placeholder before importing this module.
address = sys.argv[1]


_DEFAULT_EXPECTED_CHANNELS = {
    "brightfield_nuclei": 3,
    "fluorescence_nuclei_and_cells": None,  # channel-invariant: any C OK
    "single_channel_nuclei": 1,
}


def setup(
    model_id: str = "brightfield_nuclei",
    device: int | str | None = 0,
    image_reader: str = "skimage.io",
    verbosity: int = 1,
    expected_channels: int | None = None,
) -> tuple[Callable, dict]:
    """Load an InstanSeg model and return (processor_partial, info_dict).

    Parameters
    ----------
    model_id : str
        Pretrained model identifier from InstanSeg's bioimageio registry.
        Built-in choices include ``brightfield_nuclei``,
        ``fluorescence_nuclei_and_cells``, ``single_channel_nuclei``.
        Weights are downloaded on first use.
    device : int | str | None
        CUDA device index (e.g. 0) or torch device string ("cuda", "cpu").
    image_reader : str
        Forwarded to ``InstanSeg``; we never read images from disk in the
        server, so this only matters if someone later calls ``model.eval``
        with a path. ``skimage.io`` is always present.
    verbosity : int
        InstanSeg verbosity (0/1/2).
    """
    # Resolve device: prefer the requested CUDA index, else fall back.
    if isinstance(device, int) and torch.cuda.is_available():
        torch_device_str = f"cuda:{int(device)}"
    elif isinstance(device, str) and device:
        torch_device_str = device
    elif torch.cuda.is_available():
        torch_device_str = "cuda:0"
    else:
        torch_device_str = "cpu"

    model = InstanSeg(
        model_type=model_id,
        device=torch_device_str,
        image_reader=image_reader,
        verbosity=int(verbosity),
    )

    # `_choose_device` may resolve to "cuda" without an index; normalize.
    resolved_device = str(model.inference_device)
    if resolved_device == "cuda":
        resolved_device = "cuda:0"

    if expected_channels is None:
        expected_channels = _DEFAULT_EXPECTED_CHANNELS.get(model_id, None)

    info = {
        "device": resolved_device,
        "model_id": model_id,
        "cells_and_nuclei": bool(getattr(model.instanseg, "cells_and_nuclei", False)),
        "model_pixel_size": float(getattr(model.instanseg, "pixel_size", 0.0) or 0.0),
        "expected_channels": expected_channels,
    }

    processor = partial(process, model=model, expected_channels=expected_channels)
    return processor, info


def process(
    pixels: numpy.ndarray,
    model: InstanSeg,
    pixel_size: float | None = None,
    target: str = "all_outputs",
    expected_channels: int | None = None,
    **kwargs,
) -> numpy.ndarray:
    """Run InstanSeg on a 5-D NCZYX numpy array.

    The Z dimension is squeezed (we take the first slice — InstanSeg is 2-D).
    Each (C, H, W) item in the batch is run independently through
    ``eval_small_image`` and the resulting instance label maps are stacked.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, H, W)`` with int32 instance labels (0 = background).
    """
    if pixels.ndim != 5:
        raise ValueError(
            f"Expected NCZYX (5D) array, got shape {pixels.shape}"
        )

    n, c, z, h, w = pixels.shape
    # Squeeze Z (InstanSeg is 2-D); take the first plane.
    chw_batch = pixels[:, :, 0, :, :]  # (N, C, H, W)

    # Pad channel dim if the model expects more channels than provided
    # (e.g. brightfield_nuclei wants 3). If the input already has >=
    # expected_channels we pass it through unchanged.
    if expected_channels is not None and chw_batch.shape[1] < expected_channels:
        pad_c = expected_channels - chw_batch.shape[1]
        pad = numpy.zeros((n, pad_c, h, w), dtype=chw_batch.dtype)
        chw_batch = numpy.concatenate((chw_batch, pad), axis=1)

    out_labels: list[numpy.ndarray] = []
    for i in range(n):
        chw = chw_batch[i]  # (C, H, W)
        # eval_small_image accepts 3-D or 4-D float tensors. Pass float32.
        chw_t = torch.from_numpy(numpy.ascontiguousarray(chw)).float()
        instances = model.eval_small_image(
            image=chw_t,
            pixel_size=pixel_size,
            return_image_tensor=False,
            target=target,
            **kwargs,
        )
        # `instances` is a torch.Tensor of shape (1, C_out, H, W).
        # Collapse C_out (nuclei/cells channel) by taking the max over it,
        # which yields a single 2-D label map per item.
        instances = instances.detach().cpu()
        if instances.ndim == 4:
            inst_2d = instances[0].max(dim=0).values
        elif instances.ndim == 3:
            inst_2d = instances.max(dim=0).values
        else:
            inst_2d = instances
        out_labels.append(inst_2d.numpy().astype(numpy.int32))

    return numpy.stack(out_labels, axis=0)


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"InstanSeg server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(responder, setup=setup), sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
