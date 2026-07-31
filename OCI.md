# InstanSeg Nahual OCI image

Build the reproducible archive and load it into Podman or Docker:

```console
nix build .#oci-image
podman load < result                         # or: docker load < result
```

The image is tagged `nahual/instanseg:local` and listens on TCP port 5555.
Pretrained weights are downloaded from the upstream InstanSeg release on first
setup, so persist `/tmp/nahual` as a model cache:

```console
podman run --rm --device nvidia.com/gpu=all -p 5555:5555 \
  -v nahual-instanseg-cache:/tmp/nahual nahual/instanseg:local
```

For Docker, replace the CDI device option with `--gpus all`. CPU operation is
supported. With Nahual and NumPy installed on the host, run pretrained
end-to-end segmentation with:

```console
NAHUAL_DEVICE=cpu python oci/smoke_test.py
```

The default model is `brightfield_nuclei`. Other model identifiers supported by
the bundled model index can be selected in the Nahual setup request.
