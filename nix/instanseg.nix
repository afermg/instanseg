{
  lib,
  # build deps
  buildPythonPackage,
  # Py build
  hatchling,
  # Deps
  torch,
  einops,
  fastremap,
  matplotlib,
  numpy,
  tqdm,
  scikit-image,
  requests,
}:
buildPythonPackage {
  pname = "instanseg-torch";
  version = "0.1.1";

  src = ./..; # For local testing, add flag --impure when running

  pyproject = true;
  build-system = [
    hatchling
  ];
  dependencies = [
    torch
    einops
    fastremap
    matplotlib
    numpy
    tqdm
    scikit-image
    requests
  ];

  # The repo carries `instanseg/bioimageio_models/model-index.json` which
  # `instanseg.utils.utils.download_model` reads via `pkgutil.get_data`.
  # Hatch's include rules already cover it; just keep the import smoke
  # check minimal so the build doesn't pull network during `nix build`.
  pythonImportsCheck = [
    "instanseg"
  ];

  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  meta = {
    description = "Embedding-based instance segmentation for cells and nuclei.";
    homepage = "https://github.com/afermg/instanseg";
    license = lib.licenses.asl20;
  };
}
