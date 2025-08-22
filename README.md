# BlindSpotDenoiser

BlindSpotDenoiser is a research codebase for denoising one–dimensional astronomical spectra.
It implements a blind‑spot convolutional network in PyTorch and provides utilities for
loading spectral data, training and evaluating the model, and visualising the results.

## Repository structure

```
.
├── configs/      # YAML configuration files that describe experiments
├── scripts/      # Command line entry points for training and hyper‑parameter sweeps
└── src/          # Core library code
```

### `src/`
* **`utils.py`** – miscellaneous helper functions and the `SVDDenoiser` baseline, along with
  routines for signal‑to‑noise calculations and spectral line measurements.
* **`basemodule.py`** – reusable PyTorch Lightning components such as dataset and model
  base classes, a configurable optimiser wrapper, and utility trainer callbacks.
* **`blindspot.py`** – domain‑specific implementations: spectral datasets, the
  `BlindspotModel1D` U‑Net style network, Lightning modules, and the `Experiment` runner
  that ties everything together.
* **`plotter.py`** – plotting utilities for inspecting spectra, signal‑to‑noise improvement,
  and equivalent‑width statistics.

### `configs/`
Configuration files are organised into sections (`model`, `train`, `loss`, `opt`, `data`,
`mask`, `noise`, and `project`).  These values are consumed by the components in
`basemodule.py` and `blindspot.py` to build datasets, models and training loops.

### `scripts/`
The `scripts` directory exposes ready‑to‑run entry points:

* `run_blindspot.py` – launch a training run using a configuration file.
* `run_blindspot_gh.py` – variant with alternative default paths.
* `sweep.py` – orchestration script for Weights & Biases hyper‑parameter sweeps.

Example training command:

```bash
python scripts/run_blindspot.py -f configs/blindspot.yaml -g 1
```

## Data

Experiments expect spectra stored in HDF5 files with datasets named
`spectrumdataset/wave`, `dataset/arrays/flux/value`, and `dataset/arrays/error/value`.
Paths to these files are supplied in the configuration under the `data` section.

## Requirements

The project depends on Python 3, PyTorch, PyTorch Lightning, NumPy, Pandas, h5py, YAML and
Weights & Biases.  Install the dependencies in your environment before running the scripts.

## License

This repository does not currently include an explicit licence; consult the authors before
reusing the code.
