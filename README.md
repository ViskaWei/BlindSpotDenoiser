# BlindSpotDenoiser

Source code for **_Self-Supervised Blindspot Denoising for Low Signal-to-Noise
Stellar Spectra_** (Wei et al. 2026).

We train a one-dimensional self-supervised blindspot denoiser on noisy stellar
spectra alone, predicting each wavelength bin from a receptive field that
excludes that bin — so no paired clean reference is ever required. On a
survey-relevant simulated benchmark with heteroscedastic photon, sky, and read
noise, the mean per-spectrum reconstruction S/N rises from **7.6** on the noisy
input to **122** on the denoised output. A surface-gravity (log g) estimator
trained on clean spectra sees its test residual scatter fall from **1.26 dex**
on noisy input to **0.80 dex** on the denoised output, while validation-tuned
classical smoothers stay at the noisy-input scale.

## Authors

Viska Wei, Alexander S. Szalay, László Dobos, Xiaosheng Zhao,
Tamás Budavári, Balázs Pál, Rosemary F. G. Wyse.

## Repository layout

```
src/                     Core library
  basemodule.py          Reusable PyTorch-Lightning base classes (dataset/model/optimizer)
  blindspot.py           BlindspotModel1D architecture + datasets + losses + Experiment runner
  utils.py               EW / S/N measurement, SVD baseline, config + air/vac helpers
  plotter.py             SpecPlotter (appendix figures)
  metrics/freeze_v1.py   Canonical metric definitions — single source of truth for every
                         paper number (reconstruction S/N, Ca II EW, polarity, O2 band, ...)
scripts/                 Reproduction entry points (see REPRODUCE.md)
configs/                 Paper-canonical training config (M1a, mag 20.5-22.5)
sentinels/               Number-reproduction gate (verify_sentinels.py + canonical_bundle.yaml)
data/fixed_preview/      Committed spectrum bundles for figure reproduction
results/                 Committed downstream-calibrator CSVs (Figure 4 inputs)
figs/                    Architecture figure source (standalone TikZ)
```

## Install

```bash
pip install -r requirements.txt
```

Python ≥ 3.11. GPU (CUDA) is only needed to retrain the model or regenerate the
appendix figures; everything else runs on CPU.

## Quick start

Verify every headline number in the paper against the locked registry:

```bash
python sentinels/verify_sentinels.py --self-test
```

Regenerate the main-text figures from committed data (no GPU, no cluster):

```bash
python scripts/render_current_paper_figs.py                    # Figures 1, 2, 3
python scripts/plot_downstream_clean_domain_transfer.py        # Figure 4
```

Outputs land in `reproduced_figs/`.

Train the denoiser (GPU, requires the simulated spectra — see below):

```bash
python scripts/run_blindspot.py -f configs/mu_only_baseline_mag205_225.yaml -g 1
```

## Data

Training and inference expect simulated stellar spectra stored in HDF5, keyed by
`spectrumdataset/wave`, `dataset/arrays/flux/value`, and
`dataset/arrays/error/value`. The canonical grid is generated with
[`pfsspec`](https://github.com/ViskaWei) / BOSZ templates (Dobos et al.); the
`data:` paths in the config point to the survey compute cluster and must be set
to your own environment. The committed `.npz` / `.csv` files are the exact
inputs needed to reproduce the figures and numbers without re-running training.

## Reproducibility

See **[REPRODUCE.md](REPRODUCE.md)** for a line-by-line map from every paper
number and figure to the script and data that produce it.

## Citation

```bibtex
@article{wei2026blindspot,
  title  = {Self-Supervised Blindspot Denoising for Low Signal-to-Noise Stellar Spectra},
  author = {Wei, Viska and Szalay, Alexander S. and Dobos, L\'aszl\'o and
            Zhao, Xiaosheng and Budav\'ari, Tam\'as and P\'al, Bal\'azs and
            Wyse, Rosemary F. G.},
  year   = {2026}
}
```

## License

[MIT](LICENSE) © 2026 Viska Wei.
