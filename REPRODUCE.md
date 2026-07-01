# Reproducibility

Every number and figure in the paper maps to a script and a committed data
file below. The canonical run is the M1a config (μ-only, mag 20.5–22.5,
`configs/mu_only_baseline_mag205_225.yaml`) at checkpoint `epoch=190`, evaluated
on a held-out 10,000-spectrum test split at noise level 1.

Run everything from the repository root.

## Numbers → data → script

All headline numbers are locked in `sentinels/canonical_bundle.yaml` and are
computed by the canonical metric definitions in `src/metrics/freeze_v1.py`.

| Quantity | Value | Producing script |
|---|---|---|
| Reconstruction S/N (noisy → blindspot) | 7.6 → 122 | `scripts/inference.py` |
| Supervised U-Net ceiling S/N | 154 | `scripts/inference.py` |
| Ca II EW, denoised (8498 / 8542 / 8662 Å) | 0.121 / 0.098 / 0.147 | `scripts/sec5_4_ew_paired_bootstrap_ci.py` |
| Ca II 8498 Å EW, noisy | 0.737 | `scripts/sec5_4_ew_paired_bootstrap_ci.py` |
| Polarity flip rate (noisy / denoised) | 0.276 / 0.00302 | `scripts/inference.py` |
| Integrity audit (permute / occlude, ×1e-7) | 1.19 / 1.49 | `scripts/inference.py` |
| O2 A-band S/N (noisy / denoised) | 0.16 / 3.6 | `scripts/inference.py` |
| Classical baselines S/N (wavelet / noisy-PCA k=16) | 26.6 / 54.0 | `scripts/e1_cpu_baselines.py` + `paper_cpu_common.py` |
| Downstream log g residual (noisy → denoised) | 1.26 → 0.80 dex | `scripts/downstream_logg_calibrator.py` |

### Verify the numbers without a GPU

`sentinels/verify_sentinels.py` cross-checks the numbers printed in the
manuscript against the locked registry (`canonical_bundle.yaml`). It is a pure
text check — no cluster, no GPU:

```bash
python sentinels/verify_sentinels.py --self-test           # calibration (known-good / known-bad)
python sentinels/verify_sentinels.py --tex path/to/main.tex # 18/18 sentinels must PASS
```

The classical-baseline PCA basis used across the paper is fit once by
`scripts/fit_canonical_pca_basis.py` (single source of truth); the γ-measurement
table comes from `scripts/e4_gamma_measurement.py`; downstream-calibrator sweeps
from `scripts/downstream_calibrator_sweep.py` and
`scripts/downstream_logg_baselines.py` (with an independent recon-S/N
cross-check in `scripts/verify_baselines_recon_snr.py`).

## Figures → data → script

| Figure | Script | Input data | Reproducible on CPU? |
|---|---|---|---|
| Fig 1 (representative spectrum) | `scripts/render_current_paper_figs.py` | `data/fixed_preview/bundle_idx336_fig1.npz` | ✅ yes |
| Fig 2 (noise ensemble) | `scripts/render_current_paper_figs.py` + `scripts/reproduce_noise_ensemble_diagnostic.py` | `data/fixed_preview/idx4_m1a123_noise_ensemble_10k.npz` | ✅ yes |
| Fig 3 (Ca II zoom panels) | `scripts/render_current_paper_figs.py` | `data/fixed_preview/bundle_idx336_fig1.npz` | ✅ yes |
| Fig 4 (downstream clean-domain transfer) | `scripts/plot_downstream_clean_domain_transfer.py` | `results/downstream_logg_calibrator/predictions_main.csv`, `results/downstream_calibrator_sweeps/lgbm_sweep_metrics.csv` | ✅ yes |
| Architecture figure | `figs/fig_architecture_blindspot.tex` | none (standalone TikZ) | ✅ `pdflatex` |
| Appendix: S/N improvement | `scripts/plot_paper_figs.py` | raw inference dump (`inference.py --dump-raw`) | ⚠️ needs GPU re-run |
| Appendix: EW residual scatter | `scripts/plot_paper_figs.py` | raw inference dump | ⚠️ needs GPU re-run |

```bash
python scripts/render_current_paper_figs.py             # Figures 1, 2, 3 → reproduced_figs/
python scripts/plot_downstream_clean_domain_transfer.py # Figure 4 → reproduced_figs/
```

The two appendix figures need the per-pixel raw inference dump, which is
regenerated on a GPU node with `python scripts/inference.py --dump-raw <out.h5>`
against the canonical checkpoint, then plotted with `scripts/plot_paper_figs.py`.

## From scratch (training)

Full retraining requires the simulated spectra (HDF5) and a GPU:

```bash
python scripts/run_blindspot.py -f configs/mu_only_baseline_mag205_225.yaml -g 1
python scripts/inference.py --test-path <test_10k>.h5 --num-test-samples 10000 \
       --n-realizations 1 --config configs/mu_only_baseline_mag205_225.yaml
```

Set the `data:` paths in the config to your local spectra before running.
