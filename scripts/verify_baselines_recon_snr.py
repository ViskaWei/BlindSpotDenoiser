#!/usr/bin/env python3
"""Independent correctness check for downstream_logg_baselines.py.

The downstream cross-check (clean/bsn/noisy) validates the calibrator+split+metric
path but does NOT touch the six baseline denoising operators. This script closes
that gap: it recomputes the per-spectrum reconstruction S/N (||x||/||x-xhat||,
freeze_v1.recon_snr_linear) of each baseline's denoised output on the SAME
downstream test split, and compares against tab:appendix_baselines. Matching
magnitudes prove the PCA/wavelet/filter outputs are genuine denoised spectra
(not corrupted), so a high recon-S/N with a negative downstream R^2 is a real
"reconstruction looks good but line shape destroyed" result, not a code bug.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from downstream_logg_baselines import build_representations, DEFAULT_INPUT

CANONICAL_PCA_BASIS = Path(
    "/datascope/subaru/user/swei20/BlindSpot/paper-experiments/canonical_pca_basis.npz"
)


def recon_snr_linear(x_clean: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(x_clean, axis=-1)
    den = np.linalg.norm(x_clean - x_pred, axis=-1)
    out = np.full_like(num, np.inf, dtype=np.float64)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def main() -> None:
    npz = np.load(DEFAULT_INPUT, mmap_mode="r")
    n_total = int(npz["flux"].shape[0])
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_total)
    test_idx = perm[30000:35000]
    reps = build_representations(npz, test_idx, CANONICAL_PCA_BASIS)
    clean = reps["clean"]

    # tab:appendix_baselines mean recon S/N (reconstruction split, different pool).
    appendix = {
        "noisy": 7.6, "savgol_e1": 20.8, "gaussian": 25.8, "median": 15.7,
        "wavelet": 26.6, "noisy_pca_k16": 54.0, "clean_pca_k16": 55.7,
    }
    print(f"{'method':16s} {'recon_SN(downstream)':>22s} {'appendix(recon split)':>24s}")
    for m in ["noisy", "savgol_e1", "gaussian", "median", "wavelet", "noisy_pca_k16", "clean_pca_k16"]:
        sn = recon_snr_linear(clean, reps[m])
        print(f"{m:16s} {float(np.mean(sn)):>22.1f} {appendix[m]:>24.1f}", flush=True)


if __name__ == "__main__":
    main()
