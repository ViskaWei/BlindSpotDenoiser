#!/usr/bin/env python3
"""Fit the canonical PCA basis — single source of truth for ALL PCA baselines.

Why this exists
---------------
The PCA baseline is the only data-dependent classical reference: it must fit a
component basis on a fixed spectrum pool. The reconstruction appendix (e1) fits
its basis on the 200k reconstruction-train pool (-> recon S/N 54.0/55.7), while
the downstream calibrator originally re-fit a separate basis on its own 30k pool
(-> recon S/N 42.3/43.3). Same concept, two bases, two numbers. This script
materializes ONE basis so both tables agree.

How alignment is guaranteed
---------------------------
It calls the EXACT e1 ``fit_pca_models`` (same n_components 128/512, same
train_noise_seed 20260428, same random_state 20260428, same iterated_power 3) on
the SAME train_200k_0 split. e1 fits this identical basis internally, so the
stored file is numerically equivalent to the appendix basis. Downstream then
LOADS this file instead of re-fitting -> PCA reconstruction S/N matches the
appendix, and there is exactly one PCA basis in the whole paper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paper_cpu_common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_MASK_PATH,
    load_mask,
    load_split,
)
from scripts.e1_cpu_baselines import fit_pca_models  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-split", default="train_200k_0")
    ap.add_argument("--pca-fit-n", type=int, default=200_000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--train-noise-seed", type=int, default=20260428)
    ap.add_argument("--random-state", type=int, default=20260428)
    ap.add_argument("--iterated-power", type=int, default=3)
    a = ap.parse_args()

    mask = load_mask(DEFAULT_MASK_PATH)
    wave, x_train, e_train, meta = load_split(
        DEFAULT_DATA_ROOT, a.train_split, mask=mask, n=a.pca_fit_n
    )
    print(f"[fit] e1 fit_pca_models on x_train={x_train.shape}", flush=True)
    clean_pca, noisy_pca = fit_pca_models(
        x_train,
        e_train,
        n_fit=a.pca_fit_n,
        train_noise_seed=a.train_noise_seed,
        random_state=a.random_state,
        iterated_power=a.iterated_power,
    )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        clean_mean=clean_pca.mean_.astype(np.float32),
        clean_components=clean_pca.components_[: a.k].astype(np.float32),
        noisy_mean=noisy_pca.mean_.astype(np.float32),
        noisy_components=noisy_pca.components_[: a.k].astype(np.float32),
        wave=wave.astype(np.float64),
        k=a.k,
        fit_split=a.train_split,
        fit_n=int(min(a.pca_fit_n, x_train.shape[0])),
        train_noise_seed=a.train_noise_seed,
        random_state=a.random_state,
        iterated_power=a.iterated_power,
        mask_path=DEFAULT_MASK_PATH,
    )
    print(
        f"[done] wrote {out} clean_comp={clean_pca.components_[:a.k].shape} "
        f"noisy_comp={noisy_pca.components_[:a.k].shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
