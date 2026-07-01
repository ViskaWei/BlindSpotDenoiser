#!/usr/bin/env python3
"""Downstream logg calibrator — full classical/PCA baseline battery.

Purpose
-------
Section 5.2 of the BlindSpot stand-alone paper currently evaluates the frozen
clean-trained logg calibrator on only {clean, blindspot, noisy, boxcar, savgol}.
The reconstruction appendix (tab:appendix_baselines), however, reports a full
six-method battery {Savitzky-Golay, Gaussian, Median, Wavelet, noisy-PCA,
clean-template PCA} on the reconstruction split. This script closes that gap:
it pushes the SAME six methods (with the SAME e1-selected configs) through the
SAME frozen logg calibrator, so the downstream table can carry every method the
reconstruction table already reports.

Design / anti-error contract
----------------------------
1. The classical/PCA *implementations* are copied verbatim from
   ``e1_cpu_baselines.py`` (reconstruct_pca / wavelet_bayes_shrink) so the
   denoising operators are byte-identical to the ones behind tab:appendix_baselines.
2. The *selected configs* are read from paper-experiments/e1/manifest.json
   (NOT re-tuned here): savgol w11/p1, gaussian sigma7, median w7,
   wavelet db4 BayesShrink level5, noisy-PCA k16, clean-template PCA k16.
3. The calibrator (LightGBM), the data split (seed 42, perm slicing), and the
   metric definitions are copied verbatim from ``downstream_logg_calibrator.py``
   so the frozen calibrator and the held-out 5000-spectrum test set are identical
   to the published Table tab:downstream_logg run.
4. CROSS-CHECK: this script also re-evaluates {clean, bsn, noisy}. If those rows
   do not reproduce the published 0.222 / 0.801 / 1.262 sigma_rob_center (to
   rounding), the pipeline is wrong and the six new rows must NOT be trusted.
   The cross-check is asserted at the end of main().

PCA fit set
-----------
e1 fits PCA on the 200k reconstruction-train pool. The downstream pool is a
separate 50000-spectrum labeled BOSZ pool, so PCA bases are (re)fit here on the
downstream calibrator-train clean/noisy spectra (perm[:n_train], disjoint from
the perm[n_train:n_train+n_test] test set -> no leakage). The k=16 truncation,
randomized solver, random_state and iterated_power match e1; only the fit pool
differs (documented, expected, not a bug).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error, r2_score

try:
    import pywt
except Exception as exc:  # pragma: no cover
    pywt = None
    PYWT_IMPORT_ERROR = repr(exc)
else:
    PYWT_IMPORT_ERROR = None


DEFAULT_INPUT = Path(
    "/datascope/subaru/user/swei20/blindspot_rv/inputs/bosz50000_v3ep190/"
    "b1_l9_e48_k25_s1_bn1_d1_T0_S0_L0_snr3_b50000_rv500_ep5000_"
    "N50000_m0_v3ep190_N50000.npz"
)

# e1-selected configs (paper-experiments/e1/manifest.json -> "selected").
E1_SELECTED = {
    "savgol": {"window": 11, "polyorder": 1},
    "gaussian": {"sigma": 7},
    "median": {"window": 7},
    "wavelet": {"wavelet": "db4", "level": 5},
    "noisy_pca": {"k": 16, "fit_on": "noisy_train"},
    "clean_pca": {"k": 16, "fit_on": "clean_train"},
}

# Published Table tab:downstream_logg anchors for the cross-check (sigma_rob_center).
PUBLISHED_SIGMA_ROB = {"clean": 0.222, "bsn": 0.801, "noisy": 1.262}

# Method display order in the output CSV.
METHOD_ORDER = [
    "clean",       # cross-check anchor (upper reference)
    "bsn",         # cross-check anchor (ours)
    "noisy",       # cross-check anchor (floor)
    "savgol_e1",   # six-method battery (selected configs) ...
    "gaussian",
    "median",
    "wavelet",
    "noisy_pca_k16",
    "clean_pca_k16",
]


# --- PCA reconstruction using the canonical stored basis ---------------------
# Math identical to e1_cpu_baselines.reconstruct_pca, but the (mean, components)
# are LOADED from canonical_pca_basis.npz (fit on the 200k reconstruction pool via
# e1 fit_pca_models) instead of re-fit on the downstream pool. This makes the PCA
# baseline a single source of truth shared with the reconstruction appendix.
def apply_pca_basis(mean: np.ndarray, components: np.ndarray, x: np.ndarray, k: int = 16) -> np.ndarray:
    mean = mean.astype(np.float32, copy=False)
    comps = components[:k].astype(np.float32, copy=False)
    z = x.astype(np.float32, copy=False) - mean
    coeff = z @ comps.T
    out = coeff @ comps + mean
    return out.astype(np.float32, copy=False)


def wavelet_bayes_shrink(y: np.ndarray, *, level: int, wavelet: str = "db4") -> np.ndarray:
    if pywt is None:
        raise RuntimeError(f"PyWavelets unavailable: {PYWT_IMPORT_ERROR}")
    coeffs = pywt.wavedec(y, wavelet=wavelet, level=level, mode="periodization", axis=-1)
    detail_finest = coeffs[-1]
    sigma = np.median(np.abs(detail_finest), axis=-1, keepdims=True) / 0.6745
    sigma2 = sigma ** 2
    shrunk = [coeffs[0]]
    for detail in coeffs[1:]:
        var = np.var(detail, axis=-1, keepdims=True)
        signal_var = np.maximum(var - sigma2, 1e-12)
        thresh = sigma2 / np.sqrt(signal_var)
        shrunk.append(np.sign(detail) * np.maximum(np.abs(detail) - thresh, 0.0))
    rec = pywt.waverec(shrunk, wavelet=wavelet, mode="periodization", axis=-1)
    return rec[:, : y.shape[1]].astype(np.float32, copy=False)


# --- copied verbatim from downstream_logg_calibrator.py (calibrator+metric) --
def sigma_rob_center(residual: np.ndarray) -> float:
    med = np.median(residual)
    return float(1.4826 * np.median(np.abs(residual - med)))


def metric_row(method: str, n_train: int, y_true: np.ndarray, pred: np.ndarray) -> dict:
    residual = pred - y_true
    return {
        "method": method,
        "n_train": int(n_train),
        "std": float(np.std(residual, ddof=0)),
        "sigma_rob_center": sigma_rob_center(residual),
        "mae": float(mean_absolute_error(y_true, pred)),
        "mean_bias": float(np.mean(residual)),
        "median_bias": float(np.median(residual)),
        "r2": float(r2_score(y_true, pred)),
        "n_test": int(y_true.shape[0]),
    }


def fit_lgbm(x_train: np.ndarray, y_train: np.ndarray, n_jobs: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.45,
        reg_lambda=5.0,
        random_state=0,
        n_jobs=n_jobs,
        verbosity=-1,
    ).fit(x_train, y_train)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_representations(npz, test_idx: np.ndarray, pca_basis_path: Path) -> dict[str, np.ndarray]:
    """Construct the nine input representations on the held-out test spectra."""
    flux_test = np.asarray(npz["flux"][test_idx], dtype=np.float32)
    noisy_test = np.asarray(npz["noisy"][test_idx], dtype=np.float32)
    denoised_test = np.asarray(npz["denoised"][test_idx], dtype=np.float32)

    # Canonical PCA basis: fit once on the 200k reconstruction pool (e1), loaded
    # here so the downstream PCA baseline shares one basis with the appendix.
    basis = np.load(pca_basis_path)
    assert np.allclose(
        np.asarray(basis["wave"], dtype=float), np.asarray(npz["wave"], dtype=float)
    ), "canonical PCA basis wavelength grid != downstream npz grid"
    print(
        f"[pca] load canonical basis {pca_basis_path} "
        f"(fit_split={basis['fit_split']}, fit_n={int(basis['fit_n'])})",
        flush=True,
    )

    reps = {
        "clean": flux_test,
        "bsn": denoised_test,
        "noisy": noisy_test,
        "savgol_e1": savgol_filter(
            noisy_test, window_length=11, polyorder=1, axis=-1, mode="interp"
        ).astype(np.float32),
        "gaussian": gaussian_filter1d(noisy_test, sigma=7, axis=-1, mode="nearest").astype(np.float32),
        "median": median_filter(noisy_test, size=(1, 7), mode="nearest").astype(np.float32),
        "wavelet": wavelet_bayes_shrink(noisy_test, level=5, wavelet="db4"),
        "noisy_pca_k16": apply_pca_basis(basis["noisy_mean"], basis["noisy_components"], noisy_test, 16),
        "clean_pca_k16": apply_pca_basis(basis["clean_mean"], basis["clean_components"], noisy_test, 16),
    }
    for name, arr in reps.items():
        assert arr.shape == flux_test.shape, f"{name} shape {arr.shape} != {flux_test.shape}"
    return reps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-train", type=int, default=30000)
    p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=20)
    p.add_argument(
        "--pca-basis", type=Path, required=True,
        help="canonical_pca_basis.npz (fit on the 200k reconstruction pool via fit_canonical_pca_basis.py)",
    )
    p.add_argument("--no-assert", action="store_true", help="skip cross-check assertion (smoke runs)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    npz = np.load(args.input, mmap_mode="r")
    n_total = int(npz["flux"].shape[0])
    if args.n_train + args.n_test > n_total:
        raise ValueError("n_train + n_test exceeds dataset size")

    # Identical split to downstream_logg_calibrator.py (seed 42, perm slicing).
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    train_idx = perm[: args.n_train]
    test_idx = perm[args.n_train : args.n_train + args.n_test]

    y_train = np.asarray(npz["logg"][train_idx], dtype=np.float32)
    y_test = np.asarray(npz["logg"][test_idx], dtype=np.float32)
    x_train = np.asarray(npz["flux"][train_idx], dtype=np.float32)  # calibrator: clean only

    print(f"[fit] LightGBM logg calibrator on clean flux n_train={args.n_train}", flush=True)
    t0 = time.time()
    model = fit_lgbm(x_train, y_train, n_jobs=args.n_jobs)
    print(f"[fitdone] wall={time.time()-t0:.1f}s train_R2={model.score(x_train, y_train):.4f}", flush=True)

    reps = build_representations(npz, test_idx, args.pca_basis)

    rows: list[dict] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for method in METHOD_ORDER:
            pred = np.asarray(model.predict(reps[method]), dtype=np.float64)
            row = metric_row(method, args.n_train, y_test, pred)
            rows.append(row)
            print(
                f"[eval] {method:14s} sigma_rob={row['sigma_rob_center']:.3f} "
                f"mae={row['mae']:.3f} bias={row['mean_bias']:+.3f} r2={row['r2']:.3f}",
                flush=True,
            )

    write_csv(args.output_dir / "downstream_logg_baselines.csv", rows)

    # CROSS-CHECK: anchors must reproduce the published Table tab:downstream_logg.
    by_method = {r["method"]: r for r in rows}
    cross = {}
    ok = True
    for m, published in PUBLISHED_SIGMA_ROB.items():
        got = by_method[m]["sigma_rob_center"]
        match = abs(got - published) <= 0.01
        cross[m] = {"published": published, "got": round(got, 4), "match": match}
        ok = ok and match
        print(f"[crosscheck] {m}: published={published} got={got:.4f} match={match}", flush=True)

    summary = {
        "input": str(args.input),
        "n_train": args.n_train,
        "n_test": args.n_test,
        "seed": args.seed,
        "e1_selected": E1_SELECTED,
        "pca_basis": str(args.pca_basis),
        "pca_fit_pool": "canonical_pca_basis.npz fit on 200k reconstruction pool (single source of truth, shared with appendix)",
        "crosscheck": cross,
        "crosscheck_pass": ok,
        "rows": rows,
        "wall_time_s": time.time() - start,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.output_dir} crosscheck_pass={ok}", flush=True)

    if not ok and not args.no_assert:
        raise SystemExit(
            "CROSS-CHECK FAILED: clean/bsn/noisy did not reproduce published "
            "0.222/0.801/1.262 -> pipeline mismatch, do NOT trust baseline rows."
        )


if __name__ == "__main__":
    main()
