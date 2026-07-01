#!/usr/bin/env python3
"""E4 model-free gamma measurement for the BlindSpot stand-alone paper."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA

from scripts.paper_cpu_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MASK_PATH,
    DEFAULT_OUT_ROOT,
    add_noise,
    ensure_dir,
    finite_summary,
    load_mask,
    load_split,
    metric_contract,
    sha256_file,
    write_csv,
    write_json,
)
from src.metrics import freeze_v1 as mfv1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-split", default="train_200k_0")
    parser.add_argument("--val-split", default="test_1k_0")
    parser.add_argument("--test-split", default="test_1k_1")
    parser.add_argument("--mask-path", default=DEFAULT_MASK_PATH)
    parser.add_argument("--out", default=str(Path(DEFAULT_OUT_ROOT) / "e4"))
    parser.add_argument("--pca-fit-n", type=int, default=200_000)
    parser.add_argument("--ridge-fit-n", type=int, default=50_000)
    parser.add_argument("--pca-max-k", type=int, default=256)
    parser.add_argument("--pca-random-state", type=int, default=20260428)
    parser.add_argument("--pca-iterated-power", type=int, default=3)
    return parser.parse_args()


def residual_stats(
    x_clean: np.ndarray,
    pred: np.ndarray,
    error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resid = x_clean.astype(np.float64, copy=False) - pred.astype(np.float64, copy=False)
    tau2 = np.var(resid, axis=0, ddof=1)
    sigma2 = np.mean(error.astype(np.float64, copy=False) ** 2, axis=0)
    gamma = tau2 / np.clip(sigma2, 1e-30, None)
    return tau2, sigma2, gamma


def gamma_summary(gamma: np.ndarray, wave: np.ndarray, prefix: str = "") -> Dict[str, float]:
    row: Dict[str, float] = {}
    stats = finite_summary(gamma)
    for key, value in stats.items():
        row[f"{prefix}gamma_{key}"] = value
    row[f"{prefix}gamma_p95"] = float(np.nanquantile(gamma, 0.95))
    row[f"{prefix}gamma_p99"] = float(np.nanquantile(gamma, 0.99))
    for name, center in zip(mfv1.CA_II_LINE_NAMES, mfv1.CA_II_LINES):
        idx = int(np.argmin(np.abs(wave - center)))
        row[f"{prefix}gamma_{name}"] = float(gamma[idx])
    o2 = (wave >= mfv1.O2_BAND[0]) & (wave <= mfv1.O2_BAND[1])
    row[f"{prefix}gamma_o2_aband_median"] = float(np.nanmedian(gamma[o2]))
    tio = (wave >= 7640.0) & (wave <= 7660.0)
    row[f"{prefix}gamma_tio_7650_median"] = float(np.nanmedian(gamma[tio]))
    return row


def local_cubic_context_predict(x: np.ndarray, wave: np.ndarray, radius: int) -> np.ndarray:
    n, l = x.shape
    pred = np.empty((n, l), dtype=np.float32)
    all_idx = np.arange(l)
    for j in range(l):
        lo = max(0, j - radius)
        hi = min(l, j + radius + 1)
        idx = all_idx[lo:hi]
        idx = idx[idx != j]
        if idx.size < 4:
            pred[:, j] = np.mean(x[:, idx], axis=1).astype(np.float32)
        else:
            spline = CubicSpline(wave[idx], x[:, idx], axis=1, bc_type="natural")
            pred[:, j] = spline(wave[j]).astype(np.float32)
    return pred


def context_pca_predict(pca: PCA, x: np.ndarray, k: int) -> np.ndarray:
    mean = pca.mean_.astype(np.float64, copy=False)
    comps = pca.components_[:k].astype(np.float64, copy=False)  # (k, L)
    z = x.astype(np.float64, copy=False) - mean
    full_coeff = z @ comps.T
    n, l = x.shape
    pred = np.empty((n, l), dtype=np.float32)
    for j in range(l):
        v = comps[:, j]
        s = float(v @ v)
        denom = max(1.0 - s, 1e-10)
        b = full_coeff - z[:, [j]] * v[None, :]
        dot = b @ v
        c = b + (dot[:, None] / denom) * v[None, :]
        pred[:, j] = (mean[j] + c @ v).astype(np.float32)
    return pred


def compute_covariance(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64)
    z = train.astype(np.float64, copy=False) - mean
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    return mean, cov


def ridge_context_predict(
    x: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    radius: int,
    alpha: float,
) -> np.ndarray:
    n, l = x.shape
    pred = np.empty((n, l), dtype=np.float32)
    z = x.astype(np.float64, copy=False) - mean
    all_idx = np.arange(l)
    for j in range(l):
        lo = max(0, j - radius)
        hi = min(l, j + radius + 1)
        idx = all_idx[lo:hi]
        idx = idx[idx != j]
        cxx = cov[np.ix_(idx, idx)].copy()
        cxx.flat[:: cxx.shape[0] + 1] += alpha
        cxy = cov[idx, j]
        beta = np.linalg.solve(cxx, cxy)
        pred[:, j] = (mean[j] + z[:, idx] @ beta).astype(np.float32)
    return pred


def tune_family(
    family: str,
    candidates: Sequence[tuple[str, Dict[str, Any], np.ndarray]],
    x_val: np.ndarray,
    e_val: np.ndarray,
    wave: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best = None
    for label, params, pred in candidates:
        tau2, sigma2, gamma = residual_stats(x_val, pred, e_val)
        row = {
            "family": family,
            "label": label,
            "params_json": json.dumps(params, sort_keys=True),
            "tuning_objective": "min validation median gamma",
            **gamma_summary(gamma, wave),
        }
        rows.append(row)
        score = row["gamma_median"]
        print(f"[E4] tune {label}: median gamma={score:.6g}", flush=True)
        if best is None or score < best["score"]:
            best = {
                "family": family,
                "label": label,
                "params": params,
                "score": score,
            }
    assert best is not None
    return best, rows


def main() -> None:
    args = parse_args()
    start_time = time.time()
    out_dir = ensure_dir(args.out)
    mask = load_mask(args.mask_path)
    print(f"[E4] output: {out_dir}", flush=True)
    print(f"[E4] metric hash: {metric_contract()['metric_code_hash']}", flush=True)

    wave, x_val, e_val, val_meta = load_split(args.data_root, args.val_split, mask=mask)
    _, x_test, e_test, test_meta = load_split(args.data_root, args.test_split, mask=mask)
    _, x_train_pca, _, train_pca_meta = load_split(
        args.data_root, args.train_split, mask=mask, n=args.pca_fit_n
    )
    _, x_train_ridge, _, train_ridge_meta = load_split(
        args.data_root, args.train_split, mask=mask, n=args.ridge_fit_n
    )

    print(f"[E4] fitting PCA on {x_train_pca.shape[0]} clean spectra", flush=True)
    pca = PCA(
        n_components=args.pca_max_k,
        svd_solver="randomized",
        random_state=args.pca_random_state,
        iterated_power=args.pca_iterated_power,
    )
    pca.fit(x_train_pca)

    print(f"[E4] computing ridge covariance on {x_train_ridge.shape[0]} clean spectra", flush=True)
    ridge_mean, ridge_cov = compute_covariance(x_train_ridge)
    del x_train_pca, x_train_ridge

    tuning_rows: list[dict[str, Any]] = []

    local_candidates = []
    for radius in (3, 5, 8, 16, 32):
        label = f"local_cubic_radius{radius}"
        pred = local_cubic_context_predict(x_val, wave, radius)
        local_candidates.append((label, {"radius": radius}, pred))
    best_local, rows = tune_family("local_cubic_spline", local_candidates, x_val, e_val, wave)
    tuning_rows.extend(rows)
    del local_candidates

    pca_candidates = []
    for k in (16, 32, 64, 128, 256):
        label = f"context_pca_k{k}"
        pred = context_pca_predict(pca, x_val, k)
        pca_candidates.append((label, {"k": k}, pred))
    best_pca, rows = tune_family("context_pca", pca_candidates, x_val, e_val, wave)
    tuning_rows.extend(rows)
    del pca_candidates

    ridge_candidates = []
    for radius in (3, 5, 8, 16, 32):
        for alpha in (1e-8, 1e-6, 1e-4, 1e-2, 1.0):
            label = f"local_ridge_radius{radius}_alpha{alpha:g}"
            pred = ridge_context_predict(x_val, ridge_mean, ridge_cov, radius=radius, alpha=alpha)
            ridge_candidates.append((label, {"radius": radius, "alpha": alpha}, pred))
    best_ridge, rows = tune_family("local_ridge", ridge_candidates, x_val, e_val, wave)
    tuning_rows.extend(rows)
    del ridge_candidates

    selected = {
        "local_cubic_spline": best_local,
        "context_pca": best_pca,
        "local_ridge": best_ridge,
    }

    print("[E4] evaluating selected predictors on held-out test split", flush=True)
    pred_local = local_cubic_context_predict(x_test, wave, int(best_local["params"]["radius"]))
    pred_pca = context_pca_predict(pca, x_test, int(best_pca["params"]["k"]))
    pred_ridge = ridge_context_predict(
        x_test,
        ridge_mean,
        ridge_cov,
        radius=int(best_ridge["params"]["radius"]),
        alpha=float(best_ridge["params"]["alpha"]),
    )
    predictions = {
        "local_cubic_spline": pred_local,
        "context_pca": pred_pca,
        "local_ridge": pred_ridge,
        "median_of_three_prediction": np.median(
            np.stack([pred_local, pred_pca, pred_ridge], axis=0), axis=0
        ).astype(np.float32),
    }

    sigma2_ref = None
    npz_payload: Dict[str, Any] = {"wavelength": wave}
    summary_rows = []
    family_gamma = []
    for family, pred in predictions.items():
        tau2, sigma2, gamma = residual_stats(x_test, pred, e_test)
        if sigma2_ref is None:
            sigma2_ref = sigma2
            npz_payload["sigma2_mean_test"] = sigma2
        npz_payload[f"tau2_{family}"] = tau2
        npz_payload[f"gamma_{family}"] = gamma
        row = {
            "family": family,
            "selected_params_json": json.dumps(
                selected.get(family, {"params": {"derived": True}})["params"],
                sort_keys=True,
            ),
            **gamma_summary(gamma, wave),
        }
        summary_rows.append(row)
        if family in ("local_cubic_spline", "context_pca", "local_ridge"):
            family_gamma.append(gamma)

    gamma_stack = np.stack(family_gamma, axis=0)
    gamma_worst = np.nanmax(gamma_stack, axis=0)
    gamma_best_family = np.nanmin(gamma_stack, axis=0)
    npz_payload["gamma_worst_family"] = gamma_worst
    npz_payload["gamma_best_family"] = gamma_best_family
    summary_rows.append({"family": "worst_of_three_families", **gamma_summary(gamma_worst, wave)})
    summary_rows.append({"family": "best_of_three_families", **gamma_summary(gamma_best_family, wave)})

    np.savez_compressed(out_dir / "gamma_per_pixel.npz", **npz_payload)
    write_csv(out_dir / "tuning_grid.csv", tuning_rows)
    write_csv(out_dir / "summary.csv", summary_rows)

    plt.figure(figsize=(10, 5))
    for key in ("gamma_local_cubic_spline", "gamma_context_pca", "gamma_local_ridge", "gamma_median_of_three_prediction"):
        plt.plot(wave, npz_payload[key], linewidth=0.8, label=key.replace("gamma_", ""))
    plt.plot(wave, gamma_worst, linewidth=0.8, linestyle="--", label="worst_family")
    for center in mfv1.CA_II_LINES:
        plt.axvline(center, color="0.4", linewidth=0.5, alpha=0.5)
    plt.axvspan(mfv1.O2_BAND[0], mfv1.O2_BAND[1], color="0.85", alpha=0.5, label="O2 A-band")
    plt.yscale("log")
    plt.xlabel("Wavelength (Angstrom)")
    plt.ylabel(r"$\gamma_\lambda = \tau^2_\lambda / \sigma^2_\lambda$")
    plt.title("E4 model-free context predictability")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "gamma_per_pixel.png", dpi=180)
    plt.savefig(out_dir / "gamma_per_pixel.pdf")
    plt.close()

    write_json(
        out_dir / "manifest.json",
        {
            "experiment": "E4 independent gamma measurement",
            "metric_contract": metric_contract(),
            "data_root": args.data_root,
            "mask_path": args.mask_path,
            "mask_sha256": sha256_file(args.mask_path),
            "splits": {
                "train_pca": train_pca_meta,
                "train_ridge": train_ridge_meta,
                "val": val_meta,
                "test": test_meta,
            },
            "full_file_sha256": {
                "val": sha256_file(Path(args.data_root) / args.val_split / "dataset.h5"),
                "test": sha256_file(Path(args.data_root) / args.test_split / "dataset.h5"),
            },
            "selected": selected,
            "environment": {
                "hostname": platform.node(),
                "python": platform.python_version(),
                "cwd": os.getcwd(),
                "git_head": subprocess.getoutput("git rev-parse HEAD 2>/dev/null"),
            },
            "runtime_sec": round(time.time() - start_time, 3),
        },
    )
    print("[E4] done", flush=True)


if __name__ == "__main__":
    main()
