#!/usr/bin/env python3
"""E1 CPU baseline battery for the BlindSpot stand-alone paper.

This script tunes classical/PCA baselines on the validation split and evaluates
the selected candidates on the held-out test split with ten fixed noise seeds.
It does not import or run the neural model.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

try:
    import pywt
except Exception as exc:  # pragma: no cover - handled at runtime on elephant6
    pywt = None
    PYWT_IMPORT_ERROR = repr(exc)
else:
    PYWT_IMPORT_ERROR = None

from scripts.paper_cpu_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MASK_PATH,
    DEFAULT_OUT_ROOT,
    DEFAULT_TEST_SEEDS,
    add_noise,
    aggregate_seed_metrics,
    ensure_dir,
    load_mask,
    load_split,
    metric_arrays,
    metric_contract,
    metrics_to_jsonable,
    parse_seed_list,
    scalar_metric_summary,
    sha256_file,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-split", default="train_200k_0")
    parser.add_argument("--val-split", default="test_1k_0")
    parser.add_argument("--test-split", default="test_1k_1")
    parser.add_argument("--mask-path", default=DEFAULT_MASK_PATH)
    parser.add_argument("--out", default=str(Path(DEFAULT_OUT_ROOT) / "e1"))
    parser.add_argument("--pca-fit-n", type=int, default=200_000)
    parser.add_argument("--train-noise-seed", type=int, default=20260428)
    parser.add_argument("--val-noise-seed", type=int, default=42)
    parser.add_argument("--test-seeds", default=",".join(map(str, DEFAULT_TEST_SEEDS)))
    parser.add_argument("--pca-random-state", type=int, default=20260428)
    parser.add_argument("--pca-iterated-power", type=int, default=3)
    parser.add_argument("--skip-pca", action="store_true")
    parser.add_argument("--skip-wavelet", action="store_true")
    return parser.parse_args()


def reconstruct_pca(pca: PCA, x: np.ndarray, k: int) -> np.ndarray:
    mean = pca.mean_.astype(np.float32, copy=False)
    comps = pca.components_[:k].astype(np.float32, copy=False)
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


def candidate_grid(
    *,
    clean_pca: PCA | None,
    noisy_pca: PCA | None,
    skip_wavelet: bool,
) -> Dict[str, list[tuple[str, Dict[str, Any], Callable[[np.ndarray], np.ndarray]]]]:
    grid: Dict[str, list[tuple[str, Dict[str, Any], Callable[[np.ndarray], np.ndarray]]]] = {}

    sg: list[tuple[str, Dict[str, Any], Callable[[np.ndarray], np.ndarray]]] = []
    for window in (3, 5, 7, 9, 11):
        for poly in (1, 2, 3):
            if poly >= window:
                continue
            label = f"savgol_w{window}_p{poly}"
            params = {"window": window, "polyorder": poly}
            sg.append(
                (
                    label,
                    params,
                    lambda y, window=window, poly=poly: savgol_filter(
                        y, window_length=window, polyorder=poly, axis=-1, mode="interp"
                    ).astype(np.float32, copy=False),
                )
            )
    grid["savgol"] = sg

    gauss = []
    for sigma in (1, 2, 3, 5, 7):
        label = f"gaussian_sigma{sigma}"
        gauss.append(
            (
                label,
                {"sigma": sigma},
                lambda y, sigma=sigma: gaussian_filter1d(
                    y, sigma=sigma, axis=-1, mode="nearest"
                ).astype(np.float32, copy=False),
            )
        )
    grid["gaussian"] = gauss

    med = []
    for window in (3, 5, 7):
        label = f"median_w{window}"
        med.append(
            (
                label,
                {"window": window},
                lambda y, window=window: median_filter(
                    y, size=(1, window), mode="nearest"
                ).astype(np.float32, copy=False),
            )
        )
    grid["median"] = med

    if not skip_wavelet:
        wav = []
        for level in (3, 4, 5):
            label = f"wavelet_db4_bayes_l{level}"
            wav.append(
                (
                    label,
                    {"wavelet": "db4", "threshold": "BayesShrink", "level": level},
                    lambda y, level=level: wavelet_bayes_shrink(y, level=level),
                )
            )
        grid["wavelet"] = wav

    if noisy_pca is not None:
        noisy = []
        for k in (16, 32, 64, 128, 256, 512):
            if k > noisy_pca.n_components_:
                continue
            label = f"noisy_pca_k{k}"
            noisy.append(
                (
                    label,
                    {"k": k, "fit_on": "noisy_train"},
                    lambda y, k=k: reconstruct_pca(noisy_pca, y, k),
                )
            )
        grid["noisy_pca"] = noisy

    if clean_pca is not None:
        oracle = []
        for k in (16, 32, 64, 128):
            if k > clean_pca.n_components_:
                continue
            label = f"clean_template_pca_oracle_k{k}"
            oracle.append(
                (
                    label,
                    {"k": k, "fit_on": "clean_train", "oracle": True},
                    lambda y, k=k: reconstruct_pca(clean_pca, y, k),
                )
            )
        grid["clean_template_pca_oracle"] = oracle

    return grid


def fit_pca_models(
    train_flux: np.ndarray,
    train_error: np.ndarray,
    *,
    n_fit: int,
    train_noise_seed: int,
    random_state: int,
    iterated_power: int,
) -> tuple[PCA, PCA]:
    n = min(n_fit, train_flux.shape[0])
    x_train = train_flux[:n].astype(np.float32, copy=False)
    e_train = train_error[:n].astype(np.float32, copy=False)
    y_train = add_noise(x_train, e_train, train_noise_seed)
    noisy_k = min(512, n, x_train.shape[1])
    clean_k = min(128, n, x_train.shape[1])
    print(f"[E1] fitting noisy PCA on {n} spectra, max k={noisy_k}", flush=True)
    noisy_pca = PCA(
        n_components=noisy_k,
        svd_solver="randomized",
        random_state=random_state,
        iterated_power=iterated_power,
    )
    noisy_pca.fit(y_train)
    print(f"[E1] fitting clean-template PCA on {n} spectra, max k={clean_k}", flush=True)
    clean_pca = PCA(
        n_components=clean_k,
        svd_solver="randomized",
        random_state=random_state + 1,
        iterated_power=iterated_power,
    )
    clean_pca.fit(x_train)
    return clean_pca, noisy_pca


def tune_candidates(
    grid: Dict[str, list[tuple[str, Dict[str, Any], Callable[[np.ndarray], np.ndarray]]]],
    x_val: np.ndarray,
    y_val: np.ndarray,
    wave: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[str, dict[str, Any]] = {}
    tuning_rows: list[dict[str, Any]] = []
    for family, candidates in grid.items():
        best = None
        for label, params, fn in candidates:
            start = time.time()
            pred = fn(y_val)
            metrics = metric_arrays(x_val, pred, wave)
            summary = scalar_metric_summary(metrics)
            row = {
                "family": family,
                "label": label,
                "params_json": json.dumps(params, sort_keys=True),
                "tuning_objective": "max validation mean recon_snr_linear",
                "runtime_sec": round(time.time() - start, 3),
                **summary,
            }
            tuning_rows.append(row)
            score = row["recon_snr_linear_mean"]
            print(f"[E1] tune {label}: val recon_snr_linear_mean={score:.6g}", flush=True)
            if best is None or score > best["score"]:
                best = {"family": family, "label": label, "params": params, "fn": fn, "score": score}
        assert best is not None
        selected[family] = best
        print(f"[E1] selected {family}: {best['label']} score={best['score']:.6g}", flush=True)
    return selected, tuning_rows


def evaluate_selected(
    selected: dict[str, dict[str, Any]],
    x_test: np.ndarray,
    error_test: np.ndarray,
    wave: np.ndarray,
    seeds: Iterable[int],
    out_dir: Path,
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    per_dir = ensure_dir(out_dir / "per_baseline_json")

    eval_entries: list[tuple[str, str, Dict[str, Any], Callable[[np.ndarray], np.ndarray]]] = [
        ("noisy", "noisy_input", {"baseline": "noisy"}, lambda y: y)
    ]
    for family, item in selected.items():
        eval_entries.append((family, item["label"], item["params"], item["fn"]))

    noisy_seed_metrics: list[dict[str, np.ndarray]] | None = None
    for family, label, params, fn in eval_entries:
        seed_payloads = []
        seed_metrics: list[dict[str, np.ndarray]] = []
        for seed in seeds:
            y = add_noise(x_test, error_test, int(seed))
            pred = fn(y)
            metrics = metric_arrays(x_test, pred, wave)
            seed_metrics.append(metrics)
            seed_payloads.append({"seed": int(seed), "metrics": metrics_to_jsonable(metrics)})
            print(f"[E1] eval {label} seed={seed}", flush=True)

        if family == "noisy":
            noisy_seed_metrics = seed_metrics

        row: dict[str, Any] = {
            "family": family,
            "label": label,
            "selected_params_json": json.dumps(params, sort_keys=True),
            "n_seeds": len(seed_metrics),
            **aggregate_seed_metrics(seed_metrics),
        }
        if noisy_seed_metrics is not None and family != "noisy":
            deltas: list[dict[str, np.ndarray]] = []
            for baseline_m, noisy_m in zip(seed_metrics, noisy_seed_metrics):
                deltas.append({k: np.asarray(baseline_m[k]) - np.asarray(noisy_m[k]) for k in baseline_m})
            delta_summary = aggregate_seed_metrics(deltas)
            for key, value in delta_summary.items():
                row[f"delta_vs_noisy_{key}"] = value
        summary_rows.append(row)

        json_path = per_dir / f"{label}.json.gz"
        with gzip.open(json_path, "wt", encoding="utf-8") as f:
            json.dump(
                {
                    "family": family,
                    "label": label,
                    "selected_params": params,
                    "metric_contract": metric_contract(),
                    "seeds": seed_payloads,
                },
                f,
                sort_keys=True,
            )
            f.write("\n")
    return summary_rows


def main() -> None:
    args = parse_args()
    start_time = time.time()
    out_dir = ensure_dir(args.out)
    mask = load_mask(args.mask_path)
    seeds = parse_seed_list(args.test_seeds)
    print(f"[E1] output: {out_dir}", flush=True)
    print(f"[E1] metric hash: {metric_contract()['metric_code_hash']}", flush=True)

    wave, x_val, e_val, val_meta = load_split(args.data_root, args.val_split, mask=mask)
    _, x_test, e_test, test_meta = load_split(args.data_root, args.test_split, mask=mask)
    train_meta = None
    clean_pca = noisy_pca = None
    if not args.skip_pca:
        _, x_train, e_train, train_meta = load_split(
            args.data_root, args.train_split, mask=mask, n=args.pca_fit_n
        )
        clean_pca, noisy_pca = fit_pca_models(
            x_train,
            e_train,
            n_fit=args.pca_fit_n,
            train_noise_seed=args.train_noise_seed,
            random_state=args.pca_random_state,
            iterated_power=args.pca_iterated_power,
        )
        del x_train, e_train

    y_val = add_noise(x_val, e_val, args.val_noise_seed)
    grid = candidate_grid(
        clean_pca=clean_pca,
        noisy_pca=noisy_pca,
        skip_wavelet=args.skip_wavelet,
    )
    selected, tuning_rows = tune_candidates(grid, x_val, y_val, wave)
    summary_rows = evaluate_selected(selected, x_test, e_test, wave, seeds, out_dir)

    write_csv(out_dir / "tuning_grid.csv", tuning_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_json(
        out_dir / "manifest.json",
        {
            "experiment": "E1 classical + PCA baseline battery",
            "metric_contract": metric_contract(),
            "data_root": args.data_root,
            "mask_path": args.mask_path,
            "mask_sha256": sha256_file(args.mask_path),
            "splits": {
                "train": train_meta,
                "val": val_meta,
                "test": test_meta,
            },
            "full_file_sha256": {
                "val": sha256_file(Path(args.data_root) / args.val_split / "dataset.h5"),
                "test": sha256_file(Path(args.data_root) / args.test_split / "dataset.h5"),
            },
            "seeds": {
                "train_noise_seed_for_noisy_pca": args.train_noise_seed,
                "val_noise_seed": args.val_noise_seed,
                "test_seeds": list(seeds),
            },
            "pca_fit_n": args.pca_fit_n,
            "selected": {
                family: {
                    "label": item["label"],
                    "params": item["params"],
                    "val_recon_snr_linear_mean": item["score"],
                }
                for family, item in selected.items()
            },
            "environment": {
                "hostname": platform.node(),
                "python": platform.python_version(),
                "cwd": os.getcwd(),
                "git_head": subprocess.getoutput("git rev-parse HEAD 2>/dev/null"),
                "pywavelets_available": pywt is not None,
                "pywavelets_error": PYWT_IMPORT_ERROR,
            },
            "runtime_sec": round(time.time() - start_time, 3),
        },
    )

    lines = [
        "# E1 CPU Baseline Tuning Grid",
        "",
        f"- Metric hash: `{metric_contract()['metric_code_hash']}`",
        f"- Data root: `{args.data_root}`",
        f"- Validation split: `{args.val_split}`, noise seed `{args.val_noise_seed}`",
        f"- Test split: `{args.test_split}`, seeds `{','.join(map(str, seeds))}`",
        "- Tuning objective: maximize validation mean `recon_snr_linear`.",
        "- PCA is fit only on the training split; filters use no fitted test information.",
        "",
        "See `tuning_grid.csv`, `summary.csv`, `manifest.json`, and `per_baseline_json/*.json.gz` in this output directory.",
    ]
    (out_dir / "tuning_grid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[E1] done", flush=True)


if __name__ == "__main__":
    main()
