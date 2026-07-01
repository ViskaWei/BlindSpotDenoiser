#!/usr/bin/env python3
"""Run downstream clean-trained calibrator sweeps.

This script complements ``downstream_logg_calibrator.py`` with broad robustness
checks: multiple random splits, multiple stellar labels, train-size sweeps, and
SNR-bin diagnostics. The core protocol is unchanged: train on clean spectra,
freeze the calibrator, and evaluate degraded held-out inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = Path(
    "/datascope/subaru/user/swei20/blindspot_rv/inputs/bosz50000_v3ep190/"
    "b1_l9_e48_k25_s1_bn1_d1_T0_S0_L0_snr3_b50000_rv500_ep5000_"
    "N50000_m0_v3ep190_N50000.npz"
)

METHOD_ORDER = ["clean", "bsn", "boxcar", "savgol", "noisy"]
METHOD_LABELS = {
    "clean": "Clean",
    "bsn": "BSN",
    "boxcar": "Boxcar",
    "savgol": "SavGol",
    "noisy": "Noisy",
}
METHOD_COLORS = {
    "clean": "dodgerblue",
    "bsn": "navy",
    "boxcar": "darkolivegreen",
    "savgol": "darkgoldenrod",
    "noisy": "black",
}


@dataclass(frozen=True)
class TaskConfig:
    input_path: str
    label: str
    seed: int
    train_sizes: tuple[int, ...]
    n_test: int
    n_jobs: int
    snr_bins: int


def sigma_rob_zero(residual: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(residual)))


def sigma_rob_center(residual: np.ndarray) -> float:
    med = np.median(residual)
    return float(1.4826 * np.median(np.abs(residual - med)))


def metric_row(
    *,
    label: str,
    seed: int,
    model: str,
    n_train: int,
    method: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    extra: dict | None = None,
) -> dict:
    residual = pred - y_true
    row = {
        "label": label,
        "seed": int(seed),
        "model": model,
        "n_train": int(n_train),
        "method": method,
        "std": float(np.std(residual, ddof=0)),
        "sigma_rob_zero": sigma_rob_zero(residual),
        "sigma_rob_center": sigma_rob_center(residual),
        "mae": float(mean_absolute_error(y_true, pred)),
        "mean_bias": float(np.mean(residual)),
        "median_bias": float(np.median(residual)),
        "r2": float(r2_score(y_true, pred)),
        "n_test": int(y_true.shape[0]),
    }
    if extra:
        row.update(extra)
    return row


def build_boxcar(x: np.ndarray, width: int = 5) -> np.ndarray:
    pad = width // 2
    padded = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        out += padded[:, i : i + x.shape[1]].astype(np.float32)
    out /= float(width)
    return out


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


def load_test_sets(npz, test_idx: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    noisy = np.asarray(npz["noisy"][test_idx], dtype=np.float32)
    test_sets = {
        "clean": np.asarray(npz["flux"][test_idx], dtype=np.float32),
        "bsn": np.asarray(npz["denoised"][test_idx], dtype=np.float32),
        "noisy": noisy,
        "savgol": savgol_filter(noisy, window_length=11, polyorder=3, axis=1).astype(np.float32),
        "boxcar": build_boxcar(noisy, width=5),
    }
    return noisy, test_sets


def quantile_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def run_label_seed_task(cfg: TaskConfig) -> dict:
    start = time.time()
    npz = np.load(cfg.input_path, mmap_mode="r")
    n_total = int(npz["flux"].shape[0])
    max_train = max(cfg.train_sizes)
    if max_train + cfg.n_test > n_total:
        raise ValueError("max train size plus test size exceeds dataset size")

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_total)
    test_idx = perm[max_train : max_train + cfg.n_test]
    y_test = np.asarray(npz[cfg.label][test_idx], dtype=np.float64)
    snr_noisy = np.asarray(npz["snr_noisy_db"][test_idx], dtype=np.float64)
    _, test_sets = load_test_sets(npz, test_idx)

    metric_rows: list[dict] = []
    snr_rows: list[dict] = []
    max_train_predictions: dict[str, np.ndarray] = {}

    for n_train in cfg.train_sizes:
        train_idx = perm[:n_train]
        x_train = np.asarray(npz["flux"][train_idx], dtype=np.float32)
        y_train = np.asarray(npz[cfg.label][train_idx], dtype=np.float64)
        model = fit_lgbm(x_train, y_train, n_jobs=cfg.n_jobs)
        train_r2 = float(model.score(x_train, y_train))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            preds = {
                method: np.asarray(model.predict(test_sets[method]), dtype=np.float64)
                for method in METHOD_ORDER
            }
        for method, pred in preds.items():
            metric_rows.append(
                metric_row(
                    label=cfg.label,
                    seed=cfg.seed,
                    model="lgbm_pixels",
                    n_train=n_train,
                    method=method,
                    y_true=y_test,
                    pred=pred,
                    extra={"train_r2": train_r2},
                )
            )
        if n_train == max_train:
            max_train_predictions = preds

    edges = quantile_bin_edges(snr_noisy, cfg.snr_bins)
    for b in range(cfg.snr_bins):
        mask = (snr_noisy >= edges[b]) & (snr_noisy < edges[b + 1])
        if not np.any(mask):
            continue
        for method, pred in max_train_predictions.items():
            snr_rows.append(
                metric_row(
                    label=cfg.label,
                    seed=cfg.seed,
                    model="lgbm_pixels",
                    n_train=max_train,
                    method=method,
                    y_true=y_test[mask],
                    pred=pred[mask],
                    extra={
                        "snr_bin": int(b),
                        "snr_noisy_db_lo": float(np.min(snr_noisy[mask])),
                        "snr_noisy_db_hi": float(np.max(snr_noisy[mask])),
                    },
                )
            )

    return {
        "label": cfg.label,
        "seed": cfg.seed,
        "metric_rows": metric_rows,
        "snr_rows": snr_rows,
        "wall_time_s": time.time() - start,
    }


def run_logg_sanity(input_path: Path, output_dir: Path, n_train: int, n_test: int, seed: int) -> list[dict]:
    npz = np.load(input_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(int(npz["flux"].shape[0]))
    train_idx = perm[:n_train]
    test_idx = perm[n_train : n_train + n_test]
    x_train = np.asarray(npz["flux"][train_idx], dtype=np.float32)
    y_train = np.asarray(npz["logg"][train_idx], dtype=np.float64)
    y_test = np.asarray(npz["logg"][test_idx], dtype=np.float64)
    _, test_sets = load_test_sets(npz, test_idx)

    sanity_models = {
        "ridge_a1000": make_pipeline(StandardScaler(), Ridge(alpha=1000, random_state=0)),
        "pca256_ridge100": make_pipeline(
            StandardScaler(),
            PCA(n_components=256, random_state=0),
            Ridge(alpha=100, random_state=0),
        ),
    }
    rows: list[dict] = []
    for model_name, model in sanity_models.items():
        t0 = time.time()
        model.fit(x_train, y_train)
        train_r2 = float(model.score(x_train, y_train))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for method in METHOD_ORDER:
                pred = np.asarray(model.predict(test_sets[method]), dtype=np.float64)
                rows.append(
                    metric_row(
                        label="logg",
                        seed=seed,
                        model=model_name,
                        n_train=n_train,
                        method=method,
                        y_true=y_test,
                        pred=pred,
                        extra={"train_r2": train_r2, "fit_wall_time_s": time.time() - t0},
                    )
                )
    write_csv(output_dir / "logg_sanity_models.csv", rows)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for row in rows:
        k = tuple(row[key] for key in keys)
        out.setdefault(k, []).append(row)
    return out


def aggregate_seed_summary(rows: list[dict], max_train: int) -> list[dict]:
    rows = [r for r in rows if int(r["n_train"]) == max_train and r["model"] == "lgbm_pixels"]
    by = group_rows(rows, ("label", "method"))
    out: list[dict] = []
    for (label, method), items in sorted(by.items()):
        vals = {metric: np.array([float(r[metric]) for r in items]) for metric in ["sigma_rob_center", "mae", "mean_bias", "r2"]}
        row = {"label": label, "method": method, "n_train": max_train, "n_seeds": len(items)}
        for metric, arr in vals.items():
            row[f"{metric}_mean"] = float(np.mean(arr))
            row[f"{metric}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        out.append(row)

    noisy_by_label = {r["label"]: r for r in out if r["method"] == "noisy"}
    for row in out:
        noisy = noisy_by_label.get(row["label"])
        if noisy:
            row["sigma_rob_center_ratio_vs_noisy"] = row["sigma_rob_center_mean"] / noisy["sigma_rob_center_mean"]
            row["mae_ratio_vs_noisy"] = row["mae_mean"] / noisy["mae_mean"]
            row["delta_r2_vs_noisy"] = row["r2_mean"] - noisy["r2_mean"]
    return out


def plot_label_summary(path: Path, summary_rows: list[dict]) -> None:
    labels = sorted({r["label"] for r in summary_rows})
    methods = ["bsn", "boxcar", "savgol"]
    x = np.arange(len(labels), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    for i, method in enumerate(methods):
        rows = [next(r for r in summary_rows if r["label"] == lab and r["method"] == method) for lab in labels]
        vals = [r["sigma_rob_center_ratio_vs_noisy"] for r in rows]
        ax.bar(x + (i - 1) * width, vals, width=width, label=METHOD_LABELS[method], color=METHOD_COLORS[method], alpha=0.85)
    ax.axhline(1.0, color="#111827", lw=0.8, ls="--")
    ax.set_xticks(x, labels)
    ax.set_ylabel("robust scatter ratio vs noisy")
    ax.set_title("Downstream calibrator gain by stellar label")
    ax.legend(frameon=False)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_logg_train_size(path: Path, rows: list[dict]) -> None:
    rows = [r for r in rows if r["label"] == "logg" and r["model"] == "lgbm_pixels"]
    train_sizes = sorted({int(r["n_train"]) for r in rows})
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    for method in METHOD_ORDER:
        means = []
        stds = []
        for n_train in train_sizes:
            vals = [float(r["sigma_rob_center"]) for r in rows if r["method"] == method and int(r["n_train"]) == n_train]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ls = "-" if method in {"clean", "bsn", "noisy"} else "--"
        alpha = 1.0 if method in {"clean", "bsn", "noisy"} else 0.65
        ax.errorbar(train_sizes, means, yerr=stds, marker="o", lw=1.8, ls=ls, alpha=alpha, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ax.set_xscale("log")
    ax.set_xlabel("clean training spectra")
    ax.set_ylabel("logg robust scatter [dex]")
    ax.set_title("Multi-seed logg train-size stability")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_logg_snr_bins(path: Path, rows: list[dict], max_train: int) -> None:
    rows = [r for r in rows if r["label"] == "logg" and int(r["n_train"]) == max_train]
    bins = sorted({int(r["snr_bin"]) for r in rows})
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    for method in ["clean", "bsn", "boxcar", "savgol", "noisy"]:
        means = []
        stds = []
        labels = []
        for b in bins:
            items = [r for r in rows if int(r["snr_bin"]) == b and r["method"] == method]
            vals = np.array([float(r["sigma_rob_center"]) for r in items])
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            lo = np.mean([float(r["snr_noisy_db_lo"]) for r in items])
            hi = np.mean([float(r["snr_noisy_db_hi"]) for r in items])
            labels.append(f"{lo:.1f}..{hi:.1f}")
        ls = "-" if method in {"clean", "bsn", "noisy"} else "--"
        alpha = 1.0 if method in {"clean", "bsn", "noisy"} else 0.65
        ax.errorbar(bins, means, yerr=stds, marker="o", lw=1.8, ls=ls, alpha=alpha, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ax.set_xticks(bins, labels, rotation=20, ha="right")
    ax.set_xlabel("held-out noisy SNR bin [dB]")
    ax.set_ylabel("logg robust scatter [dex]")
    ax.set_title("Logg downstream gain by SNR bin")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["logg", "teff", "mh", "rv"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[1000, 3000, 10000, 30000])
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--snr-bins", type=int, default=5)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--skip-sanity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    tasks = [
        TaskConfig(
            input_path=str(args.input),
            label=label,
            seed=seed,
            train_sizes=tuple(args.train_sizes),
            n_test=args.n_test,
            n_jobs=args.n_jobs,
            snr_bins=args.snr_bins,
        )
        for label in args.labels
        for seed in args.seeds
    ]
    print(f"[sweep] tasks={len(tasks)} workers={args.n_workers} lgbm_threads={args.n_jobs}", flush=True)

    metric_rows: list[dict] = []
    snr_rows: list[dict] = []
    task_summaries: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = [ex.submit(run_label_seed_task, task) for task in tasks]
        for fut in as_completed(futs):
            result = fut.result()
            metric_rows.extend(result["metric_rows"])
            snr_rows.extend(result["snr_rows"])
            task_summaries.append({k: result[k] for k in ["label", "seed", "wall_time_s"]})
            print(f"[done-task] {result['label']} seed={result['seed']} wall={result['wall_time_s']:.1f}s", flush=True)

    max_train = max(args.train_sizes)
    write_csv(args.output_dir / "lgbm_sweep_metrics.csv", metric_rows)
    write_csv(args.output_dir / "lgbm_snr_bin_metrics.csv", snr_rows)
    seed_summary = aggregate_seed_summary(metric_rows, max_train=max_train)
    write_csv(args.output_dir / "lgbm_seed_summary.csv", seed_summary)

    sanity_rows: list[dict] = []
    if not args.skip_sanity:
        print("[sanity] logg ridge/pca", flush=True)
        sanity_rows = run_logg_sanity(args.input, args.output_dir, max_train, args.n_test, seed=42)

    plot_label_summary(args.output_dir / "label_gain_summary.pdf", seed_summary)
    plot_logg_train_size(args.output_dir / "logg_train_size_stability_multiseed.pdf", metric_rows)
    plot_logg_snr_bins(args.output_dir / "logg_snr_bin_stability_multiseed.pdf", snr_rows, max_train=max_train)

    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "labels": args.labels,
        "seeds": args.seeds,
        "train_sizes": args.train_sizes,
        "n_test": args.n_test,
        "snr_bins": args.snr_bins,
        "n_workers": args.n_workers,
        "n_jobs": args.n_jobs,
        "protocol": "train LightGBM on clean spectra only; freeze; evaluate held-out clean/bsn/noisy/savgol/boxcar",
        "task_summaries": task_summaries,
        "sanity_models": ["ridge_a1000", "pca256_ridge100"] if sanity_rows else [],
        "wall_time_s": time.time() - start,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = f"""# Downstream Calibrator Supplemental Sweeps

Protocol: clean-trained frozen LightGBM calibrator evaluated on held-out clean, BSN-denoised, noisy, Savitzky-Golay, and boxcar spectra.

Sweeps:
- labels: `{', '.join(args.labels)}`
- seeds: `{', '.join(map(str, args.seeds))}`
- train sizes: `{', '.join(map(str, args.train_sizes))}`
- held-out spectra per split: `{args.n_test}`

Files:
- `lgbm_sweep_metrics.csv`: all label/seed/train-size/method metrics.
- `lgbm_seed_summary.csv`: max-train multi-seed means and ratios versus noisy.
- `lgbm_snr_bin_metrics.csv`: max-train metrics by held-out noisy-SNR quantile bin.
- `label_gain_summary.pdf` / `.png`: BSN/smoother robust-scatter ratios against noisy by label.
- `logg_train_size_stability_multiseed.pdf` / `.png`: multi-seed logg train-size stability.
- `logg_snr_bin_stability_multiseed.pdf` / `.png`: logg SNR-bin stability.
- `logg_sanity_models.csv`: Ridge/PCA-Ridge sanity calibrators for logg.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"[done] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
