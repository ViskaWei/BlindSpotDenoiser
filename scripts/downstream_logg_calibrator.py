#!/usr/bin/env python3
"""Downstream logg calibrator for BlindSpotDenoiser.

Train a logg regressor on clean spectra only, freeze it, and evaluate the same
calibrator on clean/noisy/denoised/smoothed held-out spectra.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error, r2_score


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
BIAS_LINE_COLOR = "aquamarine"


@dataclass(frozen=True)
class PredictionBlock:
    name: str
    pred: np.ndarray
    residual: np.ndarray


def sigma_rob_zero(residual: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(residual)))


def sigma_rob_center(residual: np.ndarray) -> float:
    med = np.median(residual)
    return float(1.4826 * np.median(np.abs(residual - med)))


def metric_row(model_name: str, n_train: int, method: str, y_true: np.ndarray, pred: np.ndarray) -> dict:
    residual = pred - y_true
    return {
        "model": model_name,
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


def build_boxcar(x: np.ndarray, width: int = 5) -> np.ndarray:
    pad = width // 2
    padded = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        out += padded[:, i : i + x.shape[1]].astype(np.float32)
    out /= float(width)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_test_sets(npz, test_idx: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    y_test = np.asarray(npz["logg"][test_idx], dtype=np.float32)
    noisy = np.asarray(npz["noisy"][test_idx], dtype=np.float32)
    test_sets = {
        "clean": np.asarray(npz["flux"][test_idx], dtype=np.float32),
        "bsn": np.asarray(npz["denoised"][test_idx], dtype=np.float32),
        "noisy": noisy,
        "savgol": savgol_filter(noisy, window_length=11, polyorder=3, axis=1).astype(np.float32),
        "boxcar": build_boxcar(noisy, width=5),
    }
    return y_test, test_sets


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


def bootstrap_main(
    blocks: dict[str, PredictionBlock],
    y_true: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    metric_rows: list[dict] = []
    compare_rows: list[dict] = []

    for boot in range(n_boot):
        idx = rng.integers(0, n, size=n)
        per_method = {}
        for method in METHOD_ORDER:
            pred = blocks[method].pred[idx]
            truth = y_true[idx]
            row = metric_row("lgbm_pixels", n, method, truth, pred)
            row["bootstrap"] = int(boot)
            metric_rows.append(row)
            per_method[method] = row

        noisy = per_method["noisy"]
        for method in ["bsn", "boxcar", "savgol", "clean"]:
            cur = per_method[method]
            compare_rows.append(
                {
                    "bootstrap": int(boot),
                    "comparison": f"{method}_vs_noisy",
                    "delta_sigma_rob_center_noisy_minus_method": noisy["sigma_rob_center"]
                    - cur["sigma_rob_center"],
                    "ratio_sigma_rob_center_method_over_noisy": cur["sigma_rob_center"]
                    / noisy["sigma_rob_center"],
                    "delta_mae_noisy_minus_method": noisy["mae"] - cur["mae"],
                    "ratio_mae_method_over_noisy": cur["mae"] / noisy["mae"],
                    "delta_mean_bias_abs_noisy_minus_method": abs(noisy["mean_bias"])
                    - abs(cur["mean_bias"]),
                    "delta_r2_method_minus_noisy": cur["r2"] - noisy["r2"],
                }
            )
    return metric_rows, compare_rows


def ci_from_bootstrap(rows: list[dict], key: str, method: str) -> tuple[float, float]:
    vals = np.array([float(r[key]) for r in rows if r["method"] == method], dtype=float)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def comparison_ci(rows: list[dict], key: str, comparison: str) -> tuple[float, float, float]:
    vals = np.array([float(r[key]) for r in rows if r["comparison"] == comparison], dtype=float)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def plot_main_figure(
    out_path: Path,
    y_true: np.ndarray,
    blocks: dict[str, PredictionBlock],
    main_rows: list[dict],
    boot_rows: list[dict],
    compare_rows: list[dict],
    sweep_rows: list[dict],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.5), constrained_layout=True)

    # Panel A: residual distributions.
    residuals = [blocks[m].residual for m in METHOD_ORDER]
    parts = axes[0].violinplot(residuals, showmeans=False, showmedians=True, showextrema=False)
    for body, method in zip(parts["bodies"], METHOD_ORDER):
        body.set_facecolor(METHOD_COLORS[method])
        body.set_edgecolor("white")
        body.set_alpha(0.72)
    parts["cmedians"].set_color(BIAS_LINE_COLOR)
    parts["cmedians"].set_linewidth(1.4)
    axes[0].axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.65)
    axes[0].set_xticks(range(1, len(METHOD_ORDER) + 1), [METHOD_LABELS[m] for m in METHOD_ORDER])
    axes[0].set_ylabel(r"Residual $r=\widehat{\log g}-\log g_{\mathrm{true}}$ [dex]")
    lo, hi = np.percentile(np.concatenate(residuals), [1, 99])
    pad = 0.15 * (hi - lo)
    axes[0].set_ylim(lo - pad, hi + pad)
    axes[0].tick_params(axis="x", labelsize=8, rotation=0)
    for tick in axes[0].get_xticklabels():
        tick.set_ha("center")
    for i, method in enumerate(METHOD_ORDER, start=1):
        bias = np.median(blocks[method].residual)
        axes[0].text(
            i,
            axes[0].get_ylim()[1],
            f"{bias:+.2f}",
            ha="center",
            va="top",
            fontsize=7,
            color="#111827",
        )

    # Panel B: robust scatter and MAE with paired-bootstrap CIs.
    x = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.34
    row_by_method = {r["method"]: r for r in main_rows if r["model"] == "lgbm_pixels"}
    for offset, metric, label, hatch in [
        (-width / 2, "sigma_rob_center", r"$\tilde{\sigma}(r)$", ""),
        (width / 2, "mae", r"MAE", "///"),
    ]:
        vals = np.array([row_by_method[m][metric] for m in METHOD_ORDER])
        ci = np.array([ci_from_bootstrap(boot_rows, metric, m) for m in METHOD_ORDER])
        err = np.vstack([vals - ci[:, 0], ci[:, 1] - vals])
        bars = axes[1].bar(
            x + offset,
            vals,
            width,
            yerr=err,
            capsize=2,
            label=label,
            color=[METHOD_COLORS[m] for m in METHOD_ORDER],
            alpha=0.88 if metric == "sigma_rob_center" else 0.52,
            edgecolor="#111827",
            linewidth=0.4,
            hatch=hatch,
        )
        for bar in bars:
            bar.set_linewidth(0.4)
    axes[1].set_xticks(x, [METHOD_LABELS[m] for m in METHOD_ORDER])
    axes[1].set_ylabel(r"Residual error summary [dex]")
    axes[1].set_ylim(0.0, axes[1].get_ylim()[1] * 1.45)
    axes[1].tick_params(axis="x", labelsize=8, rotation=0)
    for tick in axes[1].get_xticklabels():
        tick.set_ha("center")
    metric_legend = [
        Patch(
            facecolor="#9ca3af",
            edgecolor="#111827",
            label=r"solid: $\tilde{\sigma}(r)$",
        ),
        Patch(facecolor="#d1d5db", edgecolor="#111827", hatch="///", label=r"hatched: $\mathrm{mean}|r|$"),
    ]
    axes[1].legend(handles=metric_legend, frameon=False, fontsize=7, loc="upper left")

    # Panel C: train-size stability.
    for method in ["clean", "bsn", "boxcar", "savgol", "noisy"]:
        rows = [
            r
            for r in sweep_rows
            if r["model"] == "lgbm_pixels" and r["method"] == method
        ]
        rows = sorted(rows, key=lambda r: int(r["n_train"]))
        if not rows:
            continue
        ls = "-" if method in {"clean", "bsn", "noisy"} else "--"
        alpha = 1.0 if method in {"clean", "bsn", "noisy"} else 0.6
        axes[2].plot(
            [r["n_train"] for r in rows],
            [r["sigma_rob_center"] for r in rows],
            marker="o",
            ms=4,
            lw=1.8,
            ls=ls,
            alpha=alpha,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("clean training spectra")
    axes[2].set_ylabel(r"$\tilde{\sigma}(r)$ [dex]")
    axes[2].set_xlim(850, 47000)
    y_offsets = {"clean": -0.01, "bsn": -0.02, "boxcar": -0.015, "savgol": 0.015, "noisy": 0.015}
    for method in ["clean", "bsn", "boxcar", "savgol", "noisy"]:
        rows = [
            r
            for r in sweep_rows
            if r["model"] == "lgbm_pixels" and r["method"] == method
        ]
        rows = sorted(rows, key=lambda r: int(r["n_train"]))
        if not rows:
            continue
        last = rows[-1]
        axes[2].text(
            int(last["n_train"]) * 1.08,
            float(last["sigma_rob_center"]) + y_offsets[method],
            METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            fontsize=7,
            ha="left",
            va="center",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[1000, 3000, 10000, 30000])
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260504)
    parser.add_argument("--n-jobs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    npz = np.load(args.input, mmap_mode="r")
    n_total = int(npz["flux"].shape[0])
    n_pixels = int(npz["flux"].shape[1])
    max_train = max(args.train_sizes)
    if max_train + args.n_test > n_total:
        raise ValueError("max(train_sizes) + n_test exceeds dataset size")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    test_idx = perm[max_train : max_train + args.n_test]
    y_test, test_sets = load_test_sets(npz, test_idx)

    metrics_rows: list[dict] = []
    main_blocks: dict[str, PredictionBlock] | None = None
    main_rows: list[dict] | None = None

    for n_train in args.train_sizes:
        train_idx = perm[:n_train]
        x_train = np.asarray(npz["flux"][train_idx], dtype=np.float32)
        y_train = np.asarray(npz["logg"][train_idx], dtype=np.float32)
        print(f"[fit] lgbm_pixels n_train={n_train}", flush=True)
        t0 = time.time()
        model = fit_lgbm(x_train, y_train, n_jobs=args.n_jobs)
        train_r2 = float(model.score(x_train, y_train))
        print(f"[fitdone] n_train={n_train} wall={time.time() - t0:.1f}s train_R2={train_r2:.4f}", flush=True)

        blocks: dict[str, PredictionBlock] = {}
        rows_this_train: list[dict] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for method in METHOD_ORDER:
                pred = np.asarray(model.predict(test_sets[method]), dtype=np.float64)
                residual = pred - y_test
                blocks[method] = PredictionBlock(method, pred=pred, residual=residual)
                row = metric_row("lgbm_pixels", n_train, method, y_test, pred)
                row["train_r2"] = train_r2
                rows_this_train.append(row)
        metrics_rows.extend(rows_this_train)

        if n_train == max_train:
            main_blocks = blocks
            main_rows = rows_this_train

    assert main_blocks is not None and main_rows is not None

    print(f"[bootstrap] n={args.bootstrap}", flush=True)
    boot_rows, compare_rows = bootstrap_main(
        main_blocks, y_test, n_boot=args.bootstrap, seed=args.bootstrap_seed
    )

    pred_rows: list[dict] = []
    for i, source_idx in enumerate(test_idx):
        for method in METHOD_ORDER:
            pred_rows.append(
                {
                    "test_row": int(i),
                    "source_index": int(source_idx),
                    "method": method,
                    "logg_true": float(y_test[i]),
                    "logg_pred": float(main_blocks[method].pred[i]),
                    "residual": float(main_blocks[method].residual[i]),
                }
            )

    write_csv(args.output_dir / "metrics_by_train_size.csv", metrics_rows)
    write_csv(args.output_dir / "bootstrap_metrics_main.csv", boot_rows)
    write_csv(args.output_dir / "bootstrap_comparisons_main.csv", compare_rows)
    write_csv(args.output_dir / "predictions_main.csv", pred_rows)

    figure_path = args.output_dir / "downstream_logg_calibrator.pdf"
    plot_main_figure(
        figure_path,
        y_true=y_test,
        blocks=main_blocks,
        main_rows=main_rows,
        boot_rows=boot_rows,
        compare_rows=compare_rows,
        sweep_rows=metrics_rows,
    )

    bsn_ratio, bsn_ratio_lo, bsn_ratio_hi = comparison_ci(
        compare_rows, "ratio_sigma_rob_center_method_over_noisy", "bsn_vs_noisy"
    )
    bsn_mae_ratio, bsn_mae_lo, bsn_mae_hi = comparison_ci(
        compare_rows, "ratio_mae_method_over_noisy", "bsn_vs_noisy"
    )
    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "bootstrap_seed": args.bootstrap_seed,
        "n_total": n_total,
        "n_pixels": n_pixels,
        "train_sizes": args.train_sizes,
        "n_test": args.n_test,
        "max_train": max_train,
        "protocol": "train logg calibrator on clean flux only; freeze; evaluate held-out clean/bsn/noisy/savgol/boxcar",
        "headline": {
            "comparison": "bsn_vs_noisy",
            "sigma_rob_center_ratio_mean": bsn_ratio,
            "sigma_rob_center_ratio_ci95": [bsn_ratio_lo, bsn_ratio_hi],
            "mae_ratio_mean": bsn_mae_ratio,
            "mae_ratio_ci95": [bsn_mae_lo, bsn_mae_hi],
        },
        "main_metrics": main_rows,
        "wall_time_s": time.time() - start,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = f"""# Downstream logg calibrator result

This directory contains the main downstream-parameter evidence for BlindSpotDenoiser.

Protocol: train a LightGBM logg calibrator on clean spectra only, freeze it, and evaluate the same held-out spectra as clean, BSN denoised, noisy, Savitzky-Golay smoothed noisy, and boxcar-smoothed noisy.

Headline: BSN/noisy centered robust-scatter ratio = {bsn_ratio:.3f} [{bsn_ratio_lo:.3f}, {bsn_ratio_hi:.3f}], and BSN/noisy MAE ratio = {bsn_mae_ratio:.3f} [{bsn_mae_lo:.3f}, {bsn_mae_hi:.3f}] from paired bootstrap over held-out spectra.

Files:
- `downstream_logg_calibrator.pdf` / `.png`: three-panel result figure.
- `metrics_by_train_size.csv`: LightGBM metrics for each train size and input type.
- `bootstrap_metrics_main.csv`: paired-bootstrap metric samples for the max-train run.
- `bootstrap_comparisons_main.csv`: paired-bootstrap BSN/smoother/clean comparisons against noisy.
- `predictions_main.csv`: held-out predictions and residuals for the max-train run.
- `summary.json`: provenance and headline statistics.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"[done] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
