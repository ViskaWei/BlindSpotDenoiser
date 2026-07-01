#!/usr/bin/env python3
"""Plot the downstream clean-domain transfer figure candidate.

This script intentionally writes only to results/ and does not edit paper files.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


METHOD_COLORS = {
    "clean": "#2D8CFF",
    "bsn": "#0B1B8F",
    "boxcar": "#6F8A58",
    "savgol": "#C29A24",
    "noisy": "#202020",
}
METHOD_MARKERS = {
    "clean": "o",
    "bsn": "o",
    "boxcar": "s",
    "savgol": "^",
}
METHOD_LABELS = {
    "clean": "Clean",
    "bsn": "BSN",
    "boxcar": "Box-car",
    "savgol": "SavGol",
    "noisy": "Noisy",
}
LABEL_ORDER = ["teff", "logg", "mh", "rv"]
LABEL_TITLES = {
    "teff": r"$T_{\rm eff}$",
    "logg": r"$\log g$",
    "mh": r"$[\mathrm{M/H}]$",
    "rv": r"RV",
}
RATIO_METHODS = ["clean", "bsn", "boxcar", "savgol"]
RESIDUAL_ORDER = ["noisy", "savgol", "boxcar", "bsn", "clean"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grouped(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[tuple(row[key] for key in keys)].append(row)
    return dict(out)


def ratio_by_label_seed_method(sweep_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    max_train = max(int(row["n_train"]) for row in sweep_rows if row["model"] == "lgbm_pixels")
    rows = [
        row
        for row in sweep_rows
        if row["model"] == "lgbm_pixels" and int(row["n_train"]) == max_train
    ]
    by = grouped(rows, "label", "seed", "method")
    ratios: dict[tuple[str, str, str], float] = {}
    for label in LABEL_ORDER:
        seeds = sorted({row["seed"] for row in rows if row["label"] == label})
        for seed in seeds:
            noisy = float(by[(label, seed, "noisy")][0]["sigma_rob_center"])
            for method in RATIO_METHODS:
                ratios[(label, seed, method)] = float(
                    by[(label, seed, method)][0]["sigma_rob_center"]
                ) / noisy
    return ratios


def draw_protocol_strip(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.text(
        0.01,
        0.92,
        "A. Clean-domain transfer protocol",
        ha="left",
        va="top",
        fontsize=9.8,
        transform=ax.transAxes,
    )
    boxes = [
        ("Clean BOSZ spectra\n+ labels", 0.08),
        ("Train full-spectrum\nLightGBM calibrator", 0.31),
        ("Freeze calibrator", 0.52),
        ("Evaluate same 5,000\nheld-out spectra", 0.71),
        (r"Residuals: $\hat{y}-y_{\rm true}$", 0.91),
    ]
    y = 0.46
    for text, x in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.8),
            transform=ax.transAxes,
        )
    for x0, x1 in [(0.18, 0.23), (0.40, 0.46), (0.58, 0.63), (0.80, 0.86)]:
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=0.9, color="#475569"),
        )
    ax.text(
        0.5,
        0.03,
        "trained only on clean spectra; frozen before noisy, smoothed, or BSN-denoised evaluation",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#334155",
        transform=ax.transAxes,
    )


def draw_ratio_panel(ax: plt.Axes, ratios: dict[tuple[str, str, str], float]) -> None:
    base_y = np.arange(len(LABEL_ORDER))[::-1].astype(float)
    offsets = {"clean": 0.24, "bsn": 0.08, "boxcar": -0.08, "savgol": -0.24}
    rng = np.random.default_rng(123)

    for label_idx, label in enumerate(LABEL_ORDER):
        y0 = base_y[label_idx]
        seeds = sorted({seed for lab, seed, _method in ratios if lab == label})
        for method in RATIO_METHODS:
            values = np.array([ratios[(label, seed, method)] for seed in seeds], dtype=float)
            ys = y0 + offsets[method] + rng.normal(0, 0.012, size=len(values))
            ax.scatter(
                values,
                ys,
                s=18,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                alpha=0.26,
                linewidths=0,
                zorder=2,
            )
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ax.errorbar(
                mean,
                y0 + offsets[method],
                xerr=sd,
                fmt=METHOD_MARKERS[method],
                ms=7.0,
                color=METHOD_COLORS[method],
                mec="#111827",
                mew=0.35,
                capsize=2.2,
                lw=1.0,
                zorder=4,
            )
            if method == "bsn":
                ax.text(
                    mean + 0.035,
                    y0 + offsets[method],
                    f"{mean:.2f}",
                    color=METHOD_COLORS[method],
                    va="center",
                    ha="left",
                    fontsize=8.4,
                    fontweight="bold",
                )

    ax.axvline(1.0, color="#111827", ls="--", lw=0.9, alpha=0.72)
    ax.text(1.0, base_y[0] + 0.58, "raw noisy = 1", ha="right", va="center", fontsize=7.8)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(-0.65, len(LABEL_ORDER) - 0.35)
    ax.set_yticks(base_y, [LABEL_TITLES[label] for label in LABEL_ORDER])
    ax.set_xlabel(r"$\sigma_{\rm rob,center}({\rm input}) / \sigma_{\rm rob,center}({\rm noisy})$")
    ax.set_title("B. Frozen clean-trained proxy errors")
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method],
            color="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="#111827",
            markeredgewidth=0.35,
            markersize=6.5,
            label=METHOD_LABELS[method],
        )
        for method in RATIO_METHODS
    ]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=8)
    ax.text(0.02, -0.52, "small points: split seeds; large points: seed mean +/- SD", fontsize=7.5, color="#475569")


def draw_residual_panel(ax: plt.Axes, prediction_rows: list[dict[str, str]]) -> None:
    by_method = grouped(prediction_rows, "method")
    residuals = [
        np.array([float(row["residual"]) for row in by_method[(method,)]], dtype=float)
        for method in RESIDUAL_ORDER
    ]
    positions = np.arange(len(RESIDUAL_ORDER))[::-1] + 1
    parts = ax.violinplot(
        residuals,
        positions=positions,
        vert=False,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, method in zip(parts["bodies"], RESIDUAL_ORDER):
        body.set_facecolor(METHOD_COLORS[method])
        body.set_edgecolor("white")
        body.set_alpha(0.76)
    parts["cmedians"].set_color("#7FFFD4")
    parts["cmedians"].set_linewidth(1.4)
    ax.axvline(0, color="#111827", lw=0.8, ls="--", alpha=0.68)
    ax.set_yticks(positions, [METHOD_LABELS[m] for m in RESIDUAL_ORDER])
    ax.set_xlabel(r"$\widehat{\log g}-\log g_{\rm true}$ [dex]")
    ax.set_title(r"C. $\log g$ residuals for one frozen calibrator")
    ax.set_xlim(-3.0, 3.0)
    for pos, method, vals in zip(positions, RESIDUAL_ORDER, residuals):
        sigma = 1.4826 * np.median(np.abs(vals - np.median(vals)))
        ax.text(2.88, pos, rf"$\sigma_{{rob}}={sigma:.3f}$", ha="right", va="center", fontsize=7.6)


def draw_metric_box(ax: plt.Axes) -> None:
    ax.axis("off")
    text = "\n".join(
        [
            r"$\log g$ fixed split",
            r"$\sigma_{\rm rob}$: noisy 1.262 dex",
            r"BSN 0.801 dex",
            r"smoothers 1.087-1.088 dex",
            r"clean reference 0.222 dex",
            "",
            r"BSN/noisy:",
            r"$\sigma_{\rm rob}$ ratio 0.636 [0.609, 0.662]",
            r"MAE ratio 0.700 [0.683, 0.717]",
            r"$R^2$: noisy 0.011, BSN 0.463, clean 0.940",
        ]
    )
    ax.text(
        0.02,
        0.98,
        text,
        ha="left",
        va="top",
        fontsize=8.1,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.8),
        transform=ax.transAxes,
    )


def build_figure(logg_dir: Path, sweep_dir: Path, out_path: Path) -> None:
    prediction_rows = read_csv(logg_dir / "predictions_main.csv")
    sweep_rows = read_csv(sweep_dir / "lgbm_sweep_metrics.csv")
    ratios = ratio_by_label_seed_method(sweep_rows)

    plt.rcParams.update(
        {
            "font.size": 8.7,
            "axes.titlesize": 10.6,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    fig = plt.figure(figsize=(8.8, 7.15), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.78, 2.25, 1.85], width_ratios=[1.08, 1.0, 0.82])
    ax_protocol = fig.add_subplot(gs[0, :])
    ax_ratio = fig.add_subplot(gs[1, :])
    ax_resid = fig.add_subplot(gs[2, :2])
    ax_box = fig.add_subplot(gs[2, 2])

    draw_protocol_strip(ax_protocol)
    draw_ratio_panel(ax_ratio, ratios)
    draw_residual_panel(ax_resid, prediction_rows)
    draw_metric_box(ax_box)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logg-dir",
        type=Path,
        default=Path("results/downstream_logg_calibrator"),
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("results/downstream_calibrator_sweeps"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reproduced_figs/fig4_downstream_clean_domain_transfer.pdf"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figure(args.logg_dir, args.sweep_dir, args.output)


if __name__ == "__main__":
    main()
