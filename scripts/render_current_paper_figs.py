#!/usr/bin/env python3
"""Render BlindSpot paper figures from current local assets.

This script intentionally avoids archived `result_image/s*` figures. It uses:

1. Current fixed preview / repeated-noise NPZ bundles from `SpecDenoiser`.
2. Current paper-facing CSV measurements in `BlindSpotDenoiser/paper/data/`.

Outputs are written into `research/BlindSpotDenoiser/paper/figs/`.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
# Noise-ensemble bundle + reproduce script are vendored into this repo so the
# figure pipeline is self-contained (no sibling-repo dependency).
SPEC_ROOT = ROOT
FIG_DIR = ROOT / "reproduced_figs"
DATA_DIR = ROOT / "data"

# DEFAULT_PREVIEW_BUNDLE: idx=336 is the canonical paper representative spectrum
# (mag=22.45, S/N≈3, BSN ckpt-123 epoch=190). All future fig1 re-renders MUST
# use this bundle so the displayed S/N=3 → S/N=124 visual contrast is preserved.
# Generated 2026-04-30 from inference.py --representative-idx 336
# on volta04 (paper canonical config: mu_only_baseline_mag205_225 + ckpt-123).
DEFAULT_PREVIEW_BUNDLE = ROOT / "data" / "fixed_preview" / "bundle_idx336_fig1.npz"
DEFAULT_ENSEMBLE_BUNDLE = SPEC_ROOT / "data" / "fixed_preview" / "idx4_m1a123_noise_ensemble_10k.npz"
# Updated 2026-05-01: superseded `idx4_m1a134_noise_ensemble_10k.npz` (rendered
# from epoch=134-val/snr_mu_x=117.ckpt, the pre-canonical-lock best117 ckpt) is
# replaced by `idx4_m1a123_noise_ensemble_10k.npz` (rendered from canonical
# epoch=190-val/snr_mu_x=123.ckpt). See `paper/results.md` §2.8.7 item 6.


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _snr_linear(reference: np.ndarray, estimate: np.ndarray) -> float:
    residual = reference - estimate
    num = max(float(np.linalg.norm(reference)), 1e-30)
    den = max(float(np.linalg.norm(residual)), 1e-30)
    return num / den


def _snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    residual = reference - estimate
    var_ref = max(float(np.var(reference, ddof=1)), 1e-30)
    var_res = max(float(np.var(residual, ddof=1)), 1e-30)
    return 10.0 * np.log10(var_ref / var_res)


def render_representative(bundle_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Vertical 3-panel representative spectrum.

    Layout (per professor feedback): single-column-friendly, axis/tick labels
    at caption-font scale. Three rows stacked:
      Top:    full noisy + clean.
      Middle: clean + (xhat - offset) on top; residual (clean - xhat) on bottom.
      Bottom: Ca II zoom — clean + (xhat - offset) on top; residual zoom below.

    All axes show ylabel + xlabel; legend is a single horizontal row at the
    figure top. Transparent background preserved.
    """
    bundle = np.load(bundle_path, allow_pickle=True)
    wave = np.asarray(bundle["wave"], dtype=float)
    clean = np.asarray(bundle["clean"], dtype=float)
    noisy = np.asarray(bundle["noisy"], dtype=float)
    xhat = np.asarray(bundle["mu"], dtype=float)
    # Fallback defaults for older `bundle_idx336_fig1.npz` archives that predate
    # the ca_lo/ca_hi/mu_offset metadata fields. Values match the canonical Ca II
    # window and the offset used in fig1 caption (xhat - 0.2).
    files = set(bundle.files)
    ca_lo = float(bundle["ca_lo"]) if "ca_lo" in files else 8480.0
    ca_hi = float(bundle["ca_hi"]) if "ca_hi" in files else 8680.0
    xhat_offset = float(bundle["mu_offset"]) if "mu_offset" in files else 0.2

    ca_mask = (wave >= ca_lo) & (wave <= ca_hi)
    snr_y = _snr_linear(clean, noisy)
    snr_xhat = _snr_linear(clean, xhat)
    snr_xhat_ca = _snr_linear(clean[ca_mask], xhat[ca_mask])
    snr_y_db = _snr_db(clean, noisy)
    snr_xhat_db = _snr_db(clean, xhat)
    snr_xhat_ca_db = _snr_db(clean[ca_mask], xhat[ca_mask])

    colors = PAPER_FIG_COLORS
    alpha_ca = 0.5

    # Caption-font alignment: label/tick at 9 pt, legend at 8 pt (vs aastex
    # caption ~9 pt). Local rcParams scope avoids polluting other renders.
    # Font family: serif (STIX, matches AAS body Times); mathtext fontset 'stix'
    # so $..$ symbols render in matching glyphs instead of matplotlib default
    # DejaVuSans-Oblique. Audited 2026-05-01: previous DejaVuSans embedding
    # produced sans-serif figures clashing with serif body text (figure-quality
    # FAIL).
    rc = {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
    with plt.rc_context(rc):
        # Vertical layout: keep height under aastex single-column textheight
        # (~9 in usable). 7.7 in fits cleanly.
        fig = plt.figure(figsize=(7.0, 7.7), dpi=200)
        fig.patch.set_alpha(0.0)
        outer_gs = fig.add_gridspec(
            3, 1,
            height_ratios=[1.0, 1.4, 1.0],
            hspace=0.5,
        )
        mid_gs = outer_gs[1, 0].subgridspec(2, 1, height_ratios=[1.4, 1.0], hspace=0.0)
        bot_gs = outer_gs[2, 0].subgridspec(2, 1, height_ratios=[1.4, 1.0], hspace=0.0)

        ax_top = fig.add_subplot(outer_gs[0, 0])
        ax_mid_top = fig.add_subplot(mid_gs[0, 0])
        ax_mid_bot = fig.add_subplot(mid_gs[1, 0], sharex=ax_mid_top)
        ax_bot_top = fig.add_subplot(bot_gs[0, 0])
        ax_bot_bot = fig.add_subplot(bot_gs[1, 0], sharex=ax_bot_top)

        for ax in (ax_top, ax_mid_top, ax_mid_bot, ax_bot_top, ax_bot_bot):
            ax.patch.set_alpha(0.0)

        # Top: full noisy + clean
        ax_top.plot(wave, noisy, c=colors["noisy"], lw=0.6, label=r"$y$")
        ax_top.plot(wave, clean, c=colors["clean"], lw=1.2, label=r"$x$")
        ax_top.axvspan(ca_lo, ca_hi, color=colors["highlight"], alpha=alpha_ca)
        ax_top.set_xlim(float(wave.min()), float(wave.max()))
        ax_top.set_ylim(0.0, 1.0)
        ax_top.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
        ax_top.set_ylabel("Normalized Flux")
        ax_top.text(0.99, 0.04,
                    rf"$\mathrm{{S/N}}[y]={snr_y:.0f},\ {snr_y_db:.0f}\,\mathrm{{dB}}$",
                    transform=ax_top.transAxes, ha="right", va="bottom")

        # Middle top: clean + (xhat - offset)
        ax_mid_top.plot(wave, clean, c=colors["clean"], lw=0.85, label=r"$x$")
        ax_mid_top.plot(wave, xhat - xhat_offset, c=colors["xhat"], lw=0.85,
                        label=rf"$\hat{{x}} - {xhat_offset:.1f}$")
        ax_mid_top.axvspan(ca_lo, ca_hi, color=colors["highlight"], alpha=alpha_ca)
        ax_mid_top.set_xlim(float(wave.min()), float(wave.max()))
        ax_mid_top.set_ylabel("Normalized Flux")
        plt.setp(ax_mid_top.get_xticklabels(), visible=False)

        # Middle bottom: residual
        ax_mid_bot.plot(wave, clean - xhat, c=colors["residual"], lw=0.85,
                        label=r"$x-\hat{x}$")
        ax_mid_bot.axvspan(ca_lo, ca_hi, color=colors["highlight"], alpha=alpha_ca)
        ax_mid_bot.set_xlim(float(wave.min()), float(wave.max()))
        ax_mid_bot.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
        ax_mid_bot.set_ylabel("Residual")
        ax_mid_bot.axhline(0.0, color="0.7", lw=0.5, ls=":")
        ax_mid_top.text(0.99, 0.04,
                        rf"$\mathrm{{S/N}}[\hat{{x}}]={snr_xhat:.0f},\ {snr_xhat_db:.0f}\,\mathrm{{dB}}$",
                        transform=ax_mid_top.transAxes, ha="right", va="bottom")

        # Bottom top: Ca II zoom
        zoom_pad = 10.0
        ax_bot_top.plot(wave, clean, c=colors["clean"], lw=0.85)
        ax_bot_top.plot(wave, xhat - xhat_offset, c=colors["xhat"], lw=0.85)
        ax_bot_top.axvspan(ca_lo, ca_hi, color=colors["highlight"], alpha=alpha_ca)
        ax_bot_top.set_xlim(ca_lo - zoom_pad, ca_hi + zoom_pad)
        ax_bot_top.set_ylabel("Normalized Flux")
        plt.setp(ax_bot_top.get_xticklabels(), visible=False)

        # Bottom bottom: residual zoom
        ax_bot_bot.plot(wave, clean - xhat, c=colors["residual"], lw=0.85)
        ax_bot_bot.axvspan(ca_lo, ca_hi, color=colors["highlight"], alpha=alpha_ca)
        ax_bot_bot.set_xlim(ca_lo - zoom_pad, ca_hi + zoom_pad)
        ax_bot_bot.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
        ax_bot_bot.set_ylabel("Residual")
        ax_bot_bot.axhline(0.0, color="0.7", lw=0.5, ls=":")
        ax_bot_top.text(0.99, 0.04,
                        rf"$\mathrm{{S/N}}[\hat{{x}}_{{\mathrm{{Ca\,II}}}}]={snr_xhat_ca:.0f},\ {snr_xhat_ca_db:.0f}\,\mathrm{{dB}}$",
                        transform=ax_bot_top.transAxes, ha="right", va="bottom")

        # Single horizontal legend at top of figure
        legend_handles = [
            Line2D([], [], color=colors["noisy"], lw=2.0, label=r"$y$"),
            Line2D([], [], color=colors["clean"], lw=2.0, label=r"$x$"),
            Line2D([], [], color=colors["xhat"], lw=2.0,
                   label=rf"$\hat{{x}} - {xhat_offset:.1f}$"),
            Line2D([], [], color=colors["residual"], lw=2.0, label=r"$x-\hat{x}$"),
            Patch(facecolor=colors["highlight"], edgecolor="none",
                  alpha=alpha_ca, label="Ca II Region"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(legend_handles),
            frameon=False,
            handlelength=1.6,
            columnspacing=0.9,
            handletextpad=0.4,
        )

        fig.subplots_adjust(left=0.105, right=0.985, top=0.955, bottom=0.055)

        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / "fig1_representative.pdf"
        png_path = out_dir / "fig1_representative.png"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, transparent=True)
        fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.02,
                    transparent=True)
        plt.close(fig)
    return pdf_path, png_path


def render_noise_ensemble(bundle_path: Path, out_dir: Path) -> tuple[Path, Path]:
    mod = _load_module(
        SPEC_ROOT / "scripts" / "reproduce_noise_ensemble_diagnostic.py",
        "specdenoiser_reproduce_noise_ensemble_diagnostic",
    )
    return mod.render_bundle(bundle_path, out_dir, "fig2_noise_ensemble")


CA_II_LINE_CENTERS_A = (8498.0, 8542.0, 8662.0)
# Half-window 10 Å (was 12). 12 Å produced 4 Å of empty x-axis on the panel-3
# right edge because the wave grid only reaches ~8670.77 within 8650-8674.
CA_II_HALF_WINDOW_A = 10.0

# Canonical paper-figure palette (matches scripts/inference.py
# PAPER_FIG_COLORS and SpecDenoiser/scripts/make_fig1_blindspot_fixed_preview.py).
PAPER_FIG_COLORS = {
    "noisy": "black",
    "clean": "dodgerblue",
    "xhat": "navy",
    "residual": "seagreen",
    "highlight": "aquamarine",
}


def render_caII_zoom_panels(bundle_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Three-panel zoom on the Ca II triplet for the representative spectrum.

    Each panel overlays the noisy input, the clean reference, and the blindspot
    output around one of the three Ca II line centers, giving a visual companion
    to the per-line numbers in the headline table.
    """
    bundle = np.load(bundle_path, allow_pickle=True)
    wave = np.asarray(bundle["wave"], dtype=float)
    clean = np.asarray(bundle["clean"], dtype=float)
    noisy = np.asarray(bundle["noisy"], dtype=float)
    xhat = np.asarray(bundle["mu"], dtype=float)

    colors = PAPER_FIG_COLORS

    # Typography (locked 2026-05-01 v6 design): serif (STIX) matching fig1 +
    # paper body Times. Sized for 16x4 page-spanning band; xlabel/tick larger
    # than v5 13x3.6 / 15x5 because vertical real estate is compressed.
    rc = {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    }
    with plt.rc_context(rc):
        # 16x4 page-spanning band ("横着铺满, 扁扁的"). wspace=0.04 makes panels
        # hug each other since y-axis is shared (no whitespace tax).
        fig, axes = plt.subplots(
            1, 3, figsize=(16.0, 4.0), dpi=220, sharey=True,
            gridspec_kw={"wspace": 0.04},
        )

        clean_in_panels = []
        xhat_in_panels = []
        for ax, lc in zip(axes, CA_II_LINE_CENTERS_A):
            lo = lc - CA_II_HALF_WINDOW_A
            hi = lc + CA_II_HALF_WINDOW_A
            mask = (wave >= lo) & (wave <= hi)
            ax.plot(wave[mask], noisy[mask], color=colors["noisy"], lw=0.7, alpha=0.55,
                    label=r"$y$")
            ax.plot(wave[mask], clean[mask], color=colors["clean"], lw=1.6, label=r"$x$")
            ax.plot(wave[mask], xhat[mask], color=colors["xhat"], lw=1.2, label=r"$\hat{x}$")
            # Per-bin scatter markers visualize the discrete sampled wavelength bins;
            # the connecting lines above are visual guides only.
            ax.scatter(wave[mask], noisy[mask], color=colors["noisy"],
                       s=8, alpha=0.85, zorder=3)
            ax.scatter(wave[mask], clean[mask], color=colors["clean"],
                       s=10, alpha=0.85, zorder=4)
            ax.scatter(wave[mask], xhat[mask], color=colors["xhat"],
                       s=10, alpha=0.85, zorder=5)
            # axvline at line center removed: stellar RV shift makes it land
            # away from the actual absorption dip, producing a misleading marker.
            wmin = float(wave[mask].min()) if mask.any() else lo
            wmax = float(wave[mask].max()) if mask.any() else hi
            ax.set_xlim(wmin, wmax)
            ax.set_xlabel(r"Wavelength ($\AA$)")
            # Inset title bottom-right (was top axes.set_title): keeps the panel
            # frame clean and sits in the natural baseline whitespace; opaque
            # bbox blocks any noisy-line bleed-through.
            ax.text(0.96, 0.06,
                    rf"Ca II $\lambda${int(lc)} Å",
                    transform=ax.transAxes,
                    fontsize=rc["axes.titlesize"],
                    ha="right", va="bottom",
                    zorder=20,
                    bbox=dict(facecolor="white", alpha=0.95,
                              edgecolor="none", boxstyle="round,pad=0.25"))
            # Sparser ticks ("底下 tick 不要太多") + prune endpoints to avoid
            # overlap with adjacent panel ticks at wspace=0.04.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
            ax.grid(alpha=0.2, linestyle="--", lw=0.5)

            clean_in_panels.append(clean[mask])
            xhat_in_panels.append(xhat[mask])

        # Robust shared y-range from clean+mu only (do NOT let noisy outliers
        # set the range — at S/N=3 a single outlier pushes y to [0, 1.2] and
        # flattens all absorption-line detail, which was the 31c0227 bug).
        ref = np.concatenate(clean_in_panels + xhat_in_panels)
        lo_q = float(np.quantile(ref, 0.005))
        hi_q = float(np.quantile(ref, 0.995))
        pad = 0.06 * max(hi_q - lo_q, 0.05)
        for ax in axes:
            ax.set_ylim(lo_q - pad - 0.02, hi_q + pad)

        axes[0].set_ylabel("Normalized Flux")
        axes[0].legend(frameon=False, loc="lower left", ncol=3,
                       handlelength=1.6, columnspacing=0.9)

        # Manual margin pinning: tight_layout would override gridspec wspace.
        fig.subplots_adjust(left=0.04, right=0.995, top=0.985, bottom=0.18,
                            wspace=0.04)

        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / "fig3_caII_zoom_panels.pdf"
        png_path = out_dir / "fig3_caII_zoom_panels.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preview-bundle", type=Path, default=DEFAULT_PREVIEW_BUNDLE)
    p.add_argument("--ensemble-bundle", type=Path, default=DEFAULT_ENSEMBLE_BUNDLE)
    p.add_argument("--out-dir", type=Path, default=FIG_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pdf1, png1 = render_representative(args.preview_bundle, args.out_dir)
    pdf2, png2 = render_noise_ensemble(args.ensemble_bundle, args.out_dir)
    pdf3, png3 = render_caII_zoom_panels(args.preview_bundle, args.out_dir)
    print(f"[fig1] pdf={pdf1}")
    print(f"[fig1] png={png1}")
    print(f"[fig2] pdf={pdf2}")
    print(f"[fig2] png={png2}")
    print(f"[fig3] pdf={pdf3}")
    print(f"[fig3] png={png3}")


if __name__ == "__main__":
    main()
