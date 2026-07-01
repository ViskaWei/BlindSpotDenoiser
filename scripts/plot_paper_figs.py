#!/usr/bin/env python3
"""plot_paper_figs.py — reproduce 5 paper figures from a raw inference dump.

Reads the h5 dump produced by `inference.py --dump-raw <path>`
and emits 5 PDF/PNG pairs into `paper/figs/appendix/`:

  fig_snr_improvement.{pdf,png}        — KDE of (Initial S/N, Final S/N)
  fig_snr_improvement_log.{pdf,png}    — same on log axes with y=x diagonal
  fig_ew_kde_ca2_snr2.{pdf,png}        — Ca II 8542 EW relative-error KDE in S/N (2,4]
  fig_ew_violin_ca2.{pdf,png}          — Ca II 8542 EW relative-error violin per S/N bin
  fig_ew_residual_std.{pdf,png}        — EW relative-error σ vs S/N for 4 Ca lines

Existing main-text figures (fig1_representative, fig2_noise_ensemble,
fig3_caII_zoom_panels, fig3_line_eval_summary) are NOT touched.

Usage (local, after scp'ing the dump.h5 from volta04):
    cd research/BlindSpotDenoiser
    python scripts/plot_paper_figs.py \
        --dump /path/to/best117_test1k_dump.h5 \
        --out-dir paper/figs/appendix
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.plotter import SpecPlotter  # noqa: E402


def _broadcast(arr_per_spec: np.ndarray, K: int) -> np.ndarray:
    """Broadcast (N,) → (K*N,) by tiling K times to align with flattened K×N."""
    return np.tile(arr_per_spec[None, :], (K, 1)).reshape(-1)


class DumpDataset:
    """Adapter exposing the attributes SpecPlotter expects.

    Layout: dump has [K=n_realizations, N=n_specs, L] arrays. SpecPlotter
    plot_* methods consume a flat [K*N, L] view, so reshape and broadcast
    per-spec metadata (T_eff/M_H/log_g/mag/snr/redshift) K-fold.
    For mu-only models the user-facing denoised output is mu_x (matches
    main.tex Method §4 + compute_m1_m9 line 320).
    """

    def __init__(self, h5_path: Path):
        with h5py.File(h5_path, "r") as f:
            wave = np.asarray(f["wave"])
            flux = np.asarray(f["flux"], dtype=np.float32)
            noisy = np.asarray(f["noisy"], dtype=np.float32)
            mu_x = np.asarray(f["mu_x"], dtype=np.float32)
            output_mode = f.attrs.get("output_mode", "mu")
            user_facing = mu_x if output_mode == "mu" else np.asarray(f["denoised"], dtype=np.float32)
            snr_noisy = np.asarray(f["snr_noisy_linear"], dtype=np.float32)
            snr_user = np.asarray(f["snr_user_facing_linear"], dtype=np.float32)
            meta = {}
            if "meta" in f:
                for k in f["meta"].keys():
                    meta[k] = np.asarray(f[f"meta/{k}"])
            ckpt = str(f.attrs.get("ckpt", "unknown"))
            n_real = int(f.attrs.get("n_realizations", noisy.shape[0]))
        K, N, L = noisy.shape
        assert K == n_real, f"shape K={K} vs attr n_realizations={n_real} mismatch"

        flat_clean = np.broadcast_to(flux[None, :, :], (K, N, L)).reshape(K * N, L).copy()
        flat_noisy = noisy.reshape(K * N, L)
        flat_user = user_facing.reshape(K * N, L)

        self.wave = torch.tensor(wave, dtype=torch.float32)
        self.flux = torch.tensor(flat_clean)
        self.noisy = torch.tensor(flat_noisy)
        self.denoised = torch.tensor(flat_user)
        self.snr_no_mask = snr_noisy.reshape(-1)
        self.snr = snr_user.reshape(-1)

        z_per = meta.get("redshift", meta.get("z", np.zeros(N, dtype=np.float64)))
        self.z = _broadcast(np.asarray(z_per, dtype=np.float64), K)

        for src, dst in [("T_eff", "teff"), ("M_H", "mh"), ("log_g", "logg"),
                         ("mag", "mag"), ("snr", "snr00")]:
            arr = meta.get(src)
            setattr(self, dst, _broadcast(arr.astype(np.float64), K) if arr is not None else None)

        self.K = K
        self.N = N
        self.L = L
        self.ckpt = ckpt
        self.output_mode = output_mode


def _save(fig, out_dir: Path, basename: str, dpi: int = 300):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{basename}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[plot] wrote {path}", flush=True)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, type=Path,
                    help="Path to h5 dump from inference --dump-raw.")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "paper" / "figs" / "appendix",
                    help="Output directory (default: paper/figs/appendix).")
    ap.add_argument("--ew-n", type=int, default=1000,
                    help="Number of spectra to use for EW figures (default 1000; "
                         "EW computation is the expensive step).")
    ap.add_argument("--snr-n", type=int, default=10000,
                    help="Number of (noisy,denoised) pairs for the SNR-improvement KDE "
                         "(default 10000 = K*N for K=10 N=1000).")
    args = ap.parse_args()

    print(f"[plot] loading dump from {args.dump}", flush=True)
    ds = DumpDataset(args.dump)
    print(f"[plot] K={ds.K} N={ds.N} L={ds.L} output_mode={ds.output_mode} ckpt={ds.ckpt}", flush=True)
    plotter = SpecPlotter(ds)

    n_snr = min(args.snr_n, ds.K * ds.N)
    print(f"[plot] generating SNR-improvement figures (N={n_snr})", flush=True)
    fig, fig_log = plotter.plot_snr_improve(N=n_snr)
    _save(fig, args.out_dir, "fig_snr_improvement", dpi=200)
    _save(fig_log, args.out_dir, "fig_snr_improvement_log", dpi=200)

    n_ew = min(args.ew_n, ds.K * ds.N)
    print(f"[plot] computing EW residuals (N={n_ew}); this is the slow step", flush=True)
    # SpecPlotter default bins=[0,2,4,6,8,inf] but mag 20.5-22.5 noisy SNR median≈7
    # with min≈2.5 — the first bin (0,2) is empty and `if len==0: break` aborts the
    # whole line loop, leaving df_equivalent_width empty. Shift edges to start at 2.
    # plotter L273 calls int(bins[i+1]) so np.inf overflows; use finite upper edge.
    plotter.get_equivalent_width_error_dict(
        N=n_ew, bins=np.array([2.0, 4.0, 6.0, 8.0, 10.0, 20.0]))

    print("[plot] generating EW KDE (Ca II 8542, S/N bin 2-4)", flush=True)
    fig_kde = plotter.plot_equivalent_width(plot_snr_bin="2 - 4", plot_name="Ca2_LB13")
    _save(fig_kde, args.out_dir, "fig_ew_kde_ca2_snr2")

    print("[plot] generating EW violin (Ca II 8542)", flush=True)
    fig_violin = plotter.plot_equivalent_width_violin(plot_name="Ca2_LB13")
    _save(fig_violin, args.out_dir, "fig_ew_violin_ca2")

    print("[plot] generating EW residual σ vs S/N (4 Ca lines)", flush=True)
    # plotter.py L290 truncates eq_bins to len(index_columns)=4 (bug); the actual
    # number of SNR bins is 5 (matches std_* arrays). Restore correct length.
    plotter.eq_bins = [2.0, 4.0, 6.0, 8.0, 10.0]
    fig_std = plotter.plot_ew_std_comparison()
    _save(fig_std, args.out_dir, "fig_ew_residual_std", dpi=200)

    print(f"\n[plot] done. 5 figures (10 files) under {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
