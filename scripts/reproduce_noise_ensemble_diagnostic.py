#!/usr/bin/env python3
"""Reproduce the fixed-spectrum noise-ensemble diagnostic for BlindSpot.

This script has two modes:

1. Compute a frozen local bundle from a checkpoint + one fixed spectrum under
   many fresh noise realizations.
2. Render the 2x2 diagnostic figure from that frozen bundle.

The intent is the same as the historical notebook diagnostic, but for the
current mu-only model we use the user-facing model output and denote it as
hat{x} rather than mu(D).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
BLINDSPOT_ROOT = ROOT.parent / "BlindSpotDenoiser"

DEFAULT_CONFIG = "/home/swei20/BlindSpot_ablation/configs/mu_only_baseline_mag205_225.yaml"
DEFAULT_CKPT = (
    "/home/swei20/BlindSpot_ablation/checkpoints_M1a_mag205_225/"
    "epoch=134-val/snr_mu_x=117.ckpt"
)
DEFAULT_TEST_PATH = "/datascope/subaru/user/swei20/data/bosz50000/mag215/val_10k/dataset.h5"
DEFAULT_SAMPLE_IDX = 4
DEFAULT_REPEAT = 10000
DEFAULT_SEED = 42
DEFAULT_NOISE_LEVEL = 1.0
DEFAULT_TARGET_FIELD = "denoised"
DEFAULT_STEM = "idx4_m1a134_noise_ensemble_10k"
DEFAULT_FIG_DIR = ROOT / "paper" / "figs" / "preview"
DEFAULT_DATA_DIR = ROOT / "data" / "fixed_preview"
DEFAULT_MEAN_OFFSET = 0.2


def _load_bundle(path: Path) -> dict[str, np.ndarray]:
    bundle = np.load(path, allow_pickle=True)
    return {key: bundle[key] for key in bundle.files}


def _load_inference_module():
    module_path = BLINDSPOT_ROOT / "scripts" / "inference.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing inference driver: {module_path}")
    spec = importlib.util.spec_from_file_location("bsn_inference", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_meta_row(test_path: str, idx: int) -> dict[str, float]:
    import pandas as pd

    row = pd.read_hdf(test_path, start=idx, stop=idx + 1).iloc[0]
    return {
        "teff": float(row["T_eff"]),
        "mh": float(row["M_H"]),
        "logg": float(row["log_g"]),
        "mag": float(row["mag"]),
        "snr_meta": float(row["snr"]),
    }


def compute_bundle(args: argparse.Namespace, bundle_path: Path, meta_path: Path) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    inf = _load_inference_module()
    data = inf.load_test_data(args.test_path, num_samples=args.idx + 1)
    wave = data["wave"]
    flux_all = data["flux"]
    error_all = data["error"]
    clean = flux_all[args.idx]
    error = error_all[args.idx]
    meta = _load_meta_row(args.test_path, args.idx)

    config = inf.load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lmodule = inf.build_lmodule(config, args.ckpt, device, config_label=args.config)

    clean_rep = clean.unsqueeze(0).repeat(args.repeat, 1)
    error_rep = error.unsqueeze(0).repeat(args.repeat, 1)
    torch.manual_seed(args.seed)
    noisy_all = clean_rep + torch.randn_like(clean_rep) * error_rep * args.noise_level

    ds = TensorDataset(noisy_all, clean_rep, error_rep)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    pred_chunks = []
    n_batches = len(dl)
    print(
        f"[ensemble] idx={args.idx} repeat={args.repeat} field={args.target_field} "
        f"batch_size={args.batch_size} device={device}",
        flush=True,
    )
    with torch.no_grad():
        for batch_i, (noisy_b, clean_b, error_b) in enumerate(dl, start=1):
            noisy_b = noisy_b.to(device)
            clean_b = clean_b.to(device)
            error_b = error_b.to(device)
            out = lmodule(noisy_b, clean_b, error_b * args.noise_level, loss_only=False)
            if args.target_field not in out:
                raise KeyError(
                    f"target_field={args.target_field!r} missing from model output keys={list(out.keys())}"
                )
            pred_chunks.append(out[args.target_field].detach().cpu())
            if batch_i == 1 or batch_i == n_batches or batch_i % max(1, n_batches // 10) == 0:
                print(f"[ensemble] batch {batch_i}/{n_batches}", flush=True)

    pred_all = torch.cat(pred_chunks, dim=0)
    mean_output = pred_all.mean(dim=0)
    std_output = pred_all.std(dim=0, unbiased=True)
    variance_output = std_output.pow(2)

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        bundle_path,
        wave=wave.cpu().numpy(),
        clean=clean.cpu().numpy(),
        error=error.cpu().numpy(),
        mean_output=mean_output.cpu().numpy(),
        variance_output=variance_output.cpu().numpy(),
        mean_offset=np.array(args.mean_offset, dtype=float),
        idx=np.array(args.idx, dtype=int),
        repeat=np.array(args.repeat, dtype=int),
        seed=np.array(args.seed, dtype=int),
        noise_level=np.array(args.noise_level, dtype=float),
        teff=np.array(meta["teff"], dtype=float),
        mh=np.array(meta["mh"], dtype=float),
        logg=np.array(meta["logg"], dtype=float),
        mag=np.array(meta["mag"], dtype=float),
        snr_meta=np.array(meta["snr_meta"], dtype=float),
    )
    meta_payload = {
        "bundle_path": str(bundle_path),
        "test_path": args.test_path,
        "idx": int(args.idx),
        "repeat": int(args.repeat),
        "seed": int(args.seed),
        "noise_level": float(args.noise_level),
        "target_field": args.target_field,
        "mean_offset": float(args.mean_offset),
        "config": args.config,
        "ckpt": args.ckpt,
        "device": str(device),
        "teff": meta["teff"],
        "mh": meta["mh"],
        "logg": meta["logg"],
        "mag": meta["mag"],
        "snr_meta": meta["snr_meta"],
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2))
    print(f"[save] bundle -> {bundle_path}", flush=True)
    print(f"[save] meta   -> {meta_path}", flush=True)


def render_bundle(bundle_path: Path, out_dir: Path, stem: str) -> tuple[Path, Path]:
    bundle = _load_bundle(bundle_path)
    wave = np.asarray(bundle["wave"], dtype=float)
    clean = np.asarray(bundle["clean"], dtype=float)
    error = np.asarray(bundle["error"], dtype=float)
    mean_output = np.asarray(bundle["mean_output"], dtype=float)
    if "variance_output" in bundle:
        variance_output = np.asarray(bundle["variance_output"], dtype=float)
        std_output = np.sqrt(np.clip(variance_output, 0.0, None))
    else:
        std_output = np.asarray(bundle["std_output"], dtype=float)
        variance_output = std_output ** 2
    residual = clean - mean_output
    mean_offset = float(bundle["mean_offset"])

    # Match BlindSpot stand-alone fig1/fig3 typography: serif (STIX) + stix
    # mathtext, paper-caption font scale. Audited 2026-05-01: previously
    # inherited matplotlib default DejaVuSans, mismatching paper body text.
    rc = {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(12.0, 4.0), dpi=200)
        fig.patch.set_alpha(0.0)
        gs = fig.add_gridspec(2, 2)
        ax_flux = fig.add_subplot(gs[0, 0])
        ax_res = fig.add_subplot(gs[1, 0], sharex=ax_flux)
        ax_scale = fig.add_subplot(gs[:, 1], sharex=ax_flux)
        for ax in (ax_flux, ax_res, ax_scale):
            ax.set_facecolor("none")

        c_flux = "dodgerblue"
        c_mean = "navy"
        c_res = "seagreen"
        c_err = "darkgoldenrod"
        c_std = "darkolivegreen"

        ax_flux.plot(wave, clean, c=c_flux, lw=0.8, label=r"$x$")
        ax_flux.plot(wave, mean_output - mean_offset, c=c_mean, lw=0.8, label=rf"$\langle \hat{{x}} \rangle - {mean_offset:.1f}$")
        legend0 = ax_flux.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), handlelength=3.2, frameon=False)
        for line in legend0.get_lines():
            line.set_linewidth(4)
        ax_flux.tick_params(axis="y", labelcolor="k")
        ax_flux.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax_flux.set(xlim=(float(wave[0]), float(wave[-1])))
        # Audit fix 2026-05-01: previously missing left-panel ylabel.
        ax_flux.set_ylabel("Normalized Flux")

        ax_res.plot(wave, residual, c=c_res, lw=0.8, label=rf"$x - \langle \hat{{x}} \rangle$")
        legend1 = ax_res.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), handlelength=3.2, frameon=False)
        legend1.get_lines()[0].set_linewidth(4)
        # Audit fix 2026-05-01: tick labels back to black (previously seagreen,
        # colorblind-unfriendly when same hue as the residual line).
        ax_res.tick_params(axis="y", labelcolor="k")
        ax_res.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax_res.set(xlim=(float(wave[0]), float(wave[-1])), xlabel=r"Wavelength ($\AA$)")
        # Audit fix 2026-05-01: previously missing residual-panel ylabel.
        ax_res.set_ylabel(r"Residual")

        std_plot = np.clip(std_output, 1e-8, None)
        mean_err = float(error.mean())
        mean_std = float(std_plot.mean())
        ax_scale.plot(
            wave,
            error,
            c=c_err,
            alpha=1.0,
            lw=0.8,
            label=rf"$\sigma_0,\ \langle \sigma_0 \rangle = {mean_err:.1g}$",
        )
        ax_scale.plot(
            wave,
            std_plot,
            c=c_std,
            lw=0.8,
            label=rf"$\sigma_{{\hat{{x}}}},\ \langle \sigma_{{\hat{{x}}}} \rangle = {mean_std:.1g}$",
        )
        ax_scale.set(
            xlim=(float(wave[0]), float(wave[-1])),
            ylim=(0.0, max(float(np.nanmax(error)), float(np.nanmax(std_plot))) * 1.05),
            xlabel=r"Wavelength ($\AA$)",
        )
        # Audit fix 2026-05-01: previously missing right-panel ylabel.
        ax_scale.set_ylabel(r"Noise scale")
        legend2 = ax_scale.legend(loc="lower center", bbox_to_anchor=(0.5, 0.08), handlelength=3.2, frameon=False)
        for line in legend2.get_lines():
            line.set_linewidth(4)
        ax_scale.tick_params(axis="y", labelcolor="k")

        ax_flux.tick_params(axis="x", labelbottom=False)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / f"{stem}.pdf"
        png_path = out_dir / f"{stem}.png"
        fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
        fig.savefig(png_path, dpi=200, bbox_inches="tight", transparent=True)
        plt.close(fig)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=None, help="Existing frozen bundle (.npz).")
    p.add_argument("--recompute", action="store_true", help="Force recomputation before rendering.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    p.add_argument("--idx", type=int, default=DEFAULT_SAMPLE_IDX)
    p.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--noise-level", type=float, default=DEFAULT_NOISE_LEVEL)
    p.add_argument("--target-field", default=DEFAULT_TARGET_FIELD, choices=["denoised", "mu_x"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mean-offset", type=float, default=DEFAULT_MEAN_OFFSET)
    p.add_argument("--stem", default=DEFAULT_STEM)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle_path = args.bundle if args.bundle is not None else args.data_dir / f"{args.stem}.npz"
    meta_path = bundle_path.with_suffix(".json")
    if args.recompute or not bundle_path.exists():
        compute_bundle(args, bundle_path=bundle_path, meta_path=meta_path)
    pdf_path, png_path = render_bundle(bundle_path, args.out_dir, args.stem)
    print(f"[render] pdf -> {pdf_path}")
    print(f"[render] png -> {png_path}")


if __name__ == "__main__":
    main()
