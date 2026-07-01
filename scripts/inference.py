#!/usr/bin/env python3
"""inference.py — canonical inference suite for the BSN paper.

Produces the paper-facing measurements and representative figure for a
locked checkpoint / regime pair. As of 2026-04-27 the active paper
canonical target is the faint-regime M1a mu-only checkpoint on
mag20.5-22.5, but the driver remains configurable so archival runs can be
reproduced when needed.

Run on volta04, single GPU. Driver is self-contained: loads ckpt, runs
the canonical z1/test_10k_1 evaluation by default, computes per-spec
metrics, EW preservation, polarity, hard-window, blindspot integrity,
and emits a representative-spectrum figure. The sigma_x diagnostic is
kept only for archival dual-head runs.

Usage (volta04):
    cd /home/swei20/R_abuv-Propulsion/research/BlindSpotDenoiser
    source /datascope/subaru/user/swei20/miniconda3/etc/profile.d/conda.sh
    conda activate specvit
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
        python scripts/inference.py \\
        --out /datascope/subaru/user/swei20/blindspot/results/mag1921_canonical_2026-04-27/ \\
        2>&1 | tee inference.log

Outputs (M1-M6, M9; M7 only for dual-head archival runs):
    canonical_test_metrics.csv          (M1 + M9 bootstrap CI)
    ew_preservation_test.csv            (M2)
    polarity_test.csv                   (M3)
    hard_window_metric.csv              (M4)
    fig1_representative.{pdf,png}       (M5)
    fig2_noise_ensemble.{pdf,png}       (M5 repeated-noise sidecar)
    blindspot_integrity_audit.json      (M6)
    sigma_x_diagnostic.json             (M7, dual-head archival only)
    inference_meta.json                 (provenance: ckpt, config, GPU, wall, sha)

Reproducibility:
    - SEED=42 base, SEED+k for k-th noise realization
    - Default paper evaluation is z1/test_10k_1 with one realization per spectrum
    - Pass an explicit --test-path/--n-realizations pair for archival test_1k x10 runs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

# Repo-root on PYTHONPATH so `from src...` works
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.blindspot import BlindspotLModule  # noqa: E402
from src.utils import load_config  # noqa: E402

# ----------------------------------------------------------------------------
# Constants — DEC-2026-04-27-005 lock
# ----------------------------------------------------------------------------
# Paper canonical = M1a mu-only, epoch 190 (promoted 2026-04-30, commit 6eb8ff5).
# FIXED 2026-06-21: defaults previously pointed to the STALE pre-promotion E1
# baseline (ablation_e1_self_sup_baseline.yaml + ep65 best117). Loading the M1a
# ckpt under the E1 config produced GARBAGE mu_x (min=-932) -> wrong EW (0.26 vs
# correct 0.12). All callers (sec5_4 bootstrap, plot/render fig scripts) inherit
# this; the config+ckpt MUST be the matched M1a pair.
CANONICAL_CONFIG = "configs/mu_only_baseline_mag205_225.yaml"
CANONICAL_CKPT = (
    "/home/swei20/BlindSpot_ablation/checkpoints_M1a_mag205_225/"
    "epoch=190-val/snr_mu_x=123.ckpt"
)
# CRITICAL: training applies this mask (mask_ratio=0.85) BEFORE forward pass.
# Inference must apply same mask or model output is garbage (snr_mu_x drops 117 → 0.4).
# See src/blindspot.py SpecTrainDataset/SpecTestDataset.load_data().
MASK_PATH = "/datascope/subaru/user/swei20/model/bosz50000_mask.npy"
CANONICAL_TEST_PATH = (
    "/datascope/subaru/user/swei20/data/bosz50000/z1/"
    "mag205_225_lowT_1M/test_10k_1/dataset.h5"
)
CANONICAL_NUM_TEST_SAMPLES = 10000
N_NOISE_REALIZATIONS = 1  # paper canonical z1/test_10k_1
BASE_SEED = 42
BATCH_SIZE = 64
NOISE_ENSEMBLE_REPEAT = 10000
NOISE_ENSEMBLE_BATCH_SIZE = 128

# Astronomy lines in the 7100-8850 Å range (canonical regime per main.tex Table 1)
LINE_LIST: list[tuple[str, float]] = [
    ("CaII_T1", 8498.0),  # CaII triplet line 1
    ("CaII_T2", 8542.0),  # CaII triplet line 2
    ("CaII_T3", 8662.0),  # CaII triplet line 3
]
# Telluric O2 A-band canonical hard window (covers strong tellurics)
HARD_WINDOW_NAME = "O2_A_band_telluric"
HARD_WINDOW_RANGE_A = (7590.0, 7700.0)

# Canonical paper-figure palette (matches SpecDenoiser/scripts/make_fig1_blindspot_fixed_preview.py).
# Any BlindSpot paper figure rendered from this repo MUST use this palette unless explicitly noted.
PAPER_FIG_COLORS = {
    "noisy": "black",
    "clean": "dodgerblue",
    "xhat": "navy",
    "residual": "seagreen",
    "highlight": "aquamarine",
    "sigma0": "darkgoldenrod",
    "sigma_xhat": "darkolivegreen",
}
PAPER_FIG_HIGHLIGHT_ALPHA = 0.5


# ----------------------------------------------------------------------------
# Utility metrics
# ----------------------------------------------------------------------------
# ⚠️ TWO DIFFERENT S/N METRICS — do NOT equate them (2026-06-20 F1 trap):
#   - snr_pixel_db (below): VARIANCE ratio, 10·log10(var/var). Values ~ -4.26 / 19.8 dB.
#     This is what canonical_test_metrics.csv stores (snr_pixel / snr_continuum_normalized).
#   - snr_ratio() (see ~L880): NORM ratio ‖clean‖/‖clean-x‖, the paper HEADLINE (7.6 -> 122).
#     dB form = 20·log10(ratio); this is NOT the variance-pixel dB metric.
#   The paper caption must never call the variance-pixel dB metric "the dB form" of the headline ratio.
#   See paper/NUMBER_PROVENANCE_RUNBOOK.md trap 2.
def snr_pixel_db(clean: torch.Tensor, x: torch.Tensor) -> float:
    """SNR_pixel = 10·log10(var(clean) / var(clean - x)). Returns dB.

    NOTE: variance-domain dB, NOT the headline norm-ratio S/N (see snr_ratio()).
    """
    res = clean - x
    var_clean = clean.var(dim=-1, unbiased=True)
    var_res = res.var(dim=-1, unbiased=True).clamp(min=1e-30)
    return 10.0 * torch.log10(var_clean / var_res).mean().item()


def rmse_pixel(clean: torch.Tensor, x: torch.Tensor) -> float:
    return (clean - x).pow(2).mean(dim=-1).sqrt().mean().item()


def mae_pixel(clean: torch.Tensor, x: torch.Tensor) -> float:
    return (clean - x).abs().mean(dim=-1).mean().item()


def snr_continuum_normalized_db(clean: torch.Tensor, x: torch.Tensor) -> float:
    """For continuum-normalized spectra (mean ~ 1), SNR via deviations from continuum."""
    dev_clean = clean - clean.mean(dim=-1, keepdim=True)
    res = clean - x
    var_dev = dev_clean.var(dim=-1, unbiased=True).clamp(min=1e-30)
    var_res = res.var(dim=-1, unbiased=True).clamp(min=1e-30)
    return 10.0 * torch.log10(var_dev / var_res).mean().item()


def equivalent_width(flux: torch.Tensor, wave: torch.Tensor,
                     line_center_a: float, line_halfwidth_a: float = 5.0,
                     cont_inner_a: float = 5.0, cont_outer_a: float = 15.0) -> torch.Tensor:
    """Per-spectrum EW (Å) via local linear continuum fit.

    flux: (B, L) continuum-normalized flux
    wave: (L,) wavelength array
    Returns: (B,) EW values; positive = absorption.
    """
    line_low, line_high = line_center_a - line_halfwidth_a, line_center_a + line_halfwidth_a
    cont_left = (wave >= line_center_a - cont_outer_a) & (wave <= line_center_a - cont_inner_a)
    cont_right = (wave >= line_center_a + cont_inner_a) & (wave <= line_center_a + cont_outer_a)
    line_mask = (wave >= line_low) & (wave <= line_high)
    cont_mask = cont_left | cont_right

    if line_mask.sum() == 0 or cont_mask.sum() < 4:
        return torch.full((flux.shape[0],), float("nan"))

    w_line = wave[line_mask]
    w_cont = wave[cont_mask]
    f_cont = flux[:, cont_mask]

    # Local linear continuum: c(λ) = a + b·λ via least squares per spectrum
    A = torch.stack([torch.ones_like(w_cont), w_cont], dim=-1)  # (Ncont, 2)
    AtA_inv = torch.linalg.pinv(A.T @ A)  # (2,2)
    coeffs = (AtA_inv @ A.T) @ f_cont.T  # (2, B)
    a, b = coeffs[0], coeffs[1]  # each (B,)

    cont_at_line = a.unsqueeze(-1) + b.unsqueeze(-1) * w_line.unsqueeze(0)  # (B, Nline)
    f_line = flux[:, line_mask]
    integrand = 1.0 - (f_line / cont_at_line.clamp(min=1e-12))
    # trapezoidal integration over wavelength
    dw = (w_line[1:] - w_line[:-1])
    ew = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dw.unsqueeze(0)
    return ew.sum(dim=-1)


def polarity_flip_rate(clean: torch.Tensor, x: torch.Tensor,
                       region_mask: torch.Tensor | None = None,
                       eps_frac: float = 0.02) -> tuple[float, int]:
    """Per-pixel sign-flip rate between clean and x relative to per-spec median.

    BOSZ stellar spectra in this regime are not continuum-normalized to 1.0
    (flux mean ≈ 0.5), so the reference level is taken as per-spectrum median.
    Sign-flip uses sign(x - median) vs sign(clean - median).
    Pixels with |clean - median| < eps_frac * median are treated as continuum
    (no defined polarity) and excluded from the count via sign_clean == 0.
    """
    if region_mask is None:
        region_mask = torch.ones_like(clean, dtype=torch.bool)
    median = clean.median(dim=-1, keepdim=True).values
    eps = (eps_frac * median).abs()
    # zero out small deviations so they don't count as a polarity
    devs_clean = clean - median
    devs_x = x - median
    sign_clean = torch.sign(torch.where(devs_clean.abs() < eps, torch.zeros_like(devs_clean), devs_clean))
    sign_x = torch.sign(devs_x)
    flips = (sign_clean != sign_x) & region_mask & (sign_clean != 0)
    n = (region_mask & (sign_clean != 0)).sum().item()
    if n == 0:
        return float("nan"), 0
    return flips.sum().item() / n, int(n)


# ----------------------------------------------------------------------------
# Data + model loading
# ----------------------------------------------------------------------------
def load_test_data(test_path: str, num_samples: int, mask_path: str | None = MASK_PATH,
                   use_stored_noisy: bool = False) -> dict:
    print(f"[data] loading test split from {test_path} (N≤{num_samples})", flush=True)
    with h5py.File(test_path, "r") as f:
        wave = torch.tensor(f["spectrumdataset/wave"][()]).float()
        flux = torch.tensor(f["dataset/arrays/flux/value"][:num_samples]).float()
        error = torch.tensor(f["dataset/arrays/error/value"][:num_samples]).float()
        if use_stored_noisy:
            noisy_key = "dataset/arrays/noisy/value"
            if noisy_key not in f:
                raise KeyError(f"--use-stored-noisy requested, but {test_path} has no {noisy_key}")
            noisy = torch.tensor(f[noisy_key][:num_samples]).float()
        else:
            noisy = None
    flux = flux.clamp(min=0.0)
    if torch.isnan(error).any():
        # mirror BaseSpecDataset.fill_nan_with_nearest
        if torch.isnan(error[:, 0]).any():
            error[:, 0] = error[:, 1]
        if torch.isnan(error[:, -1]).any():
            error[:, -1] = error[:, -2]
        error = error.nan_to_num(nan=error[~torch.isnan(error)].median().item())
    # CRITICAL: apply training mask so sequence length matches model expectation
    # (snr_mu_x reproduces 117 only on masked input).
    if mask_path is not None:
        mask = np.load(mask_path)
        mask_t = torch.tensor(mask, dtype=torch.bool)
        flux = flux[..., mask_t]
        error = error[..., mask_t]
        if noisy is not None:
            noisy = noisy[..., mask_t]
        wave = wave[mask_t]
        print(f"[data] applied mask {mask_path}: kept {int(mask.sum())}/{mask.size} pixels "
              f"({100*mask.mean():.1f}%)", flush=True)
    if noisy is not None:
        print(f"[data] using stored noisy array from HDF5: noisy={tuple(noisy.shape)}", flush=True)
    print(f"[data] flux={tuple(flux.shape)} error={tuple(error.shape)} wave={tuple(wave.shape)}", flush=True)
    return {"wave": wave, "flux": flux, "error": error, "stored_noisy": noisy}


def build_lmodule(config: dict, ckpt_path: str, device: torch.device,
                  config_label: str | None = None) -> BlindspotLModule:
    if config_label is not None:
        print(f"[model] building BlindspotLModule from {config_label}", flush=True)
    else:
        print("[model] building BlindspotLModule", flush=True)
    lmodule = BlindspotLModule(config=config)
    print(f"[model] loading state_dict from {ckpt_path}", flush=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = lmodule.load_state_dict(sd, strict=False)
    print(f"[model] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    lmodule = lmodule.to(device).eval()
    return lmodule


@torch.no_grad()
def run_inference_one_realization(lmodule: BlindspotLModule, flux: torch.Tensor,
                                  error: torch.Tensor, noise_level: float,
                                  seed: int, device: torch.device,
                                  log_sigma_clamp_max: float = 3.0,
                                  stored_noisy: torch.Tensor | None = None) -> dict:
    """Run one full forward pass over the test set with a fresh noise realization.

    Returns dict with keys:
      - 'noisy'    : (B, L) noisy observation
      - 'mu_x'     : (B, L) blindspot-pure NN output (uses ONLY context)
      - 'log_sigma_x': (B, L) raw log_sigma_x output, clamped to training range
      - 'sigma_x'  : (B, L) = exp(log_sigma_x)
      - 'denoised' : (B, L) posterior-fused output (USER-FACING; what main.tex Table 2 reports)
                     denoised = (sigma_x² · noisy + sigma_n² · mu_x) / (sigma_x² + sigma_n²)
                     Reduces to noisy for confident NN (sigma_x → 0) or to mu_x when sigma_x dominates.

    All tensors returned on CPU.
    """
    if stored_noisy is None:
        torch.manual_seed(seed)
        noise = torch.randn_like(flux) * error * noise_level
        noisy = flux + noise
    else:
        noisy = stored_noisy

    mu_list, log_sigma_list, denoised_list = [], [], []
    for i in range(0, noisy.shape[0], BATCH_SIZE):
        nb = noisy[i:i + BATCH_SIZE].to(device)
        eb = error[i:i + BATCH_SIZE].to(device)

        # processed sigma channel (L0 = scalar rms, broadcast)
        processed_sigma = lmodule._process_error_input(eb)
        if processed_sigma is not None and lmodule.input_sigma:
            inputs = torch.cat([nb.unsqueeze(1), processed_sigma], dim=1)
        else:
            inputs = nb.unsqueeze(1)

        outputs = lmodule.model(inputs)  # (b, 2, L) for dual-head
        mu = outputs[:, 0, :]
        log_sigma = outputs[:, 1, :] if outputs.shape[1] >= 2 else torch.zeros_like(mu)

        # Clamp log_sigma to training range (per blindspot.py line 432: log_sigma_x.clamp(min=-12, max=log_sigma_clamp_max))
        log_sigma = log_sigma.clamp(min=-12.0, max=log_sigma_clamp_max)
        sigma_x = torch.exp(log_sigma)

        # Reproduce the posterior-fusion done in compute_blindspot_loss to obtain
        # the user-facing 'denoised' output. Use _get_loss_sigma to get the same
        # leakage-safe sigma_noise that training used (see AGENT_BRIEFING rule #4).
        sigma_noise_b = lmodule._get_loss_sigma(eb, processed_sigma)  # (b, L)
        var_x = sigma_x ** 2
        var_n = sigma_noise_b ** 2
        var_y = (var_x + var_n).clamp(min=1e-12)
        denoised = (var_x * nb + var_n * mu) / var_y

        mu_list.append(mu.cpu())
        log_sigma_list.append(log_sigma.cpu())
        denoised_list.append(denoised.cpu())

    log_sigma_all = torch.cat(log_sigma_list, dim=0)
    return {
        "noisy": noisy,
        "mu_x": torch.cat(mu_list, dim=0),
        "log_sigma_x": log_sigma_all,
        "sigma_x": torch.exp(log_sigma_all),
        "denoised": torch.cat(denoised_list, dim=0),
    }


# ----------------------------------------------------------------------------
# M1 + M9: canonical_test_metrics.csv with bootstrap CI
# ----------------------------------------------------------------------------
def compute_m1_m9(realizations: list[dict], flux: torch.Tensor, out_path: Path) -> dict:
    """For each metric, compute mean across N realizations (per-spec, then mean over spec).
    Bootstrap CI = 1.96·std/sqrt(N) on the mean across realizations.

    Uses 'denoised' (posterior-fused user-facing output) per main.tex Method §4.
    """
    metrics = ["snr_pixel", "rmse_pixel", "snr_continuum_normalized", "mae_pixel"]
    metric_fns = {
        "snr_pixel": snr_pixel_db,
        "rmse_pixel": rmse_pixel,
        "snr_continuum_normalized": snr_continuum_normalized_db,
        "mae_pixel": mae_pixel,
    }
    rows = []
    summary = {}
    for m in metrics:
        fn = metric_fns[m]
        noisy_vals = np.array([fn(flux, r["noisy"]) for r in realizations])
        denoised_vals = np.array([fn(flux, r["mu_x"]) for r in realizations])  # main.tex §4: mu is user-facing denoised
        delta_vals = denoised_vals - noisy_vals
        n = len(realizations)
        is_db = m.startswith("snr")
        unit = "dB" if is_db else "fluxunit"
        rows.append({
            "metric": m,
            "unit": unit,
            "noisy_input_mean": float(noisy_vals.mean()),
            "noisy_input_std": float(noisy_vals.std(ddof=1)) if n > 1 else 0.0,
            "noisy_input_ci95": float(1.96 * noisy_vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "denoised_output_mean": float(denoised_vals.mean()),
            "denoised_output_std": float(denoised_vals.std(ddof=1)) if n > 1 else 0.0,
            "denoised_output_ci95": float(1.96 * denoised_vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "delta_mean": float(delta_vals.mean()),
            "delta_std": float(delta_vals.std(ddof=1)) if n > 1 else 0.0,
            "delta_ci95": float(1.96 * delta_vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "n_realizations": n,
            "n_test_specs": int(flux.shape[0]),
        })
        summary[m] = rows[-1]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[M1+M9] wrote {out_path}", flush=True)
    return summary


# ----------------------------------------------------------------------------
# M2: ew_preservation_test.csv
# ----------------------------------------------------------------------------
def compute_m2(realization: dict, flux: torch.Tensor, wave: torch.Tensor,
               out_path: Path) -> dict:
    rows = []
    summary = {}
    noisy = realization["noisy"]
    denoised = realization["mu_x"]  # main.tex §4: "report mu as the denoised output"
    for name, center in LINE_LIST:
        ew_clean = equivalent_width(flux, wave, center)
        ew_noisy = equivalent_width(noisy, wave, center)
        ew_denoised = equivalent_width(denoised, wave, center)
        # Per-spec degradation = |EW_X - EW_clean| / |EW_clean| (clip denominator)
        denom = ew_clean.abs().clamp(min=1e-6)
        deg_noisy = (ew_noisy - ew_clean).abs() / denom
        deg_denoised = (ew_denoised - ew_clean).abs() / denom
        # Filter NaN / non-finite per-spec
        mask = torch.isfinite(deg_noisy) & torch.isfinite(deg_denoised) & torch.isfinite(ew_clean) & (ew_clean.abs() > 0.001)
        if mask.sum() == 0:
            print(f"[M2 WARN] line {name} has no valid spec; skipping", flush=True)
            continue
        rows.append({
            "line_name": name,
            "wavelength_A": center,
            "EW_clean_A_mean": float(ew_clean[mask].mean().item()),
            "EW_noisy_A_mean": float(ew_noisy[mask].mean().item()),
            "EW_denoised_A_mean": float(ew_denoised[mask].mean().item()),
            "deg_noisy_mean": float(deg_noisy[mask].mean().item()),
            "deg_noisy_median": float(deg_noisy[mask].median().item()),
            "deg_denoised_mean": float(deg_denoised[mask].mean().item()),
            "deg_denoised_median": float(deg_denoised[mask].median().item()),
            "n_specs": int(mask.sum().item()),
        })
        summary[name] = rows[-1]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[M2] wrote {out_path}", flush=True)
    return summary


# ----------------------------------------------------------------------------
# M3: polarity_test.csv
# ----------------------------------------------------------------------------
def compute_m3(realization: dict, flux: torch.Tensor, out_path: Path) -> dict:
    """Per-pixel polarity (line/continuum sign) preservation.

    Reference level = per-spec median (BOSZ data not normalized to 1.0).
    Regions defined relative to per-spec median:
      absorption: flux < median - 2% × median
      continuum:  |flux - median| < 2% × median
      emission:   flux > median + 2% × median
    """
    rows = []
    summary = {}
    eps_frac = 0.02
    median = flux.median(dim=-1, keepdim=True).values
    eps = (eps_frac * median).abs()
    regions = {
        "all": torch.ones_like(flux, dtype=torch.bool),
        "absorption": flux < (median - eps),
        "continuum": (flux >= (median - eps)) & (flux <= (median + eps)),
        "emission": flux > (median + eps),
    }
    noisy = realization["noisy"]
    denoised = realization["mu_x"]  # main.tex §4: "report mu as the denoised output"
    for name, mask in regions.items():
        flip_noisy, n = polarity_flip_rate(flux, noisy, mask, eps_frac=eps_frac)
        flip_denoised, _ = polarity_flip_rate(flux, denoised, mask, eps_frac=eps_frac)
        rows.append({
            "region": name,
            "n_pixels": n,
            "flip_rate_noisy": flip_noisy,
            "flip_rate_denoised": flip_denoised,
        })
        summary[name] = rows[-1]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[M3] wrote {out_path}", flush=True)
    return summary


# ----------------------------------------------------------------------------
# M4: hard_window_metric.csv (telluric O2 A-band)
# ----------------------------------------------------------------------------
def compute_m4(realization: dict, flux: torch.Tensor, wave: torch.Tensor,
               out_path: Path) -> dict:
    lo, hi = HARD_WINDOW_RANGE_A
    mask = (wave >= lo) & (wave <= hi)
    if mask.sum() == 0:
        print(f"[M4 WARN] hard window {lo}-{hi}Å outside wave range, skipping", flush=True)
        df = pd.DataFrame([{"window": HARD_WINDOW_NAME, "status": "OUT_OF_RANGE"}])
        df.to_csv(out_path, index=False)
        return {}
    flux_w = flux[:, mask]
    noisy_w = realization["noisy"][:, mask]
    denoised_w = realization["mu_x"][:, mask]  # main.tex §4: mu is user-facing denoised
    snr_noisy = snr_pixel_db(flux_w, noisy_w)
    snr_denoised = snr_pixel_db(flux_w, denoised_w)
    # also a within-window EW for the strongest absorption around the window
    ew_clean = equivalent_width(flux, wave, line_center_a=(lo + hi) / 2.0,
                                line_halfwidth_a=(hi - lo) / 2.0,
                                cont_inner_a=(hi - lo) / 2.0 + 5.0,
                                cont_outer_a=(hi - lo) / 2.0 + 25.0)
    ew_noisy = equivalent_width(realization["noisy"], wave,
                                line_center_a=(lo + hi) / 2.0,
                                line_halfwidth_a=(hi - lo) / 2.0,
                                cont_inner_a=(hi - lo) / 2.0 + 5.0,
                                cont_outer_a=(hi - lo) / 2.0 + 25.0)
    ew_denoised = equivalent_width(realization["mu_x"], wave,  # main.tex §4: mu is user-facing denoised
                                   line_center_a=(lo + hi) / 2.0,
                                   line_halfwidth_a=(hi - lo) / 2.0,
                                   cont_inner_a=(hi - lo) / 2.0 + 5.0,
                                   cont_outer_a=(hi - lo) / 2.0 + 25.0)
    valid = torch.isfinite(ew_clean) & torch.isfinite(ew_noisy) & torch.isfinite(ew_denoised) & (ew_clean.abs() > 0.001)
    if valid.sum() > 0:
        denom = ew_clean[valid].abs().clamp(min=1e-6)
        deg_noisy = float(((ew_noisy[valid] - ew_clean[valid]).abs() / denom).mean().item())
        deg_denoised = float(((ew_denoised[valid] - ew_clean[valid]).abs() / denom).mean().item())
        ew_n_specs = int(valid.sum().item())
    else:
        deg_noisy = float("nan")
        deg_denoised = float("nan")
        ew_n_specs = 0

    row = {
        "window": HARD_WINDOW_NAME,
        "wavelength_low_A": lo,
        "wavelength_high_A": hi,
        "n_pixels_in_window": int(mask.sum().item()),
        "snr_noisy_dB": snr_noisy,
        "snr_denoised_dB": snr_denoised,
        "delta_dB": snr_denoised - snr_noisy,
        "EW_window_clean_A_mean": float(ew_clean[valid].mean().item()) if valid.sum() > 0 else float("nan"),
        "EW_window_noisy_A_mean": float(ew_noisy[valid].mean().item()) if valid.sum() > 0 else float("nan"),
        "EW_window_denoised_A_mean": float(ew_denoised[valid].mean().item()) if valid.sum() > 0 else float("nan"),
        "deg_noisy_mean": deg_noisy,
        "deg_denoised_mean": deg_denoised,
        "n_specs_for_EW": ew_n_specs,
    }
    df = pd.DataFrame([row])
    df.to_csv(out_path, index=False)
    print(f"[M4] wrote {out_path}", flush=True)
    return row


# ----------------------------------------------------------------------------
# M5: fig1_representative.{pdf,png} + fig2_noise_ensemble.{pdf,png}
# ----------------------------------------------------------------------------
def render_noise_ensemble_panel(
    wave_np: np.ndarray,
    clean_np: np.ndarray,
    error_np: np.ndarray,
    mean_output_np: np.ndarray,
    std_output_np: np.ndarray,
    out_dir: Path,
    stem: str = "fig2_noise_ensemble",
    mean_offset: float = 0.2,
) -> tuple[Path, Path]:
    """Render the fixed-spectrum repeated-noise diagnostic.

    This mirrors the standalone SpecDenoiser renderer so the inference driver
    can regenerate the paper figure directly after M5 finishes.
    """
    residual = clean_np - mean_output_np

    fig = plt.figure(figsize=(12.0, 4.0), dpi=200)
    fig.patch.set_alpha(0.0)
    gs = fig.add_gridspec(2, 2)
    ax_flux = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_flux)
    ax_scale = fig.add_subplot(gs[:, 1], sharex=ax_flux)
    for ax in (ax_flux, ax_res, ax_scale):
        ax.set_facecolor("none")

    c_flux = PAPER_FIG_COLORS["clean"]
    c_mean = PAPER_FIG_COLORS["xhat"]
    c_res = PAPER_FIG_COLORS["residual"]
    c_err = PAPER_FIG_COLORS["sigma0"]
    c_std = PAPER_FIG_COLORS["sigma_xhat"]

    ax_flux.plot(wave_np, clean_np, c=c_flux, lw=0.8, label=r"$x$")
    ax_flux.plot(
        wave_np,
        mean_output_np - mean_offset,
        c=c_mean,
        lw=0.8,
        label=rf"$\langle \hat{{x}} \rangle - {mean_offset:.1f}$",
    )
    legend0 = ax_flux.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        handlelength=3.2,
        frameon=False,
    )
    for line in legend0.get_lines():
        line.set_linewidth(4)
    ax_flux.tick_params(axis="y", labelcolor="k")
    ax_flux.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_flux.set(xlim=(float(wave_np[0]), float(wave_np[-1])))

    ax_res.plot(wave_np, residual, c=c_res, lw=0.8, label=rf"$x - \langle \hat{{x}} \rangle$")
    legend1 = ax_res.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        handlelength=3.2,
        frameon=False,
    )
    legend1.get_lines()[0].set_linewidth(4)
    ax_res.tick_params(axis="y", labelcolor=c_res)
    ax_res.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_res.set(xlim=(float(wave_np[0]), float(wave_np[-1])), xlabel=r"Wavelength ($\AA$)")

    std_plot = np.clip(std_output_np, 1e-8, None)
    mean_err = float(error_np.mean())
    mean_std = float(std_plot.mean())
    ax_scale.plot(
        wave_np,
        error_np,
        c=c_err,
        alpha=1.0,
        lw=0.8,
        label=rf"$\sigma_0,\ \langle \sigma_0 \rangle = {mean_err:.1g}$",
    )
    ax_scale.plot(
        wave_np,
        std_plot,
        c=c_std,
        lw=0.8,
        label=rf"$\sigma_{{\hat{{x}}}},\ \langle \sigma_{{\hat{{x}}}} \rangle = {mean_std:.1g}$",
    )
    ax_scale.set(
        xlim=(float(wave_np[0]), float(wave_np[-1])),
        ylim=(0.0, max(float(np.nanmax(error_np)), float(np.nanmax(std_plot))) * 1.05),
        xlabel=r"Wavelength ($\AA$)",
    )
    legend2 = ax_scale.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        handlelength=3.2,
        frameon=False,
    )
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


@torch.no_grad()
def compute_m5_noise_ensemble(
    lmodule: BlindspotLModule,
    flux: torch.Tensor,
    error: torch.Tensor,
    wave: torch.Tensor,
    spec_idx: int,
    out_dir: Path,
    device: torch.device,
    noise_level: float,
    repeat: int = NOISE_ENSEMBLE_REPEAT,
    batch_size: int = NOISE_ENSEMBLE_BATCH_SIZE,
    seed: int = BASE_SEED,
) -> dict:
    clean_s = flux[spec_idx].to(device)
    error_s = error[spec_idx].to(device)
    sum_output = torch.zeros_like(clean_s, dtype=torch.float64, device="cpu")
    sumsq_output = torch.zeros_like(clean_s, dtype=torch.float64, device="cpu")
    n_seen = 0

    torch.manual_seed(seed)
    for start in range(0, repeat, batch_size):
        bsz = min(batch_size, repeat - start)
        clean_b = clean_s.unsqueeze(0).expand(bsz, -1)
        error_b = error_s.unsqueeze(0).expand(bsz, -1)
        noisy_b = clean_b + torch.randn_like(clean_b) * error_b * noise_level

        processed_sigma = lmodule._process_error_input(error_b)
        if processed_sigma is not None and lmodule.input_sigma:
            inputs = torch.cat([noisy_b.unsqueeze(1), processed_sigma], dim=1)
        else:
            inputs = noisy_b.unsqueeze(1)
        pred_b = lmodule.model(inputs)[:, 0, :].detach().cpu().to(torch.float64)
        sum_output += pred_b.sum(dim=0)
        sumsq_output += pred_b.pow(2).sum(dim=0)
        n_seen += bsz

    mean_output = sum_output / n_seen
    variance_output = (sumsq_output - n_seen * mean_output.pow(2)) / max(n_seen - 1, 1)
    std_output = torch.sqrt(torch.clamp(variance_output, min=0.0))

    pdf_path, png_path = render_noise_ensemble_panel(
        wave.cpu().numpy(),
        flux[spec_idx].cpu().numpy(),
        error[spec_idx].cpu().numpy(),
        mean_output.numpy(),
        std_output.numpy(),
        out_dir,
    )
    print(f"[M5] wrote repeated-noise sidecar {pdf_path} and {png_path}", flush=True)
    return {
        "spec_idx": int(spec_idx),
        "repeat": int(repeat),
        "seed": int(seed),
        "pdf": str(pdf_path),
        "png": str(png_path),
    }


def _render_fig1_vertical(*, wv: np.ndarray, noisy_s: np.ndarray,
                          clean_s: np.ndarray, mu_s: np.ndarray,
                          zoom_lo: float, zoom_hi: float,
                          snr_y: float, snr_y_db: float,
                          snr_mu: float, snr_mu_db: float,
                          snr_mu_ca: float, snr_mu_ca_db: float,
                          mu_offset: float,
                          parameter_text: str | None,
                          out_dir: Path,
                          label_fs: float, tick_fs: float,
                          legend_fs: float, snr_fs: float,
                          ) -> tuple[Path, Path]:
    """Vertical 5-row stacked layout for fig1 (replacement for horizontal).

    Per prof feedback (\\ld{Axis labels have to use the same size font as Figure
    captions} + \\ld{put them under each other for better resolution in
    wavelength}): full-page-width wavelength axis per row, line strokes thick
    enough that detail does not vanish at A&C single-column print scale.

    QC v2 changes vs v1:
      - 5-row flat grid (was 3 outer + sub-grids inside) → cleaner per-row sizing
      - figsize 7.5 × 11.0 (was 7.0 × 9.5) → more vertical room per panel
      - linewidth bumped (noisy 0.6 → 0.7, clean 1.5 → 1.5, xhat 0.85 → 1.1,
        residual 0.85 → 1.0, Ca II zoom 0.85 → 1.3) so curves register at print
      - hspace 0.45 outer → 0.42 between independent rows; sharex tied for the
        residual rows so axis ticks are not duplicated
      - dpi 100 → 130 vector output sharper
    """
    c_noisy = PAPER_FIG_COLORS["noisy"]
    c_clean = PAPER_FIG_COLORS["clean"]
    c_xhat = PAPER_FIG_COLORS["xhat"]
    c_residual = PAPER_FIG_COLORS["residual"]
    c_highlight = PAPER_FIG_COLORS["highlight"]
    alpha_ca = PAPER_FIG_HIGHLIGHT_ALPHA

    fig = plt.figure(figsize=(7.5, 9.0), dpi=130)
    fig.patch.set_alpha(0.0)

    # 2 outer groups: full (3 rows sharing x, no gap) / zoom (2 rows sharing x,
    # no gap). Small hspace=0.06 between groups for visual separation.
    outer = fig.add_gridspec(
        2, 1, height_ratios=[2.55, 1.55], hspace=0.06,
    )
    full_gs = outer[0].subgridspec(
        3, 1, height_ratios=[1.0, 1.0, 0.55], hspace=0.0,
    )
    zoom_gs = outer[1].subgridspec(
        2, 1, height_ratios=[1.0, 0.55], hspace=0.0,
    )

    ax_full_y = fig.add_subplot(full_gs[0])
    ax_full_x = fig.add_subplot(full_gs[1], sharex=ax_full_y)
    ax_full_r = fig.add_subplot(full_gs[2], sharex=ax_full_y)
    ax_zoom_x = fig.add_subplot(zoom_gs[0])
    ax_zoom_r = fig.add_subplot(zoom_gs[1], sharex=ax_zoom_x)

    for ax in (ax_full_y, ax_full_x, ax_full_r, ax_zoom_x, ax_zoom_r):
        ax.patch.set_alpha(0.0)

    full_xlim = (float(wv.min()), float(wv.max()))
    zoom_xlim = (zoom_lo - 10.0, zoom_hi + 10.0)

    # Row 1: full noisy + clean reference
    ax_full_y.plot(wv, noisy_s, c=c_noisy, lw=0.7)
    ax_full_y.plot(wv, clean_s, c=c_clean, lw=1.5)
    ax_full_y.axvspan(zoom_lo, zoom_hi, color=c_highlight, alpha=alpha_ca)
    ax_full_y.set(xlim=full_xlim, ylabel="Normalized Flux", ylim=(0.0, 1.0))
    ax_full_y.tick_params(axis="x", labelbottom=False)

    # Row 2: full clean + xhat (offset for readability)
    ax_full_x.plot(wv, clean_s, c=c_clean, lw=1.2)
    ax_full_x.plot(wv, mu_s - mu_offset, c=c_xhat, lw=1.1)
    ax_full_x.axvspan(zoom_lo, zoom_hi, color=c_highlight, alpha=alpha_ca)
    ax_full_x.set(xlim=full_xlim, ylabel="Normalized Flux")
    ax_full_x.tick_params(axis="x", labelbottom=False)

    # Row 3: full residual (shared x with row 2)
    ax_full_r.plot(wv, clean_s - mu_s, c=c_residual, lw=1.0)
    ax_full_r.axvspan(zoom_lo, zoom_hi, color=c_highlight, alpha=alpha_ca)
    ax_full_r.set(xlim=full_xlim, xlabel=r"Wavelength ($\AA$)",
                  ylabel=r"$x-\hat{x}$")

    # Row 4: Ca II zoom clean + xhat
    ax_zoom_x.plot(wv, clean_s, c=c_clean, lw=1.5)
    ax_zoom_x.plot(wv, mu_s - mu_offset, c=c_xhat, lw=1.3)
    ax_zoom_x.axvspan(zoom_lo, zoom_hi, color=c_highlight, alpha=alpha_ca)
    ax_zoom_x.set(xlim=zoom_xlim, ylabel="Normalized Flux")
    ax_zoom_x.tick_params(axis="x", labelbottom=False)
    ax_zoom_x.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_zoom_x.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # Row 5: Ca II zoom residual
    ax_zoom_r.plot(wv, clean_s - mu_s, c=c_residual, lw=1.3)
    ax_zoom_r.axvspan(zoom_lo, zoom_hi, color=c_highlight, alpha=alpha_ca)
    ax_zoom_r.set(xlim=zoom_xlim, xlabel=r"Wavelength ($\AA$)",
                  ylabel=r"$x-\hat{x}$")
    ax_zoom_r.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_zoom_r.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for ax in (ax_full_y, ax_full_x, ax_full_r, ax_zoom_x, ax_zoom_r):
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.xaxis.label.set_size(label_fs)
        ax.yaxis.label.set_size(label_fs)

    handles = [
        Line2D([], [], color=c_noisy, lw=4.0, label=r"$y$"),
        Line2D([], [], color=c_clean, lw=4.0, label=r"$x$"),
        Line2D([], [], color=c_xhat, lw=4.0, label=rf"$\hat{{x}} - {mu_offset:.1f}$"),
        Line2D([], [], color=c_residual, lw=4.0, label=r"$x-\hat{x}$"),
        Patch(facecolor=c_highlight, edgecolor="none", alpha=alpha_ca, label="Ca II Region"),
    ]
    fig.subplots_adjust(left=0.10, right=0.97, top=0.965, bottom=0.045)
    # Per user feedback (2026-04-30): legend + parameter_text both inside row 2
    # (ax_full_x) lower-area so they don't overlap. Row 2 plots clean+xhat
    # offset; the lower y region is empty, fits both lines cleanly stacked.
    ax_full_x.legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(handles), frameon=False, fontsize=legend_fs,
        handlelength=2.5, columnspacing=1.05, handletextpad=0.4,
        borderaxespad=0.3,
    )
    if parameter_text is not None:
        ax_full_x.text(
            0.5, 0.16, parameter_text,
            transform=ax_full_x.transAxes,
            ha="center", va="bottom", fontsize=label_fs,
        )

    # S/N labels right-aligned in upper-right of each conceptual row group.
    snr_x = 0.965
    fig.text(snr_x, ax_full_y.get_position().y1 - 0.005,
             rf"$\mathrm{{S/N}}[y] = {snr_y:.0f},\ {snr_y_db:.0f}\,\mathrm{{dB}}$",
             ha="right", va="top", fontsize=snr_fs)
    fig.text(snr_x, ax_full_x.get_position().y1 - 0.005,
             rf"$\mathrm{{S/N}}[\hat{{x}}] = {snr_mu:.0f},\ {snr_mu_db:.0f}\,\mathrm{{dB}}$",
             ha="right", va="top", fontsize=snr_fs)
    fig.text(snr_x, ax_zoom_x.get_position().y1 - 0.005,
             rf"$\mathrm{{S/N}}[\hat{{x}}_{{\mathrm{{Ca\,II}}}}] = {snr_mu_ca:.0f},\ {snr_mu_ca_db:.0f}\,\mathrm{{dB}}$",
             ha="right", va="top", fontsize=snr_fs)

    # parameter_text moved into ax_full_x above the legend (see lower-center
    # block) per user 2026-04-30 directive — top-of-figure caption removed.

    pdf_path = out_dir / "fig1_representative_vertical.pdf"
    png_path = out_dir / "fig1_representative_vertical.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, transparent=True)
    fig.savefig(png_path, dpi=160, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    return pdf_path, png_path


def compute_m5(realization: dict, flux: torch.Tensor, wave: torch.Tensor,
               out_dir: Path, parameter_text: str | None = None,
               representative_idx: int | None = None,
               lmodule: BlindspotLModule | None = None,
               error: torch.Tensor | None = None,
               device: torch.device | None = None,
               noise_level: float = 1.0,
               noise_ensemble_repeat: int = NOISE_ENSEMBLE_REPEAT) -> dict:
    # Pick median-SNR spectrum (not cherry-picked extreme); allow CLI override.
    per_spec_snr = []
    for i in range(flux.shape[0]):
        per_spec_snr.append(snr_pixel_db(flux[i:i + 1], realization["noisy"][i:i + 1]))
    per_spec_snr = np.array(per_spec_snr)
    if representative_idx is None:
        median_idx = int(np.argsort(per_spec_snr)[len(per_spec_snr) // 2])
        print(f"[M5] representative spec idx={median_idx} (median-SNR) "
              f"noisy_SNR={per_spec_snr[median_idx]:.2f} dB", flush=True)
    else:
        if not (0 <= representative_idx < flux.shape[0]):
            raise ValueError(f"--representative-idx={representative_idx} out of range "
                             f"[0, {flux.shape[0]})")
        median_idx = int(representative_idx)
        print(f"[M5] representative spec idx={median_idx} (user-requested) "
              f"noisy_SNR={per_spec_snr[median_idx]:.2f} dB", flush=True)

    clean_s = flux[median_idx].numpy()
    noisy_s = realization["noisy"][median_idx].numpy()
    mu_s = realization["mu_x"][median_idx].numpy()
    wv = wave.numpy()

    # HEADLINE S/N source: norm-ratio ‖ref‖/‖ref-est‖ → paper's 7.6 (noisy) / 122 (blindspot).
    # This is NOT snr_pixel_db (variance dB). dB form = 20·log10(this). See RUNBOOK trap 2.
    # NOTE: the aggregate distribution is guarded by sentinels/canonical_bundle.yaml;
    # canonical_test_metrics.csv only stores the variance-dB snr_pixel. Recompute from ep190 arrays.
    def snr_ratio(reference: np.ndarray, estimate: np.ndarray) -> float:
        residual = reference - estimate
        denom = np.linalg.norm(residual)
        if denom <= 0:
            return float("inf")
        return float(np.linalg.norm(reference) / denom)

    def snr_ratio_db(reference: np.ndarray, estimate: np.ndarray) -> float:
        residual = reference - estimate
        var_ref = max(np.var(reference, ddof=1), 1e-30)
        var_res = max(np.var(residual, ddof=1), 1e-30)
        return float(10.0 * np.log10(var_ref / var_res))

    zoom_lo, zoom_hi = 8475.0, 8680.0
    # === fixed_preview-style canonical fig1 layout (ported from
    # SpecDenoiser/scripts/make_fig1_blindspot_fixed_preview.py).
    # User additions: transparent figure background + S/N labels with no frame.
    zmask = (wv >= zoom_lo) & (wv <= zoom_hi)
    mu_offset = 0.2
    snr_y = snr_ratio(clean_s, noisy_s)
    snr_mu = snr_ratio(clean_s, mu_s)
    snr_mu_ca = snr_ratio(clean_s[zmask], mu_s[zmask]) if zmask.sum() > 0 else float("nan")
    snr_y_db = snr_ratio_db(clean_s, noisy_s)
    snr_mu_db = snr_ratio_db(clean_s, mu_s)
    snr_mu_ca_db = snr_ratio_db(clean_s[zmask], mu_s[zmask]) if zmask.sum() > 0 else float("nan")

    # Per prof feedback (\ld{Axis labels have to use the same size font as
    # Figure captions}): axis label / tick / S/N annotation font sizes match
    # AAS caption font (≈ 9pt). Legend and header annotations stay slightly
    # smaller / similar to keep visual hierarchy.
    label_fs = 9
    tick_fs = 9
    legend_fs = 9.0
    header_y = 0.985
    header_pad = 0.018
    snr_fs = 9

    c_noisy = PAPER_FIG_COLORS["noisy"]
    c_clean = PAPER_FIG_COLORS["clean"]
    c_xhat = PAPER_FIG_COLORS["xhat"]
    c_residual = PAPER_FIG_COLORS["residual"]
    c_highlight = PAPER_FIG_COLORS["highlight"]
    alpha_ca = PAPER_FIG_HIGHLIGHT_ALPHA

    fig = plt.figure(figsize=(19.2, 6.2), dpi=100)
    fig.patch.set_alpha(0.0)  # transparent figure background

    outer_gs = fig.add_gridspec(
        1, 3, width_ratios=[1.12, 1.0, 1.0], wspace=0.012,
    )
    mid_gs = outer_gs[0, 1].subgridspec(2, 1, hspace=0.0)
    right_gs = outer_gs[0, 2].subgridspec(2, 1, hspace=0.0)

    ax0 = fig.add_subplot(outer_gs[0, 0])
    ax1_top = fig.add_subplot(mid_gs[0, 0])
    ax1_bottom = fig.add_subplot(mid_gs[1, 0])
    ax2_top = fig.add_subplot(right_gs[0, 0], sharey=ax1_top)
    ax2_bottom = fig.add_subplot(right_gs[1, 0], sharey=ax1_bottom)

    for ax in (ax0, ax1_top, ax1_bottom, ax2_top, ax2_bottom):
        ax.patch.set_alpha(0.0)  # transparent axes background

    # Left: noisy + clean over full spectrum
    ax0.plot(wv, noisy_s, c=c_noisy, lw=0.6)
    ax0.plot(wv, clean_s, c=c_clean, lw=1.5)
    ax0.axvspan(zoom_lo, zoom_hi, color=c_highlight, alpha=alpha_ca)
    ax0.set(
        xlim=(float(wv.min()), float(wv.max())),
        xlabel=r"Wavelength ($\AA$)",
        ylabel="Normalized Flux",
        ylim=(0.0, 1.0),
    )

    # Middle top: clean + (xhat - offset)
    ax1_top.plot(wv, clean_s, c=c_clean, lw=0.85)
    ax1_top.plot(wv, mu_s - mu_offset, c=c_xhat, lw=0.85)
    ax1_top.axvspan(zoom_lo, zoom_hi, color=c_highlight, alpha=alpha_ca)
    ax1_top.set(xlim=(float(wv.min()), float(wv.max())), xticklabels=[])
    ax1_top.tick_params(axis="y", left=False, labelleft=False)

    # Middle bottom: residual (clean - xhat)
    ax1_bottom.plot(wv, clean_s - mu_s, c=c_residual, lw=0.85)
    ax1_bottom.axvspan(zoom_lo, zoom_hi, color=c_highlight, alpha=alpha_ca)
    ax1_bottom.set(xlim=(float(wv.min()), float(wv.max())), xlabel=r"Wavelength ($\AA$)")
    ax1_bottom.tick_params(axis="y", left=False, labelleft=False)

    # Right top: zoom on Ca II
    ax2_top.plot(wv, clean_s, c=c_clean, lw=0.85)
    ax2_top.plot(wv, mu_s - mu_offset, c=c_xhat, lw=0.85)
    ax2_top.axvspan(zoom_lo, zoom_hi, color=c_highlight, alpha=alpha_ca)
    ax2_top.set(xlim=(zoom_lo - 10.0, zoom_hi + 10.0), xticklabels=[])
    ax2_top.yaxis.tick_right()
    ax2_top.yaxis.set_label_position("right")
    ax2_top.set_ylabel("")
    ax2_top.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    ax2_top.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2_top.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # Right bottom: residual zoom
    ax2_bottom.plot(wv, clean_s - mu_s, c=c_residual, lw=0.85)
    ax2_bottom.axvspan(zoom_lo, zoom_hi, color=c_highlight, alpha=alpha_ca)
    ax2_bottom.set(xlim=(zoom_lo - 10.0, zoom_hi + 10.0), xlabel=r"Wavelength ($\AA$)")
    ax2_bottom.yaxis.tick_right()
    ax2_bottom.yaxis.set_label_position("right")
    ax2_bottom.set_ylabel("")
    ax2_bottom.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    ax2_bottom.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2_bottom.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for ax in (ax0, ax1_top, ax1_bottom, ax2_top, ax2_bottom):
        ax.tick_params(axis="both", labelsize=tick_fs)
    ax0.xaxis.label.set_size(label_fs)
    ax0.yaxis.label.set_size(label_fs)
    ax1_bottom.xaxis.label.set_size(label_fs)
    ax2_bottom.xaxis.label.set_size(label_fs)

    handles = [
        Line2D([], [], color=c_noisy, lw=8.0, label=r"$y$"),
        Line2D([], [], color=c_clean, lw=8.0, label=r"$x$"),
        Line2D([], [], color=c_xhat, lw=8.0, label=rf"$\hat{{x}} - {mu_offset:.1f}$"),
        Line2D([], [], color=c_residual, lw=8.0, label=r"$x-\hat{x}$"),
        Patch(facecolor=c_highlight, edgecolor="none", alpha=alpha_ca, label="Ca II Region"),
    ]
    fig.subplots_adjust(left=0.045, right=0.982, top=0.946, bottom=0.105, wspace=0.012)
    params_x = ax0.get_position().x0 + header_pad
    legend_x = ax2_top.get_position().x1 - header_pad
    if parameter_text is not None:
        fig.text(params_x, header_y, parameter_text, ha="left", va="top", fontsize=14.2)
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(legend_x, header_y),
        ncol=len(handles),
        frameon=False,
        fontsize=legend_fs,
        handlelength=2.9,
        columnspacing=0.95,
        handletextpad=0.5,
        borderaxespad=0.0,
    )

    # S/N labels — no frame, no facecolor (sit directly on transparent canvas).
    snr_ypos = ax1_bottom.get_position().y0 + 0.022
    ax0_center = 0.5 * (ax0.get_position().x0 + ax0.get_position().x1)
    ax1_center = 0.5 * (ax1_bottom.get_position().x0 + ax1_bottom.get_position().x1)
    ax2_center = 0.5 * (ax2_bottom.get_position().x0 + ax2_bottom.get_position().x1)
    fig.text(ax0_center, snr_ypos,
             rf"$\mathrm{{S/N}}[y] = {snr_y:.0f},\ {snr_y_db:.0f}\,\mathrm{{dB}}$",
             ha="center", va="bottom", fontsize=snr_fs)
    fig.text(ax1_center, snr_ypos,
             rf"$\mathrm{{S/N}}[\hat{{x}}] = {snr_mu:.0f},\ {snr_mu_db:.0f}\,\mathrm{{dB}}$",
             ha="center", va="bottom", fontsize=snr_fs)
    fig.text(ax2_center, snr_ypos,
             rf"$\mathrm{{S/N}}[\hat{{x}}_{{\mathrm{{Ca\,II}}}}] = {snr_mu_ca:.0f},\ {snr_mu_ca_db:.0f}\,\mathrm{{dB}}$",
             ha="center", va="bottom", fontsize=snr_fs)

    pdf_path = out_dir / "fig1_representative.pdf"
    png_path = out_dir / "fig1_representative.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.004, transparent=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0.004, transparent=True)
    plt.close(fig)
    print(f"[M5] wrote {pdf_path} and {png_path}", flush=True)

    # Vertical sibling layout (3-row stacked) — same data, same font sizes.
    # Per prof: "You can also put them under each other to have better
    # resolution in wavelength." We export both layouts so the editor can
    # choose the preferred one before submission.
    pdf_path_v, png_path_v = _render_fig1_vertical(
        wv=wv, noisy_s=noisy_s, clean_s=clean_s, mu_s=mu_s,
        zoom_lo=zoom_lo, zoom_hi=zoom_hi,
        snr_y=snr_y, snr_y_db=snr_y_db,
        snr_mu=snr_mu, snr_mu_db=snr_mu_db,
        snr_mu_ca=snr_mu_ca, snr_mu_ca_db=snr_mu_ca_db,
        mu_offset=mu_offset, parameter_text=parameter_text,
        out_dir=out_dir, label_fs=label_fs, tick_fs=tick_fs,
        legend_fs=legend_fs, snr_fs=snr_fs,
    )
    print(f"[M5] wrote {pdf_path_v} and {png_path_v}", flush=True)

    # Save the raw bundle (wave/clean/noisy/mu) so future agents can
    # re-render fig1 (idx=336) from a self-contained .npz without re-running
    # GPU inference. Filename mirrors the SpecDenoiser fixed_preview convention.
    bundle_path = out_dir / "bundle_idx336_fig1.npz"
    np.savez(
        bundle_path,
        wave=wv, clean=clean_s, noisy=noisy_s, mu=mu_s,
        idx=median_idx,
    )
    print(f"[M5] wrote bundle {bundle_path}", flush=True)

    noise_ensemble = None
    if lmodule is not None and error is not None and device is not None:
        noise_ensemble = compute_m5_noise_ensemble(
            lmodule=lmodule,
            flux=flux,
            error=error,
            wave=wave,
            spec_idx=median_idx,
            out_dir=out_dir,
            device=device,
            noise_level=noise_level,
            repeat=noise_ensemble_repeat,
        )
    return {
        "spec_idx": median_idx,
        "noisy_snr_dB": float(per_spec_snr[median_idx]),
        "pdf": str(pdf_path),
        "png": str(png_path),
        "noise_ensemble": noise_ensemble,
    }


# ----------------------------------------------------------------------------
# M6: blindspot_integrity_audit.json
# ----------------------------------------------------------------------------
@torch.no_grad()
def compute_m6(lmodule: BlindspotLModule, realization: dict, flux: torch.Tensor,
               error: torch.Tensor, device: torch.device, out_path: Path) -> dict:
    """
    Test 1 (permutation): replace center pixel of input with random; check delta on mu_x at that pixel.
    Test 2 (occlusion): replace y_i with mean(y); check output delta.
    For a true blindspot architecture, mu_x[i] should NOT depend on y[i].
    """
    n_test_spec = min(64, realization["noisy"].shape[0])
    L = realization["noisy"].shape[1]
    pix_to_test = [L // 4, L // 2, 3 * L // 4]  # 3 pixel positions

    perm_deltas, occlusion_deltas = [], []
    base_subset = realization["noisy"][:n_test_spec].clone()
    eb = error[:n_test_spec].to(device)
    base_input = base_subset.to(device)

    # base output
    processed_sigma = lmodule._process_error_input(eb)
    base_in_full = torch.cat([base_input.unsqueeze(1), processed_sigma], dim=1) if processed_sigma is not None else base_input.unsqueeze(1)
    base_out = lmodule.model(base_in_full)[:, 0, :]  # (n, L)

    torch.manual_seed(BASE_SEED + 9000)
    for pix in pix_to_test:
        # Permutation: shuffle center pixel value across spectra
        perm_input = base_input.clone()
        perm_input[:, pix] = base_input[torch.randperm(n_test_spec, device=device), pix]
        perm_in_full = torch.cat([perm_input.unsqueeze(1), processed_sigma], dim=1) if processed_sigma is not None else perm_input.unsqueeze(1)
        perm_out = lmodule.model(perm_in_full)[:, 0, :]
        perm_deltas.append((perm_out[:, pix] - base_out[:, pix]).abs().cpu().numpy())

        # Occlusion: replace center pixel with mean of spectrum
        occ_input = base_input.clone()
        occ_input[:, pix] = base_input.mean(dim=-1)
        occ_in_full = torch.cat([occ_input.unsqueeze(1), processed_sigma], dim=1) if processed_sigma is not None else occ_input.unsqueeze(1)
        occ_out = lmodule.model(occ_in_full)[:, 0, :]
        occlusion_deltas.append((occ_out[:, pix] - base_out[:, pix]).abs().cpu().numpy())

    perm_deltas_all = np.concatenate(perm_deltas)
    occ_deltas_all = np.concatenate(occlusion_deltas)
    THRESHOLD = 1e-4  # blindspot guarantee should give ≈ machine-precision delta

    audit = {
        "tested_pixels": pix_to_test,
        "n_test_specs": n_test_spec,
        "permutation_test": {
            "delta_at_center_max": float(perm_deltas_all.max()),
            "delta_at_center_mean": float(perm_deltas_all.mean()),
            "delta_at_center_p95": float(np.percentile(perm_deltas_all, 95)),
            "PASS": bool(perm_deltas_all.max() < THRESHOLD),
            "threshold_used": THRESHOLD,
            "interpretation": "Center-pixel permutation should NOT change mu_x[center] for a true blindspot.",
        },
        "occlusion_test": {
            "delta_at_center_max": float(occ_deltas_all.max()),
            "delta_at_center_mean": float(occ_deltas_all.mean()),
            "delta_at_center_p95": float(np.percentile(occ_deltas_all, 95)),
            "PASS": bool(occ_deltas_all.max() < THRESHOLD),
            "threshold_used": THRESHOLD,
            "interpretation": "Replacing center with spectrum mean should NOT change mu_x[center] for a true blindspot.",
        },
    }
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[M6] wrote {out_path} (perm PASS={audit['permutation_test']['PASS']}, occ PASS={audit['occlusion_test']['PASS']})", flush=True)
    return audit


# ----------------------------------------------------------------------------
# M7: sigma_x_diagnostic.json + Figure 2 (dual-head archival only)
# ----------------------------------------------------------------------------
def compute_m7(realization: dict, flux: torch.Tensor, error: torch.Tensor,
               out_dir: Path) -> dict:
    """sigma_x interpretation diagnostic — supports the §5 verdict table.
    Build histogram, compute calibration scatter slope, sigma_x vs sigma_n ratio.
    Also produces fig2_sigma_x_diagnostic.{pdf,png} for §7 P6.

    Uses sigma_x = exp(log_sigma_x) clamped to training range (already done
    in run_inference_one_realization). Residual is computed against mu_x
    (the blindspot-pure output, since sigma_x is paired with mu_x in the
    NLL likelihood, not denoised).
    """
    sigma_x = realization["sigma_x"].numpy()  # (B, L), already clamped + finite
    mu_x = realization["mu_x"].numpy()
    flux_np = flux.numpy()
    err_np = error.numpy()

    sigma_x_flat = sigma_x.flatten()
    residual = np.abs(flux_np - mu_x).flatten()
    # Drop any remaining non-finite (defensive)
    finite_mask = np.isfinite(sigma_x_flat) & np.isfinite(residual)
    sigma_x_flat = sigma_x_flat[finite_mask]
    residual = residual[finite_mask]

    # Calibration scatter slope (log-log; aleatoric would imply slope ~ 1 in linear)
    # Fit y = a + b*x in linear space using least squares
    valid = (sigma_x_flat > 1e-8) & (residual >= 0) & np.isfinite(sigma_x_flat) & np.isfinite(residual)
    if valid.sum() < 100:
        slope, intercept, r2 = float("nan"), float("nan"), float("nan")
    else:
        x = sigma_x_flat[valid]
        y = residual[valid]
        # subsample for speed
        if len(x) > 200_000:
            rng = np.random.default_rng(BASE_SEED + 7777)
            sub = rng.choice(len(x), size=200_000, replace=False)
            x = x[sub]; y = y[sub]
        A = np.vstack([np.ones_like(x), x]).T
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        intercept_, slope_ = coeffs[0], coeffs[1]
        ypred = A @ coeffs
        ss_res = float(((y - ypred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-30
        r2 = 1.0 - ss_res / ss_tot
        slope, intercept = float(slope_), float(intercept_)

    # sigma_x vs sigma_n per-spec: sigma_x_per_spec = mean over L; sigma_n = mean(error) per spec
    sigma_x_per_spec = sigma_x.mean(axis=-1)
    sigma_n_per_spec = err_np.mean(axis=-1)
    ratio = sigma_x_per_spec / np.maximum(sigma_n_per_spec, 1e-12)

    diagnostic = {
        "sigma_x_distribution": {
            "mean": float(sigma_x_flat.mean()),
            "median": float(np.median(sigma_x_flat)),
            "iqr_low": float(np.percentile(sigma_x_flat, 25)),
            "iqr_high": float(np.percentile(sigma_x_flat, 75)),
            "p5": float(np.percentile(sigma_x_flat, 5)),
            "p95": float(np.percentile(sigma_x_flat, 95)),
            "min": float(sigma_x_flat.min()),
            "max": float(sigma_x_flat.max()),
        },
        "calibration_scatter": {
            "linear_fit_slope": slope,
            "linear_fit_intercept": intercept,
            "r2": r2,
            "interpretation": (
                "If sigma_x were calibrated aleatoric uncertainty, residual ~ sigma_x with slope ~ 1. "
                "A weak/non-unit slope is consistent with the auxiliary / loss-weight controller reading."
            ),
            "n_pairs_used": int(min(valid.sum(), 200_000)),
        },
        "sigma_x_vs_sigma_n_per_spec": {
            "ratio_mean": float(ratio.mean()),
            "ratio_median": float(np.median(ratio)),
            "ratio_iqr_low": float(np.percentile(ratio, 25)),
            "ratio_iqr_high": float(np.percentile(ratio, 75)),
            "n_spec": int(flux.shape[0]),
        },
        "training_loss_anchor": {
            "lam_sig": 0.5,
            "loss_name": "E1 (Laplace NLL with sigma_y^2 = sigma_x^2 + sigma_n^2)",
            "ckpt_val_snr_mu_x": 117,
            "interpretation": (
                "Per DEC-2026-04-27-005, sigma_x is the dual-head auxiliary output; "
                "loss anchoring (lam_sig=0.5) constrains its scale relative to sigma_n. "
                "main.tex Section 5 verdict table interprets sigma_x as 'auxiliary / loss-weight controller'."
            ),
        },
    }
    json_path = out_dir / "sigma_x_diagnostic.json"
    with open(json_path, "w") as f:
        json.dump(diagnostic, f, indent=2)

    # Figure 2: σ_x histogram + residual-vs-σ_x scatter
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    axs[0].hist(sigma_x_flat, bins=80, color="C0", alpha=0.8, density=True)
    axs[0].set_xlabel(r"$\sigma_x$")
    axs[0].set_ylabel("Density")
    axs[0].set_title("Distribution of $\\sigma_x$")
    axs[0].grid(alpha=0.2)

    # 2D density-style scatter (bin) for residual vs sigma_x
    if valid.sum() > 100:
        x_plot = sigma_x_flat[valid]
        y_plot = residual[valid]
        if len(x_plot) > 50_000:
            rng = np.random.default_rng(BASE_SEED + 7778)
            sub = rng.choice(len(x_plot), size=50_000, replace=False)
            x_plot = x_plot[sub]; y_plot = y_plot[sub]
        axs[1].hexbin(x_plot, y_plot, gridsize=60, bins="log", cmap="Blues",
                      mincnt=1)
        # 1:1 reference line
        xmax = float(np.percentile(x_plot, 99))
        axs[1].plot([0, xmax], [0, xmax], color="k", ls="--", lw=0.7, label="1:1 (calibrated aleatoric)")
        axs[1].plot([0, xmax], [intercept, intercept + slope * xmax],
                    color="r", ls="-", lw=1.0, label=f"Fit slope={slope:.3f}")
        axs[1].set_xlim(0, xmax)
        axs[1].set_ylim(0, max(xmax, float(np.percentile(y_plot, 99))))
        axs[1].legend(loc="upper left", fontsize=8, frameon=False)
    axs[1].set_xlabel(r"$\sigma_x$")
    axs[1].set_ylabel(r"$|y_i - \mu_{x,i}|$")
    axs[1].set_title("Residual vs $\\sigma_x$")
    axs[1].grid(alpha=0.2)

    fig.tight_layout()
    pdf_path = out_dir / "fig2_sigma_x_diagnostic.pdf"
    png_path = out_dir / "fig2_sigma_x_diagnostic.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[M7] wrote {json_path}, {pdf_path}, {png_path}", flush=True)
    diagnostic["fig2_pdf"] = str(pdf_path)
    diagnostic["fig2_png"] = str(png_path)
    return diagnostic


# ----------------------------------------------------------------------------
# Raw per-spec dump (for paper-figure reproduction)
# ----------------------------------------------------------------------------
def _per_spec_linear_snr(clean: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-spectrum linear S/N = ‖clean‖₂ / ‖clean − x‖₂. clean (N,L), x (N,L) → (N,)."""
    num = np.linalg.norm(clean, axis=-1)
    den = np.linalg.norm(clean - x, axis=-1)
    return num / np.maximum(den, 1e-12)


def dump_raw_h5(out_path: Path, *, wave: torch.Tensor, flux: torch.Tensor,
                error: torch.Tensor, realizations: list[dict],
                test_path: str, ckpt: str, output_mode: str) -> None:
    """Dump per-spec arrays so SpecPlotter can reproduce result_image/s1 figures locally.

    Layout: K=N_realizations, N=N_specs, L=pixel length.
    For mu-only models, user-facing denoised = mu_x (matches main.tex Method §4 and
    compute_m1_m9 line 320 which uses r["mu_x"]). The 'denoised' key is the
    posterior-fused output used by archival dual-head runs only.
    """
    K = len(realizations)
    flux_np = flux.cpu().numpy()
    N, L = flux_np.shape
    noisy = np.stack([r["noisy"].cpu().numpy().astype(np.float32) for r in realizations], axis=0)
    mu_x = np.stack([r["mu_x"].cpu().numpy().astype(np.float32) for r in realizations], axis=0)
    denoised = np.stack([r["denoised"].cpu().numpy().astype(np.float32) for r in realizations], axis=0)
    sigma_x = np.stack([r["sigma_x"].cpu().numpy().astype(np.float32) for r in realizations], axis=0)

    # Per-spec per-realization linear S/N (plotter.plot_snr_improve uses linear ratio)
    user_facing = mu_x if output_mode == "mu" else denoised
    snr_noisy_lin = np.stack([_per_spec_linear_snr(flux_np, noisy[k]) for k in range(K)], axis=0)
    snr_user_lin = np.stack([_per_spec_linear_snr(flux_np, user_facing[k]) for k in range(K)], axis=0)

    # Per-spec metadata from the test split (T_eff/log_g/M_H/mag/snr/redshift if present)
    meta = {}
    try:
        meta_df = pd.read_hdf(test_path, start=0, stop=int(N))
        for col in ("T_eff", "M_H", "log_g", "mag", "snr", "redshift", "z"):
            if col in meta_df.columns:
                meta[col] = np.asarray(meta_df[col].values, dtype=np.float64)
    except Exception as e:
        print(f"[dump-raw] meta load skipped: {e}", flush=True)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("wave", data=wave.cpu().numpy())
        f.create_dataset("flux", data=flux_np, compression="gzip", compression_opts=4)
        f.create_dataset("error", data=error.cpu().numpy(), compression="gzip", compression_opts=4)
        f.create_dataset("noisy", data=noisy, compression="gzip", compression_opts=4)
        f.create_dataset("mu_x", data=mu_x, compression="gzip", compression_opts=4)
        f.create_dataset("denoised", data=denoised, compression="gzip", compression_opts=4)
        f.create_dataset("sigma_x", data=sigma_x, compression="gzip", compression_opts=4)
        f.create_dataset("snr_noisy_linear", data=snr_noisy_lin)
        f.create_dataset("snr_user_facing_linear", data=snr_user_lin)
        for k, v in meta.items():
            f.create_dataset(f"meta/{k}", data=v)
        f.attrs["test_path"] = test_path
        f.attrs["ckpt"] = ckpt
        f.attrs["output_mode"] = output_mode
        f.attrs["n_realizations"] = K
        f.attrs["n_specs"] = N
        f.attrs["pixel_length"] = L
        f.attrs["DEC_lock"] = "DEC-2026-04-27-005"
    print(f"[dump-raw] wrote {out_path} (K={K} N={N} L={L} output_mode={output_mode})", flush=True)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory on /datascope/.../mag1921_canonical_2026-04-27/")
    ap.add_argument("--ckpt", default=CANONICAL_CKPT)
    ap.add_argument("--config", default=CANONICAL_CONFIG)
    ap.add_argument("--label", default="canonical",
                    help="Short run label written into inference_meta.json.")
    ap.add_argument("--n-realizations", type=int, default=N_NOISE_REALIZATIONS)
    ap.add_argument("--num-test-samples", type=int, default=CANONICAL_NUM_TEST_SAMPLES)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--only-m5", action="store_true",
                    help="Generate only the representative-spectrum figure and repeated-noise sidecar (M5).")
    ap.add_argument("--noise-ensemble-repeat", type=int, default=NOISE_ENSEMBLE_REPEAT,
                    help="Number of fresh noise draws for the M5 repeated-noise sidecar.")
    ap.add_argument("--dump-raw", type=Path, default=None,
                    help="If given, dump per-spec arrays (clean+noisy[K,N,L]+mu_x+denoised+meta) to this h5 path "
                         "for paper-figure reproduction (snr_improvement / EW residuals / line preservation).")
    ap.add_argument("--representative-idx", type=int, default=None,
                    help="Override representative-spectrum index for M5 figure. "
                         "Default: median-SNR auto-pick. Use this to pin a specific test-set "
                         "spectrum (e.g. the idx=4 / idx=8 picks used in paper draft iterations).")
    ap.add_argument("--test-path", type=str, default=CANONICAL_TEST_PATH,
                    help="Override the test-split HDF5 path from config. Useful for "
                         "rendering fig1 against a fixed-magnitude validation set "
                         "(e.g. bosz50000/mag215/val_10k) while keeping the canonical "
                         "ckpt/config. Default is the paper canonical z1/test_10k_1 split.")
    ap.add_argument("--use-stored-noisy", action="store_true",
                    help="Use dataset/arrays/noisy/value from the HDF5 test split instead "
                         "of drawing fresh Gaussian noise. Intended for fixed-noisy "
                         "test_10k files; requires --n-realizations 1.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.use_stored_noisy and args.n_realizations != 1:
        raise ValueError("--use-stored-noisy represents one fixed noisy realization; "
                         "set --n-realizations 1")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    print(f"[gpu] using CUDA device {args.gpu} (visible as cuda:0); "
          f"torch.cuda.is_available={torch.cuda.is_available()}", flush=True)

    # Resolve config path relative to repo root
    config_path = REPO_ROOT / args.config
    config = load_config(str(config_path))
    config["train"]["gpus"] = 1
    config["train"]["debug"] = 0
    config["data"]["num_test_samples"] = args.num_test_samples
    if args.test_path is not None:
        test_path = args.test_path
        config["data"]["test_path"] = test_path
        print(f"[config] test_path OVERRIDE -> {test_path}", flush=True)
    else:
        test_path = config["data"]["test_path"]
    noise_level = float(config["noise"].get("noise_level", 1.0))
    print(f"[config] regime={test_path.split('/')[-3]} noise_level={noise_level} N_test={args.num_test_samples}", flush=True)

    # Load data
    data = load_test_data(test_path, args.num_test_samples,
                          use_stored_noisy=args.use_stored_noisy)
    flux = data["flux"]
    error = data["error"]
    wave = data["wave"]
    stored_noisy = data["stored_noisy"]

    # Load model
    lmodule = build_lmodule(config, args.ckpt, device, config_label=str(config_path))

    t_start = time.time()

    # Pull log_sigma_clamp_max from loss config (training default = 3.0)
    log_sigma_clamp_max = float(config.get("loss", {}).get("log_sigma_clamp_max", 3.0))
    print(f"[config] log_sigma_clamp_max = {log_sigma_clamp_max} (sigma_x ≤ {np.exp(log_sigma_clamp_max):.3f})", flush=True)

    # Run N noise realizations
    realizations = []
    for k in range(args.n_realizations):
        seed = BASE_SEED + k
        print(f"[realization {k+1}/{args.n_realizations}] seed={seed}", flush=True)
        r = run_inference_one_realization(lmodule, flux, error, noise_level, seed, device,
                                          log_sigma_clamp_max=log_sigma_clamp_max,
                                          stored_noisy=stored_noisy)
        realizations.append(r)
        if k == 0:
            # Diagnostic: show ranges so we catch any silent scaling issue early.
            print(
                f"[diag] flux: min={flux.min():.3f} max={flux.max():.3f} mean={flux.mean():.3f}; "
                f"noisy: min={r['noisy'].min():.3f} max={r['noisy'].max():.3f} mean={r['noisy'].mean():.3f}; "
                f"mu_x: min={r['mu_x'].min():.3f} max={r['mu_x'].max():.3f} mean={r['mu_x'].mean():.3f}; "
                f"denoised: min={r['denoised'].min():.3f} max={r['denoised'].max():.3f} mean={r['denoised'].mean():.3f}; "
                f"sigma_x: min={r['sigma_x'].min():.4g} max={r['sigma_x'].max():.4g} median={r['sigma_x'].median():.4g}",
                flush=True,
            )

    canonical = realizations[0]  # for per-spec quantities

    output_mode = str(config.get("model", {}).get("output_mode", "mu_sigma"))

    if args.dump_raw is not None:
        args.dump_raw.parent.mkdir(parents=True, exist_ok=True)
        dump_raw_h5(args.dump_raw, wave=wave, flux=flux, error=error,
                    realizations=realizations, test_path=test_path,
                    ckpt=str(args.ckpt), output_mode=output_mode)

    # M5
    parameter_text = None
    try:
        row = pd.read_hdf(test_path, start=0, stop=1)  # warm up pandas HDF reader
        del row
        row = pd.read_hdf(test_path, start=0, stop=args.num_test_samples)
        if args.representative_idx is None:
            median_idx_preview = int(np.argsort([
                snr_pixel_db(flux[i:i + 1], canonical["noisy"][i:i + 1]) for i in range(flux.shape[0])
            ])[len(flux) // 2])
        else:
            median_idx_preview = int(args.representative_idx)
        meta_row = row.iloc[median_idx_preview]
        # Stellar parameters only — no S/N_meta. The HDF5 'snr' field is a
        # dataset-supplied SNR estimate, NOT the paper §4 norm-ratio
        # S/N(y)=‖x‖₂/‖x-y‖₂ that the right-margin S/N labels report; mixing
        # the two confuses the reader (user 2026-04-30 explicit).
        parameter_parts = [
            rf"$T_{{\rm eff}}={float(meta_row['T_eff']):.0f}$",
            rf"$M_{{\rm H}}={float(meta_row['M_H']):.1f}$",
            rf"$\log g={float(meta_row['log_g']):.1f}$",
            rf"$\mathrm{{mag}}={float(meta_row['mag']):.1f}$",
        ]
        parameter_text = "   ".join(parameter_parts)
    except Exception as e:
        print(f"[M5] parameter-text load skipped: {e}", flush=True)
    m5 = compute_m5(canonical, flux, wave, args.out,
                    parameter_text=parameter_text,
                    representative_idx=args.representative_idx,
                    lmodule=lmodule,
                    error=error,
                    device=device,
                    noise_level=noise_level,
                    noise_ensemble_repeat=args.noise_ensemble_repeat)
    if args.only_m5:
        m1 = m2 = m3 = m4 = m6 = m7 = None
    else:
        # M1 + M9
        m1 = compute_m1_m9(realizations, flux, args.out / "canonical_test_metrics.csv")
        # M2
        m2 = compute_m2(canonical, flux, wave, args.out / "ew_preservation_test.csv")
        # M3
        m3 = compute_m3(canonical, flux, args.out / "polarity_test.csv")
        # M4
        m4 = compute_m4(canonical, flux, wave, args.out / "hard_window_metric.csv")
        # M6
        m6 = compute_m6(lmodule, canonical, flux, error, device, args.out / "blindspot_integrity_audit.json")
        # M7 is meaningful only for dual-head models that emit sigma_x.
        if output_mode == "mu":
            print("[M7] skipped: output_mode='mu' has no learned sigma_x head", flush=True)
            m7 = None
        else:
            m7 = compute_m7(canonical, flux, error, args.out)

    wall_s = time.time() - t_start

    # Provenance manifest
    loss_name = str(config.get("loss", {}).get("name", "unknown"))
    sigma_mode = str(config.get("model", {}).get("sigma_input_mode", "unknown"))
    regime = test_path.split("/")[-3]
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "ckpt": str(args.ckpt),
        "config_path": str(config_path),
        "test_split_path": test_path,
        "regime": regime,
        "model_architecture": "mu-only single-head" if output_mode == "mu" else "dual-head (mu_x, log_sigma_x)",
        "loss": loss_name,
        "input_sigma_mode": sigma_mode,
        "n_test_samples": int(flux.shape[0]),
        "n_noise_realizations": args.n_realizations,
        "noise_source": "stored_hdf5_noisy" if args.use_stored_noisy else "fresh_gaussian_seeded",
        "base_seed": BASE_SEED,
        "wall_seconds": wall_s,
        "DEC_lock": "DEC-2026-04-27-005",
        "decision_log": "research/BlindSpotDenoiser/plan/2026-04-27-self-decision-log.md",
        "dispatch_brief": "research/BlindSpotDenoiser/plan/2026-04-27-mag1921-canonical-measurement-brief.md",
        "summary": {
            "m5_representative_idx": m5["spec_idx"],
            "m5_noisy_snr_dB": m5["noisy_snr_dB"],
        },
    }
    if not args.only_m5:
        meta["summary"].update({
            "m1_snr_pixel": {
                "noisy_dB": m1["snr_pixel"]["noisy_input_mean"],
                "denoised_dB": m1["snr_pixel"]["denoised_output_mean"],
                "delta_dB": m1["snr_pixel"]["delta_mean"],
                "denoised_ci95": m1["snr_pixel"]["denoised_output_ci95"],
            },
            "m6_blindspot_perm_pass": m6["permutation_test"]["PASS"],
            "m6_blindspot_occlusion_pass": m6["occlusion_test"]["PASS"],
        })
        if m7 is not None:
            meta["summary"].update({
                "m7_sigma_x_median": m7["sigma_x_distribution"]["median"],
                "m7_calibration_slope": m7["calibration_scatter"]["linear_fit_slope"],
            })
    with open(args.out / "inference_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.only_m5:
        print(f"\n[done] representative-spectrum figure written to {args.out}", flush=True)
        print(f"[done] wall time: {wall_s/60:.1f} min", flush=True)
        print(f"[done] M5 representative idx={m5['spec_idx']} noisy_SNR={m5['noisy_snr_dB']:.2f} dB", flush=True)
    else:
        print(f"\n[done] paper-facing measurements written to {args.out}", flush=True)
        print(f"[done] wall time: {wall_s/60:.1f} min", flush=True)
        print(f"[done] M1 SNR_pixel: noisy={m1['snr_pixel']['noisy_input_mean']:.2f} dB → "
              f"denoised={m1['snr_pixel']['denoised_output_mean']:.2f} dB "
              f"(Δ={m1['snr_pixel']['delta_mean']:+.2f} dB ± "
              f"{m1['snr_pixel']['delta_ci95']:.3f} dB 95% CI, N={args.n_realizations})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
