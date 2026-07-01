#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sec5_4_ew_paired_bootstrap_ci.py — paired-bootstrap CI for Sec 5.4 EW table.

Adds 95% paired-bootstrap CI columns to the per-line + triplet-mean Ca II
EW relative-error medians reported in main.tex Table 2 (`tab:main_results`),
without modifying the frozen `compute_m2` in `inference.py`.

Pipeline:
  1. Load M1a canonical ckpt + z1/test_10k_1 split (same paper-facing
     defaults as `inference.py`).
  2. Run one inference realization (seed=42, mu_x = denoised output).
  3. Compute per-spec EW for clean / noisy / mu_x at each Ca II line via
     the same `equivalent_width` utility used in compute_m2.
  4. Paired bootstrap (1000 iterations, seed=0): same resample idx applied
     to noisy and denoised arrays simultaneously, so the CI on Δ captures
     the per-spec correlation. Triplet mean = arithmetic mean of the three
     per-line medians inside each bootstrap draw.
  5. Emit `data/ew_preservation_test_with_ci.csv` (8 rows: 3 lines + triplet
     mean × 2 metrics noisy/denoised, plus a Δ row each = 8 rows total).

Run on volta04 (single GPU):
    cd /home/swei20/R_abuv-Propulsion/research/BlindSpotDenoiser
    source /datascope/subaru/user/swei20/miniconda3/etc/profile.d/conda.sh
    conda activate specvit
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
        python3 scripts/sec5_4_ew_paired_bootstrap_ci.py \\
        --out_dir /datascope/subaru/user/swei20/blindspot/results/sec5_4_ew_ci_z1_10k_FIXED_2026-06-21/

Output:
    sec5_4_ew_paired_bootstrap_ci.csv
    sec5_4_ew_paired_bootstrap_ci.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.inference import (  # noqa: E402
    BASE_SEED,
    CANONICAL_CKPT,
    CANONICAL_CONFIG,
    LINE_LIST,
    MASK_PATH,
    build_lmodule,
    equivalent_width,
    load_test_data,
    run_inference_one_realization,
)
from src.utils import load_config  # noqa: E402

EW_CLEAN_FLOOR = 1e-3
N_BOOT = 1000
BOOT_SEED = 0


def _per_spec_relerr(ew_clean: torch.Tensor, ew_x: torch.Tensor,
                     floor: float = EW_CLEAN_FLOOR) -> np.ndarray:
    """Per-spec |EW_X - EW_clean| / |EW_clean|, NaN where line absent."""
    denom = ew_clean.abs()
    mask = torch.isfinite(ew_clean) & torch.isfinite(ew_x) & (denom > floor)
    out = np.full(ew_clean.shape[0], np.nan, dtype=np.float64)
    e = ((ew_x - ew_clean).abs() / denom).cpu().numpy()
    out[mask.cpu().numpy()] = e[mask.cpu().numpy()]
    return out


def paired_bootstrap_line(noisy: np.ndarray, denoised: np.ndarray,
                          n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Paired bootstrap CI on per-line medians and Δ = denoised - noisy median.

    Same resample idx applied to noisy and denoised arrays simultaneously so
    that the CI on Δ captures per-spec correlation (a hard spectrum inflates
    both errors).
    """
    finite = np.isfinite(noisy) & np.isfinite(denoised)
    n = int(finite.sum())
    yn = noisy[finite]
    yd = denoised[finite]
    rng = np.random.default_rng(seed)
    med_n = np.empty(n_boot, dtype=np.float64)
    med_d = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        med_n[b] = np.median(yn[idx])
        med_d[b] = np.median(yd[idx])
    delta = med_d - med_n
    return {
        "n": n,
        "noisy_median": float(np.median(yn)),
        "noisy_ci_lo": float(np.percentile(med_n, 2.5)),
        "noisy_ci_hi": float(np.percentile(med_n, 97.5)),
        "denoised_median": float(np.median(yd)),
        "denoised_ci_lo": float(np.percentile(med_d, 2.5)),
        "denoised_ci_hi": float(np.percentile(med_d, 97.5)),
        "delta_median": float(np.median(yd) - np.median(yn)),
        "delta_ci_lo": float(np.percentile(delta, 2.5)),
        "delta_ci_hi": float(np.percentile(delta, 97.5)),
    }


def paired_bootstrap_triplet(per_line_noisy: list[np.ndarray],
                             per_line_denoised: list[np.ndarray],
                             n_boot: int = N_BOOT,
                             seed: int = BOOT_SEED) -> dict:
    """Triplet-mean paired bootstrap.

    Across the 3 Ca II lines, each bootstrap iteration draws ONE resample idx
    of size n_total (per-spec union). Per-line medians are computed within the
    finite subset of that line; the triplet mean is the arithmetic mean of
    the three per-line medians inside the same bootstrap draw.
    """
    n_total = per_line_noisy[0].shape[0]
    rng = np.random.default_rng(seed)
    med_n = np.empty(n_boot, dtype=np.float64)
    med_d = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_total, n_total)
        line_meds_n = []
        line_meds_d = []
        for n_arr, d_arr in zip(per_line_noisy, per_line_denoised):
            sample_n = n_arr[idx]
            sample_d = d_arr[idx]
            keep = np.isfinite(sample_n) & np.isfinite(sample_d)
            line_meds_n.append(np.median(sample_n[keep]))
            line_meds_d.append(np.median(sample_d[keep]))
        med_n[b] = np.mean(line_meds_n)
        med_d[b] = np.mean(line_meds_d)
    delta = med_d - med_n
    # point estimate: mean of full-sample per-line medians
    full_n = []
    full_d = []
    for n_arr, d_arr in zip(per_line_noisy, per_line_denoised):
        keep = np.isfinite(n_arr) & np.isfinite(d_arr)
        full_n.append(float(np.median(n_arr[keep])))
        full_d.append(float(np.median(d_arr[keep])))
    return {
        "n_per_line": [int((np.isfinite(n_arr) & np.isfinite(d_arr)).sum())
                       for n_arr, d_arr in zip(per_line_noisy, per_line_denoised)],
        "noisy_triplet_mean": float(np.mean(full_n)),
        "noisy_ci_lo": float(np.percentile(med_n, 2.5)),
        "noisy_ci_hi": float(np.percentile(med_n, 97.5)),
        "denoised_triplet_mean": float(np.mean(full_d)),
        "denoised_ci_lo": float(np.percentile(med_d, 2.5)),
        "denoised_ci_hi": float(np.percentile(med_d, 97.5)),
        "delta_triplet": float(np.mean(full_d) - np.mean(full_n)),
        "delta_ci_lo": float(np.percentile(delta, 2.5)),
        "delta_ci_hi": float(np.percentile(delta, 97.5)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default=CANONICAL_CKPT)
    ap.add_argument("--config", default=CANONICAL_CONFIG)
    ap.add_argument("--test_split",
                    default="/datascope/subaru/user/swei20/data/bosz50000/z1/"
                            "mag205_225_lowT_1M/test_10k_1/dataset.h5")
    ap.add_argument("--num_samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--boot_seed", type=int, default=BOOT_SEED)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plan] ckpt = {args.ckpt}", flush=True)
    print(f"[plan] config = {args.config}", flush=True)
    print(f"[plan] test_split = {args.test_split}", flush=True)
    print(f"[plan] out_dir = {out_dir}", flush=True)

    t0 = time.time()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_test_data(args.test_split, args.num_samples, mask_path=MASK_PATH,
                          use_stored_noisy=False)
    flux, error, wave = data["flux"], data["error"], data["wave"]
    lmodule = build_lmodule(config, args.ckpt, device, config_label=args.config)

    print("[infer] running one realization (seed=42)", flush=True)
    realization = run_inference_one_realization(
        lmodule, flux, error, noise_level=1.0, seed=args.seed, device=device,
        stored_noisy=None,
    )
    noisy = realization["noisy"]
    denoised = realization["mu_x"]  # main.tex §4: μ_x is the reported denoised output

    per_line_results = []
    per_line_noisy_arrs = []
    per_line_denoised_arrs = []
    for name, center in LINE_LIST:
        ew_clean = equivalent_width(flux, wave, center)
        ew_noisy = equivalent_width(noisy, wave, center)
        ew_denoised = equivalent_width(denoised, wave, center)
        rel_noisy = _per_spec_relerr(ew_clean, ew_noisy)
        rel_denoised = _per_spec_relerr(ew_clean, ew_denoised)
        boot = paired_bootstrap_line(rel_noisy, rel_denoised,
                                     n_boot=args.n_boot, seed=args.boot_seed)
        boot["line_name"] = name
        boot["wavelength_A"] = center
        per_line_results.append(boot)
        per_line_noisy_arrs.append(rel_noisy)
        per_line_denoised_arrs.append(rel_denoised)
        print(f"[bootstrap] {name}: noisy_med={boot['noisy_median']:.4f} "
              f"[{boot['noisy_ci_lo']:.4f},{boot['noisy_ci_hi']:.4f}]  "
              f"denoised_med={boot['denoised_median']:.4f} "
              f"[{boot['denoised_ci_lo']:.4f},{boot['denoised_ci_hi']:.4f}]  "
              f"Δ={boot['delta_median']:.4f} "
              f"[{boot['delta_ci_lo']:.4f},{boot['delta_ci_hi']:.4f}]",
              flush=True)

    triplet = paired_bootstrap_triplet(per_line_noisy_arrs, per_line_denoised_arrs,
                                       n_boot=args.n_boot, seed=args.boot_seed)
    triplet["line_name"] = "triplet_mean"
    triplet["wavelength_A"] = float("nan")
    print(f"[bootstrap] triplet_mean: noisy={triplet['noisy_triplet_mean']:.4f} "
          f"[{triplet['noisy_ci_lo']:.4f},{triplet['noisy_ci_hi']:.4f}]  "
          f"denoised={triplet['denoised_triplet_mean']:.4f} "
          f"[{triplet['denoised_ci_lo']:.4f},{triplet['denoised_ci_hi']:.4f}]  "
          f"Δ={triplet['delta_triplet']:.4f} "
          f"[{triplet['delta_ci_lo']:.4f},{triplet['delta_ci_hi']:.4f}]",
          flush=True)

    csv_rows = []
    for rec in per_line_results:
        csv_rows.append({
            "line_name": rec["line_name"],
            "wavelength_A": rec["wavelength_A"],
            "n": rec["n"],
            "noisy_median": rec["noisy_median"],
            "noisy_ci_lo": rec["noisy_ci_lo"],
            "noisy_ci_hi": rec["noisy_ci_hi"],
            "denoised_median": rec["denoised_median"],
            "denoised_ci_lo": rec["denoised_ci_lo"],
            "denoised_ci_hi": rec["denoised_ci_hi"],
            "delta_median": rec["delta_median"],
            "delta_ci_lo": rec["delta_ci_lo"],
            "delta_ci_hi": rec["delta_ci_hi"],
        })
    csv_rows.append({
        "line_name": "triplet_mean",
        "wavelength_A": np.nan,
        "n": int(np.median(triplet["n_per_line"])),
        "noisy_median": triplet["noisy_triplet_mean"],
        "noisy_ci_lo": triplet["noisy_ci_lo"],
        "noisy_ci_hi": triplet["noisy_ci_hi"],
        "denoised_median": triplet["denoised_triplet_mean"],
        "denoised_ci_lo": triplet["denoised_ci_lo"],
        "denoised_ci_hi": triplet["denoised_ci_hi"],
        "delta_median": triplet["delta_triplet"],
        "delta_ci_lo": triplet["delta_ci_lo"],
        "delta_ci_hi": triplet["delta_ci_hi"],
    })
    df = pd.DataFrame(csv_rows)
    csv_path = out_dir / "sec5_4_ew_paired_bootstrap_ci.csv"
    df.to_csv(csv_path, index=False)
    print(f"[write] {csv_path}", flush=True)

    summary = {
        "ckpt": args.ckpt,
        "config": args.config,
        "test_split": args.test_split,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "n_test_specs": int(noisy.shape[0]),
        "ew_clean_floor": EW_CLEAN_FLOOR,
        "wall_seconds": float(time.time() - t0),
        "per_line": per_line_results,
        "triplet_mean": triplet,
    }
    json_path = out_dir / "sec5_4_ew_paired_bootstrap_ci.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[write] {json_path}", flush=True)
    print(f"[done] wall = {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
