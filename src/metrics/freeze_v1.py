"""freeze_v1.py - frozen canonical metric implementation for BlindSpot paper.

This module is the single source of truth for every metric reported in
`paper/main.tex` Table 1, Table 2, Abstract HEADLINE numbers and §5/§7
narrative. All downstream experiments (E1 / E2 / E3 / E5 / E6 / E7) and
the figure-generation code in F-1 must call into this module so that
columns are apples-to-apples comparable.

Provenance:
    Formulas are pulled verbatim from `paper/metric_definitions.md` and
    `scripts/inference.py` (the volta04 inference
    driver that produced the published Table 2 numbers under DEC-2026-04-27-005).
    Each function docstring cites the section of metric_definitions.md
    that defines it. Any change to a formula here MUST also update
    metric_definitions.md plus a new freeze_vN module - never edit in place.

Spec: `paper/experiments_post_pro_review.md` MF-0 (lines 50-101).

Conventions:
    - Inputs are numpy arrays. `x_clean` is the noiseless reference,
      `X_pred` is either the noisy observation `y` or a denoised estimate.
      Shape is `(N_spectra, L_pixels)` for both. Wavelength is `(L_pixels,)`.
    - Returns are 1-D numpy arrays of length N_spectra (per-spec scalars)
      or shape `(N_spectra, N_lines)` for per-line metrics.
    - All averages reported in the paper are per-spec then mean over spec
      (not pixel-pooled). The aggregation step is left to the caller.

This file is FROZEN: source code hash is exposed as METRIC_CODE_HASH and
written to `data/metric_freeze_v1.sha256`. A future agent must NOT edit
this file in-place; create freeze_v2.py if a new convention is needed.
"""
from __future__ import annotations

import hashlib
import inspect
import sys
from typing import Optional

import numpy as np

FREEZE_VERSION = "freeze_v1"

# Module-level constants - same numbers as scripts/inference.py.
CA_II_LINES = (8498.0, 8542.0, 8662.0)  # wavelengths in Angstroms
CA_II_LINE_NAMES = ("CaII_T1", "CaII_T2", "CaII_T3")
O2_BAND = (7590.0, 7700.0)  # telluric O2 A-band (lo, hi) in Angstroms
LINE_HALF_WIDTH = 5.0  # +/- A around line center for EW integration
CONTINUUM_INNER = 5.0  # inner edge of continuum band (excludes line region)
CONTINUUM_HALF_WIDTH = 15.0  # outer edge of continuum band
POLARITY_THRESHOLD_FRAC = 0.02  # |x - median| < 2% * median = continuum, no defined polarity
DIAGNOSTIC_BUFFER = 10.0  # +/- A buffer around Ca II + O2 windows for continuum_rms_outside


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_inputs(x_clean: np.ndarray, X_pred: np.ndarray) -> None:
    if x_clean.shape != X_pred.shape:
        raise ValueError(
            f"freeze_v1: x_clean.shape={x_clean.shape} != X_pred.shape={X_pred.shape}"
        )
    if x_clean.ndim != 2:
        raise ValueError(
            f"freeze_v1: expected (N_spectra, L) inputs, got ndim={x_clean.ndim}"
        )


def _per_spec_var(arr: np.ndarray) -> np.ndarray:
    """Unbiased variance along the wavelength axis, per-spectrum.

    Matches `inference.py:snr_pixel_db` which uses
    torch.var(unbiased=True) along dim=-1.
    """
    return np.var(arr, axis=-1, ddof=1)


def _local_continuum_fit(flux: np.ndarray, wave: np.ndarray,
                         line_center: float,
                         line_half: float = LINE_HALF_WIDTH,
                         cont_inner: float = CONTINUUM_INNER,
                         cont_outer: float = CONTINUUM_HALF_WIDTH):
    """Per-spectrum local linear continuum c(lambda) = a + b*lambda.

    Reproduces `inference.py:equivalent_width` lines 124-158.
    Continuum band: lambda in [center - cont_outer, center - cont_inner]
                  union [center + cont_inner, center + cont_outer].
    Line band: lambda in [center - line_half, center + line_half].

    Returns (a, b, line_mask, line_wave_subgrid). a, b have shape (N,).
    """
    line_mask = (wave >= line_center - line_half) & (wave <= line_center + line_half)
    cont_left = (wave >= line_center - cont_outer) & (wave <= line_center - cont_inner)
    cont_right = (wave >= line_center + cont_inner) & (wave <= line_center + cont_outer)
    cont_mask = cont_left | cont_right
    if line_mask.sum() == 0 or cont_mask.sum() < 4:
        return None
    w_cont = wave[cont_mask]
    f_cont = flux[:, cont_mask]
    # Least-squares via design matrix A = [1, lambda]. Reproduces the torch.linalg.pinv
    # branch used in the inference driver (numerically equivalent for 2-parameter fits).
    A = np.stack([np.ones_like(w_cont), w_cont], axis=-1)  # (Ncont, 2)
    AtA_inv = np.linalg.pinv(A.T @ A)  # (2, 2)
    coeffs = (AtA_inv @ A.T) @ f_cont.T  # (2, N)
    a = coeffs[0]  # (N,)
    b = coeffs[1]  # (N,)
    return a, b, line_mask, wave[line_mask]


def _equivalent_width(flux: np.ndarray, wave: np.ndarray,
                      line_center: float,
                      line_half: float = LINE_HALF_WIDTH,
                      cont_inner: float = CONTINUUM_INNER,
                      cont_outer: float = CONTINUUM_HALF_WIDTH) -> np.ndarray:
    """Per-spectrum EW (Angstrom) at line_center, positive=absorption.

    Implements metric_definitions.md Section 6 formula:
        EW(lambda_0) = integral_{lambda_0 - line_half}^{lambda_0 + line_half}
                       [1 - X(lambda) / c(lambda)] d lambda
    Trapezoidal integration on the local wavelength sub-grid.
    Returns (N_spectra,) array; NaN if continuum/line band empty.
    """
    fit = _local_continuum_fit(flux, wave, line_center, line_half,
                               cont_inner, cont_outer)
    if fit is None:
        return np.full((flux.shape[0],), np.nan)
    a, b, line_mask, w_line = fit
    f_line = flux[:, line_mask]  # (N, Nline)
    # c(lambda) at each wavelength in line band, per spec
    cont_at_line = a[:, None] + b[:, None] * w_line[None, :]  # (N, Nline)
    cont_at_line = np.clip(cont_at_line, 1e-12, None)
    integrand = 1.0 - f_line / cont_at_line
    # trapezoidal on irregular wavelength grid
    dw = np.diff(w_line)  # (Nline-1,)
    ew = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dw[None, :]
    return ew.sum(axis=-1)


# ---------------------------------------------------------------------------
# 1. recon_snr_linear / recon_snr_db (metric_definitions.md Section 1, Section 5)
# ---------------------------------------------------------------------------
def recon_snr_linear(x_clean: np.ndarray, X_pred: np.ndarray,
                     wavelength: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-spectrum L2 norm ratio S/N = ||x||_2 / ||x - X||_2 (linear).

    Section 5 of metric_definitions.md (snr_mu_x). Reproduces the
    `_per_spec_linear_snr` helper in inference.py:892.
    Returns (N,) array; +inf if denominator vanishes.
    """
    _check_inputs(x_clean, X_pred)
    num = np.linalg.norm(x_clean, axis=-1)
    den = np.linalg.norm(x_clean - X_pred, axis=-1)
    out = np.full_like(num, np.inf, dtype=np.float64)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def recon_snr_db(x_clean: np.ndarray, X_pred: np.ndarray,
                 wavelength: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-spectrum SNR in dB based on variance ratio.

    Section 1 of metric_definitions.md:
        snr_pixel = 10 * log10( var_lambda(x) / var_lambda(x - X) )
    Reproduces inference.py:snr_pixel_db (lines 99-104),
    but evaluates per-spectrum (the driver applies .mean() across spectra
    after the per-spec ratio - that mean is left to the caller here).
    """
    _check_inputs(x_clean, X_pred)
    res = x_clean - X_pred
    var_clean = _per_spec_var(x_clean)
    var_res = np.clip(_per_spec_var(res), 1e-30, None)
    return 10.0 * np.log10(var_clean / var_res)


# ---------------------------------------------------------------------------
# 2. recon_rmse / recon_mae (metric_definitions.md Sections 2, 4)
# ---------------------------------------------------------------------------
def recon_rmse(x_clean: np.ndarray, X_pred: np.ndarray,
               wavelength: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-spectrum RMSE in flux units. Section 2 of metric_definitions.md.

    Alias to the value used by inference.py:rmse_pixel
    (line 107-108) at the per-spec resolution.
    """
    _check_inputs(x_clean, X_pred)
    return np.sqrt(np.mean((x_clean - X_pred) ** 2, axis=-1))


def recon_mae(x_clean: np.ndarray, X_pred: np.ndarray,
              wavelength: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-spectrum MAE in flux units. Section 4 of metric_definitions.md.

    Mirrors inference.py:mae_pixel (line 111-112).
    """
    _check_inputs(x_clean, X_pred)
    return np.mean(np.abs(x_clean - X_pred), axis=-1)


# ---------------------------------------------------------------------------
# 3. Ca II equivalent-width metrics (metric_definitions.md Section 6)
# ---------------------------------------------------------------------------
def ca2_ew_relerr_per_line(x_clean: np.ndarray, X_pred: np.ndarray,
                           wavelength: np.ndarray,
                           ew_clean_floor: float = 1e-3) -> np.ndarray:
    """Per-spec, per-line Ca II EW relative error: |EW_X - EW_clean|/|EW_clean|.

    Section 6 of metric_definitions.md (per-line degradation). Lines 8498,
    8542, 8662 A. Returns shape (N_spectra, 3). NaN where |EW_clean| <
    ew_clean_floor (line absent).
    """
    _check_inputs(x_clean, X_pred)
    out = np.zeros((x_clean.shape[0], len(CA_II_LINES)), dtype=np.float64)
    for j, center in enumerate(CA_II_LINES):
        ew_c = _equivalent_width(x_clean, wavelength, center)
        ew_x = _equivalent_width(X_pred, wavelength, center)
        denom = np.where(np.abs(ew_c) > ew_clean_floor, np.abs(ew_c), np.nan)
        out[:, j] = np.abs(ew_x - ew_c) / denom
    return out


def ca2_ew_relerr_triplet_mean(x_clean: np.ndarray, X_pred: np.ndarray,
                               wavelength: np.ndarray,
                               ew_clean_floor: float = 1e-3) -> np.ndarray:
    """Per-spec mean of Ca II triplet EW relative errors (Section 6).

    Returns (N_spectra,). nanmean across the 3 lines per spec; NaN if all
    three lines are below the EW floor for that spec.
    """
    per_line = ca2_ew_relerr_per_line(x_clean, X_pred, wavelength,
                                      ew_clean_floor=ew_clean_floor)
    with np.errstate(invalid="ignore", all="ignore"):
        return np.nanmean(per_line, axis=-1)


def signed_ew_bias_per_line(x_clean: np.ndarray, X_pred: np.ndarray,
                            wavelength: np.ndarray) -> np.ndarray:
    """Per-spec, per-line signed EW bias: EW_X - EW_clean (no abs, no normalization).

    NEW per spec L70. Reveals systematic over- or under-absorption
    direction that |.|/|EW_clean| hides. Returns (N_spectra, 3).
    """
    _check_inputs(x_clean, X_pred)
    out = np.zeros((x_clean.shape[0], len(CA_II_LINES)), dtype=np.float64)
    for j, center in enumerate(CA_II_LINES):
        ew_c = _equivalent_width(x_clean, wavelength, center)
        ew_x = _equivalent_width(X_pred, wavelength, center)
        out[:, j] = ew_x - ew_c
    return out


# ---------------------------------------------------------------------------
# 4. Polarity flip rate (metric_definitions.md Section 7)
# ---------------------------------------------------------------------------
def polarity_flip_rate(x_clean: np.ndarray, X_pred: np.ndarray,
                       wavelength: Optional[np.ndarray] = None,
                       eps_frac: float = POLARITY_THRESHOLD_FRAC) -> np.ndarray:
    """Per-spectrum polarity-flip rate vs per-spec median reference.

    Section 7 of metric_definitions.md. Pixels where |x - median| <
    eps_frac * median are continuum (no defined polarity) and excluded.
    A flip is sign(x - median) != sign(X - median) on remaining pixels.
    Returns (N_spectra,) flip rate in [0, 1]; NaN if a spec has zero
    polarity-bearing pixels.

    Reproduces inference.py:polarity_flip_rate (line 161-185)
    but per-spectrum rather than pooled across all pixels (per-spec then
    mean over spec is the contract enforced by MF-0 - aggregation is the
    caller's responsibility).
    """
    _check_inputs(x_clean, X_pred)
    median = np.median(x_clean, axis=-1, keepdims=True)
    eps = np.abs(eps_frac * median)
    devs_clean = x_clean - median
    devs_x = X_pred - median
    sign_clean = np.where(np.abs(devs_clean) < eps, 0.0, np.sign(devs_clean))
    sign_x = np.sign(devs_x)
    flips = (sign_clean != sign_x) & (sign_clean != 0)
    valid = sign_clean != 0
    n_valid = valid.sum(axis=-1)
    n_flips = flips.sum(axis=-1)
    out = np.full(x_clean.shape[0], np.nan, dtype=np.float64)
    nz = n_valid > 0
    out[nz] = n_flips[nz] / n_valid[nz]
    return out


# ---------------------------------------------------------------------------
# 5. Hard-window O2 A-band (metric_definitions.md Section 8)
# ---------------------------------------------------------------------------
def _o2_band_mask(wavelength: np.ndarray) -> np.ndarray:
    return (wavelength >= O2_BAND[0]) & (wavelength <= O2_BAND[1])


def o2_aband_snr(x_clean: np.ndarray, X_pred: np.ndarray,
                 wavelength: np.ndarray) -> np.ndarray:
    """Per-spec recon_snr_linear restricted to the 7590-7700 A O2 A-band.

    Section 8 of metric_definitions.md. Returns (N_spectra,). +inf where
    band-restricted residual norm is zero.
    """
    _check_inputs(x_clean, X_pred)
    band = _o2_band_mask(wavelength)
    if band.sum() == 0:
        return np.full(x_clean.shape[0], np.nan, dtype=np.float64)
    return recon_snr_linear(x_clean[:, band], X_pred[:, band])


def o2_aband_ew_relerr(x_clean: np.ndarray, X_pred: np.ndarray,
                       wavelength: np.ndarray,
                       ew_clean_floor: float = 1e-3) -> np.ndarray:
    """Per-spec EW relative error inside the O2 A-band hard window.

    Section 8 of metric_definitions.md: line center = (7590+7700)/2 = 7645,
    half-width = (7700-7590)/2 = 55 A, continuum band offset by +5 A inner
    and +25 A outer (matching inference.py compute_m4
    lines 447-460).
    """
    _check_inputs(x_clean, X_pred)
    lo, hi = O2_BAND
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    cont_inner = half + 5.0
    cont_outer = half + 25.0
    ew_c = _equivalent_width(x_clean, wavelength, center,
                             line_half=half,
                             cont_inner=cont_inner,
                             cont_outer=cont_outer)
    ew_x = _equivalent_width(X_pred, wavelength, center,
                             line_half=half,
                             cont_inner=cont_inner,
                             cont_outer=cont_outer)
    denom = np.where(np.abs(ew_c) > ew_clean_floor, np.abs(ew_c), np.nan)
    return np.abs(ew_x - ew_c) / denom


# ---------------------------------------------------------------------------
# 6. Line-depth signed bias at Ca II line centers (NEW per spec L66)
# ---------------------------------------------------------------------------
def line_depth_signed_bias(x_clean: np.ndarray, X_pred: np.ndarray,
                           wavelength: np.ndarray) -> np.ndarray:
    """Per-spec, per-line signed bias at the line center: X(lambda_c) - x(lambda_c).

    NEW per spec L66. Uses nearest-pixel value at each Ca II line center.
    Positive = X overshoots clean, negative = X is shallower than clean.
    Returns (N_spectra, 3).
    """
    _check_inputs(x_clean, X_pred)
    out = np.zeros((x_clean.shape[0], len(CA_II_LINES)), dtype=np.float64)
    for j, center in enumerate(CA_II_LINES):
        idx = int(np.argmin(np.abs(wavelength - center)))
        out[:, j] = X_pred[:, idx] - x_clean[:, idx]
    return out


# ---------------------------------------------------------------------------
# 7. Continuum RMS outside diagnostic windows (NEW per spec L67)
# ---------------------------------------------------------------------------
def continuum_rms_outside_diagnostic(x_clean: np.ndarray, X_pred: np.ndarray,
                                     wavelength: np.ndarray,
                                     ca2_half: float = LINE_HALF_WIDTH + DIAGNOSTIC_BUFFER,
                                     o2_buffer: float = DIAGNOSTIC_BUFFER) -> np.ndarray:
    """Per-spec RMS of (X - x) on pixels outside Ca II + O2 diagnostic windows.

    NEW per spec L67. Probes whether X tracks the continuum where there
    are no diagnostic features. Window definitions:
      - Ca II exclusion: union of [lambda_c - ca2_half, lambda_c + ca2_half]
        for each Ca II line (default ca2_half = 5 A line + 10 A buffer = 15 A).
      - O2 exclusion: [O2_BAND[0] - o2_buffer, O2_BAND[1] + o2_buffer]
        (default 7580-7710 A).
    Returns (N_spectra,). NaN if every pixel is excluded.
    """
    _check_inputs(x_clean, X_pred)
    excl = np.zeros_like(wavelength, dtype=bool)
    for center in CA_II_LINES:
        excl |= (wavelength >= center - ca2_half) & (wavelength <= center + ca2_half)
    excl |= (wavelength >= O2_BAND[0] - o2_buffer) & (wavelength <= O2_BAND[1] + o2_buffer)
    keep = ~excl
    if keep.sum() == 0:
        return np.full(x_clean.shape[0], np.nan, dtype=np.float64)
    diff = X_pred[:, keep] - x_clean[:, keep]
    return np.sqrt(np.mean(diff ** 2, axis=-1))


# ---------------------------------------------------------------------------
# Module hash (refuse-merge guard)
# ---------------------------------------------------------------------------
def _compute_module_hash() -> str:
    """sha256 of this module's source. Callers should compare this against
    `data/metric_freeze_v1.sha256` before merging cross-experiment results."""
    src = inspect.getsource(sys.modules[__name__])
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


METRIC_CODE_HASH = _compute_module_hash()


__all__ = [
    "FREEZE_VERSION",
    "METRIC_CODE_HASH",
    "CA_II_LINES",
    "CA_II_LINE_NAMES",
    "O2_BAND",
    "LINE_HALF_WIDTH",
    "CONTINUUM_INNER",
    "CONTINUUM_HALF_WIDTH",
    "POLARITY_THRESHOLD_FRAC",
    "DIAGNOSTIC_BUFFER",
    "recon_snr_linear",
    "recon_snr_db",
    "recon_rmse",
    "recon_mae",
    "ca2_ew_relerr_per_line",
    "ca2_ew_relerr_triplet_mean",
    "signed_ew_bias_per_line",
    "polarity_flip_rate",
    "o2_aband_snr",
    "o2_aband_ew_relerr",
    "line_depth_signed_bias",
    "continuum_rms_outside_diagnostic",
]
