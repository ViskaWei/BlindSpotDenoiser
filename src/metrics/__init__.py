"""src.metrics - frozen canonical metric implementations for BlindSpot paper.

Currently exposes a single freeze: `freeze_v1`. New conventions must
ship as a separate freeze module (e.g. `freeze_v2.py`); never edit
freeze_v1 in place. See `paper/experiments_post_pro_review.md` MF-0.
"""
from __future__ import annotations

from src.metrics.freeze_v1 import (
    FREEZE_VERSION,
    METRIC_CODE_HASH,
    CA_II_LINES,
    CA_II_LINE_NAMES,
    O2_BAND,
    LINE_HALF_WIDTH,
    CONTINUUM_INNER,
    CONTINUUM_HALF_WIDTH,
    POLARITY_THRESHOLD_FRAC,
    DIAGNOSTIC_BUFFER,
    recon_snr_linear,
    recon_snr_db,
    recon_rmse,
    recon_mae,
    ca2_ew_relerr_per_line,
    ca2_ew_relerr_triplet_mean,
    signed_ew_bias_per_line,
    polarity_flip_rate,
    o2_aband_snr,
    o2_aband_ew_relerr,
    line_depth_signed_bias,
    continuum_rms_outside_diagnostic,
)

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
