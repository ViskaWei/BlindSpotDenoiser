"""Shared CPU helpers for BlindSpot paper post-review experiments.

The helpers intentionally avoid importing the training stack.  E1 and E4 are
CPU-only evaluation/measurement jobs and must use the frozen MF-0 metric module.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from src.metrics import freeze_v1 as mfv1


DEFAULT_DATA_ROOT = (
    "/datascope/subaru/user/swei20/data/bosz50000/z1/mag205_225_lowT_1M"
)
DEFAULT_MASK_PATH = "/datascope/subaru/user/swei20/model/bosz50000_mask.npy"
DEFAULT_OUT_ROOT = "/datascope/subaru/user/swei20/BlindSpot/paper-experiments"
DEFAULT_TEST_SEEDS = tuple(range(42, 52))


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: Mapping[str, Any], *, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, sort_keys=True)
        f.write("\n")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: str | Path, *, block_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def load_mask(mask_path: str | Path = DEFAULT_MASK_PATH) -> np.ndarray:
    mask = np.load(mask_path)
    return mask.astype(bool)


def _fill_nan_edges(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, copy=True)
    if np.isnan(arr[:, 0]).any():
        arr[:, 0] = arr[:, 1]
    if np.isnan(arr[:, -1]).any():
        arr[:, -1] = arr[:, -2]
    return arr


def load_split(
    data_root: str | Path,
    split: str,
    *,
    mask: np.ndarray | None = None,
    n: int | None = None,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    import h5py

    path = Path(data_root) / split / "dataset.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        wave = np.asarray(f["spectrumdataset/wave"][()], dtype=np.float64)
        flux_ds = f["dataset/arrays/flux/value"]
        err_ds = f["dataset/arrays/error/value"]
        limit = flux_ds.shape[0] if n is None else min(int(n), flux_ds.shape[0])
        flux = np.asarray(flux_ds[:limit], dtype=dtype)
        error = np.asarray(err_ds[:limit], dtype=dtype)
    flux = np.clip(_fill_nan_edges(flux), 0.0, None)
    error = _fill_nan_edges(error)
    if mask is not None:
        wave = wave[mask]
        flux = flux[:, mask]
        error = error[:, mask]
    meta = {
        "path": str(path),
        "split": split,
        "n_loaded": int(flux.shape[0]),
        "n_pixels": int(flux.shape[1]),
        "wave_hash": sha256_array(wave),
        "flux_sample_hash": sha256_array(
            np.concatenate([flux[:8], flux[-8:]], axis=0)
            if flux.shape[0] >= 16
            else flux
        ),
        "error_sample_hash": sha256_array(
            np.concatenate([error[:8], error[-8:]], axis=0)
            if error.shape[0] >= 16
            else error
        ),
    }
    return wave, flux, error, meta


def add_noise(flux: np.ndarray, error: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    noisy = flux + rng.standard_normal(flux.shape, dtype=np.float32) * error
    return noisy.astype(np.float32, copy=False)


def metric_arrays(
    x_clean: np.ndarray,
    x_pred: np.ndarray,
    wavelength: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "recon_snr_linear": mfv1.recon_snr_linear(x_clean, x_pred, wavelength),
        "recon_snr_db": mfv1.recon_snr_db(x_clean, x_pred, wavelength),
        "recon_rmse": mfv1.recon_rmse(x_clean, x_pred, wavelength),
        "recon_mae": mfv1.recon_mae(x_clean, x_pred, wavelength),
        "ca2_ew_relerr_per_line": mfv1.ca2_ew_relerr_per_line(
            x_clean, x_pred, wavelength
        ),
        "ca2_ew_relerr_triplet_mean": mfv1.ca2_ew_relerr_triplet_mean(
            x_clean, x_pred, wavelength
        ),
        "polarity_flip_rate": mfv1.polarity_flip_rate(x_clean, x_pred, wavelength),
        "o2_aband_snr": mfv1.o2_aband_snr(x_clean, x_pred, wavelength),
        "o2_aband_ew_relerr": mfv1.o2_aband_ew_relerr(x_clean, x_pred, wavelength),
        "line_depth_signed_bias": mfv1.line_depth_signed_bias(
            x_clean, x_pred, wavelength
        ),
        "continuum_rms_outside_diagnostic": mfv1.continuum_rms_outside_diagnostic(
            x_clean, x_pred, wavelength
        ),
        "signed_ew_bias_per_line": mfv1.signed_ew_bias_per_line(
            x_clean, x_pred, wavelength
        ),
    }


def finite_summary(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "q05": float("nan"), "q95": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q95": float(np.quantile(arr, 0.95)),
    }


def scalar_metric_summary(metrics: Mapping[str, np.ndarray], prefix: str = "") -> Dict[str, float]:
    row: Dict[str, float] = {}
    for key, arr in metrics.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            stats = finite_summary(arr)
            for stat_name, value in stats.items():
                row[f"{prefix}{key}_{stat_name}"] = value
        elif arr.ndim == 2 and arr.shape[1] == 3:
            names = mfv1.CA_II_LINE_NAMES
            for j, name in enumerate(names):
                stats = finite_summary(arr[:, j])
                for stat_name, value in stats.items():
                    row[f"{prefix}{key}_{name}_{stat_name}"] = value
    return row


def aggregate_seed_metrics(seed_metrics: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, float]:
    row: Dict[str, float] = {}
    if not seed_metrics:
        return row
    keys = seed_metrics[0].keys()
    for key in keys:
        stacked = np.stack([np.asarray(m[key]) for m in seed_metrics], axis=0)
        flat_stats = finite_summary(stacked)
        for stat_name, value in flat_stats.items():
            row[f"{key}_{stat_name}"] = value
        if stacked.ndim >= 2:
            per_seed = []
            for i in range(stacked.shape[0]):
                arr = stacked[i]
                arr = arr[np.isfinite(arr)]
                per_seed.append(np.mean(arr) if arr.size else np.nan)
            per_seed_arr = np.asarray(per_seed, dtype=np.float64)
            if np.isfinite(per_seed_arr).sum() >= 2:
                row[f"{key}_seed_mean_ci95"] = float(
                    1.96 * np.nanstd(per_seed_arr, ddof=1) / np.sqrt(len(per_seed_arr))
                )
            else:
                row[f"{key}_seed_mean_ci95"] = float("nan")
    return row


def metrics_to_jsonable(metrics: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    return {key: np.asarray(value).tolist() for key, value in metrics.items()}


def metric_contract() -> Dict[str, Any]:
    return {
        "freeze_version": mfv1.FREEZE_VERSION,
        "metric_code_hash": mfv1.METRIC_CODE_HASH,
        "ca_ii_lines": list(mfv1.CA_II_LINES),
        "o2_band": list(mfv1.O2_BAND),
        "line_half_width": mfv1.LINE_HALF_WIDTH,
        "polarity_threshold_frac": mfv1.POLARITY_THRESHOLD_FRAC,
    }


def parse_seed_list(text: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(text, str):
        return tuple(int(x.strip()) for x in text.split(",") if x.strip())
    return tuple(int(x) for x in text)
