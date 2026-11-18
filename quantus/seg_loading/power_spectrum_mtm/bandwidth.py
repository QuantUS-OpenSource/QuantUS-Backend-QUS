"""Bandwidth of the multitaper spectra in the ROI windows, and the band shared by sample and
reference (S1Study DetermineBW_Reference.m + DetermineBW_ROI.m).

Every window gets FindBWEdges on the mean reference spectrum and on the sample spectrum. A window
uses one sample/reference ratio, so its band is the intersection:
    shared low = max(sample fLow, reference fLow),  shared high = min(sample fHigh, reference fHigh)
and it is usable if that band exists and both spectra are >= 5 dB above their band level.
Over the usable windows:
    strict = [max shared low, min shared high]         (every frequency valid in every window)
    pctl   = [95th pct shared low, 5th pct shared high] (robust to a few noisy windows)
both rounded inward to 0.1 MHz. Percentiles are MATLAB's prctile (numpy method "hazen").

The MATLAB evaluates the windows of DetermineInclusionPSERs / DetermineAttenuationPER; here the
mask is every window whose centre lies in the ROI (default: every window of the grid).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..correlation_length.first_pass import find_bw_edges, inpolygon

MIN_SIGNAL_POWER_DB = 5.0


def bw_options(avoid_freq: Sequence[float] = (), w_mhz: float = 0.0) -> dict:
    """FindBWEdges options of DetermineBandwidth.m: AvoidFreq, SearchAbove = max(AvoidFreq) (or
    -Inf), Guard = W (the multitaper smoothing spreads a null over +/-W), no automatic notch
    detection (DipProminence = Inf)."""
    avoid = tuple(np.atleast_1d(np.asarray(avoid_freq, dtype=float)).ravel())
    return dict(avoid_freq=avoid, search_above=max(avoid) if avoid else -np.inf, guard=float(w_mhz),
                dip_prominence=np.inf)


def window_bands(spectra: np.ndarray, freq: np.ndarray, mask: Optional[np.ndarray] = None, **options):
    """FindBWEdges on every masked window. spectra: (n_ax, n_lat, n_freq), linear.
    Returns (f_low, f_high, signal_power_db), (n_ax, n_lat) each, NaN outside the mask."""
    n_ax, n_lat, _ = spectra.shape
    if mask is None:
        mask = np.ones((n_ax, n_lat), dtype=bool)
    low, high, sp = (np.full((n_ax, n_lat), np.nan) for _ in range(3))
    for i, j in zip(*np.nonzero(mask)):
        low[i, j], high[i, j], info = find_bw_edges(spectra[i, j], freq, **options)
        sp[i, j] = info["signal_power_db"]
    return low, high, sp


def matlab_prctile(x: np.ndarray, p: float) -> float:
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    return float(np.percentile(x, p, method="hazen")) if x.size else np.nan


def _round_in(b, tol: float = 1e-6) -> tuple:
    return float(np.ceil(b[0] * 10 - tol) / 10), float(np.floor(b[1] * 10 + tol) / 10)


@dataclass
class ROIBandwidth:
    """DetermineBW_ROI outputs (window arrays are (n_ax, n_lat), NaN outside the mask)."""
    mask: np.ndarray
    sample_low: np.ndarray
    sample_high: np.ndarray
    sample_signal_power_db: np.ndarray
    ref_low: np.ndarray
    ref_high: np.ndarray
    ref_signal_power_db: np.ndarray
    shared_low: np.ndarray
    shared_high: np.ndarray
    shared_usable: np.ndarray
    shared_strict: tuple                 # rounded inward to 0.1 MHz
    shared_pctl: tuple
    sample_band: tuple                   # 95th pct fLow / 5th pct fHigh, sample only
    reference_band: tuple
    n_windows: int
    n_usable: int
    options: dict = field(default_factory=dict)
    min_signal_power_db: float = MIN_SIGNAL_POWER_DB

    def describe(self) -> str:
        return (f"{self.n_usable} of {self.n_windows} ROI windows usable in sample and reference; shared band "
                f"strict {self.shared_strict[0]:.1f} - {self.shared_strict[1]:.1f} MHz | 95/5 pct "
                f"{self.shared_pctl[0]:.1f} - {self.shared_pctl[1]:.1f} MHz")


def roi_window_mask(grid, axial: np.ndarray, lateral: np.ndarray, x_poly=None, z_poly=None) -> np.ndarray:
    """Windows whose centre lies inside the ROI outline (all windows if no outline)."""
    if x_poly is None:
        return np.ones(grid.shape, dtype=bool)
    x, z = grid.centres_mm(axial, lateral)
    return inpolygon(x, z, np.asarray(x_poly, float), np.asarray(z_poly, float))


def roi_bandwidth(sample_spectra: np.ndarray, reference_spectra: np.ndarray, freq: np.ndarray,
                  mask: Optional[np.ndarray] = None, min_signal_power_db: float = MIN_SIGNAL_POWER_DB,
                  **options) -> ROIBandwidth:
    """Sample and mean-reference bands in every ROI window and the shared band over the ROI.
    options: find_bw_edges options, normally bw_options(avoid_freq, W)."""
    if sample_spectra.shape != reference_spectra.shape:
        raise ValueError(f"Sample {sample_spectra.shape} and reference {reference_spectra.shape} spectra differ")
    if mask is None:
        mask = np.ones(sample_spectra.shape[:2], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    r_lo, r_hi, r_sp = window_bands(reference_spectra, freq, mask, **options)
    s_lo, s_hi, s_sp = window_bands(sample_spectra, freq, mask, **options)

    with np.errstate(invalid="ignore"):
        sh_lo, sh_hi = np.fmax(s_lo, r_lo), np.fmin(s_hi, r_hi)
        no_band = np.isnan(s_lo) | np.isnan(r_lo) | np.isnan(s_hi) | np.isnan(r_hi) | (sh_hi <= sh_lo)
        sh_lo[no_band] = np.nan
        sh_hi[no_band] = np.nan
        usable = mask & ~no_band & (s_sp >= min_signal_power_db) & (r_sp >= min_signal_power_db)

    if usable.any():
        sl, sh = sh_lo[usable], sh_hi[usable]
        strict = _round_in((sl.max(), sh.min()))
        pctl = _round_in((matlab_prctile(sl, 95), matlab_prctile(sh, 5)))
    else:
        strict = pctl = (np.nan, np.nan)
    return ROIBandwidth(
        mask=mask, sample_low=s_lo, sample_high=s_hi, sample_signal_power_db=s_sp,
        ref_low=r_lo, ref_high=r_hi, ref_signal_power_db=r_sp, shared_low=sh_lo, shared_high=sh_hi,
        shared_usable=usable, shared_strict=strict, shared_pctl=pctl,
        sample_band=_round_in((matlab_prctile(s_lo, 95), matlab_prctile(s_hi, 5))),
        reference_band=_round_in((matlab_prctile(r_lo, 95), matlab_prctile(r_hi, 5))),
        n_windows=int(mask.sum()), n_usable=int(usable.sum()), options=dict(options),
        min_signal_power_db=min_signal_power_db)
