"""Nakagami confidence map: a per-pixel Kolmogorov-Smirnov goodness-of-fit of the envelope to a
Nakagami distribution, used to find vessels (and other non-parenchymal structure), which fit the
Nakagami model poorly. Port of nakagami_map.m / roi_indices.m (VEVO_F2_Clarius_QUS_20260917_forPancreas,
matlab_src), confidence and Nakagami m only (the entropy map is not ported).

Per sliding window: envelope = |hilbert(RF)| (along depth), samples > 0, Nakagami maximum-likelihood
fit (m, omega), and the KS statistic
D = max |ECDF - Nakagami CDF| (ecdf steps on both sides, as nakagami_map.m). Window values are
averaged into a pixel map where windows overlap (map_creation). Pixels with D > threshold (0.10 in
MainScript_Clarius_Analysis.m) are treated as vessel.

Outliers: nakagami_map.m calls isoutlier(x, 'median', 2), where 2 is the DIMENSION (not a threshold
factor); on a column vector every value is then its own group and nothing is removed. To match the
MATLAB (and the 0.10 threshold tuned with it) no outliers are removed by default
(window_features(outlier_k=None)); remove_outliers(x, k) is the intended 'k scaled MADs' version.

Vessels vs bright structure (addition, not in nakagami_map.m): a poor Nakagami fit also marks bright
reflectors and bright/dark mixtures, not only vessel lumens. vessel_mask(darker_only=True) keeps only
pixels whose windows are also darker than the median window at the same depth (mean intensity,
omega, in dB relative to the windows on the same row: depth-dependent gain / attenuation cancel);
bright_mask() is the other side: poor fit and brighter (strong / organised reflectors).

Window: square, window_wavelengths (15) wavelengths at the probe's centre (transmit) frequency.
Difference from the MATLAB: the lateral size in A-lines uses the real mean A-line spacing of the
image (positions_from_image), not the loader's lateral resolution, which is a placeholder for
Clarius curvilinear probes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.signal import hilbert
from scipy.special import gammainc, psi

MAD_SCALE = 1.482602218505602     # -1 / (sqrt(2) * erfcinv(3/2)), MATLAB's scaled MAD
VESSEL_THRESHOLD = 0.10            # KS D above this: poor Nakagami fit (vessel), MainScript default
WINDOW_WAVELENGTHS = 15.0
OVERLAP = 0.7
SPEED_OF_SOUND = 1540.0            # m/s


# ------------------------------------------------------------------------------------- per window
def remove_outliers(x: np.ndarray, k: float = 2.0) -> np.ndarray:
    """x > 0 without isoutlier(x, 'median', k) outliers (more than k scaled MADs from the median)."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[x > 0]
    if x.size == 0:
        return x
    med = np.median(x)
    mad = MAD_SCALE * np.median(np.abs(x - med))
    return x[np.abs(x - med) <= k * mad]


def nakagami_fit(x: np.ndarray) -> tuple:
    """Maximum-likelihood Nakagami (m, omega) of samples x > 0 (as MATLAB mle(x, 'distribution',
    'Nakagami'), i.e. a gamma fit of x^2): omega = mean(x^2), m solves log m - psi(m) = s with
    s = log(mean(x^2)) - mean(log(x^2))."""
    x2 = np.asarray(x, dtype=float) ** 2
    omega = float(np.mean(x2))
    s = np.log(omega) - float(np.mean(np.log(x2)))
    if not np.isfinite(s) or s <= 0:
        return np.nan, omega
    f = lambda m: np.log(m) - psi(m) - s
    lo, hi = 1e-6, 1.0
    while f(hi) > 0:          # log m - psi(m) decreases from +inf to 0: bracket the root
        hi *= 2
        if hi > 1e8:
            return np.nan, omega
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12)), omega


def nakagami_cdf(x: np.ndarray, m: float, omega: float) -> np.ndarray:
    return gammainc(m, m * np.asarray(x, dtype=float) ** 2 / omega)


def ks_statistic(x: np.ndarray, m: float, omega: float) -> float:
    """max |ECDF - CDF| over the unique sample values, the ECDF taken just before and at each step
    (nakagami_map.m: ecdf, delta1 / delta2)."""
    xs, counts = np.unique(np.asarray(x, dtype=float), return_counts=True)
    f_at = np.cumsum(counts) / counts.sum()
    f_before = np.r_[0.0, f_at[:-1]]
    null = nakagami_cdf(xs, m, omega)
    return float(max(np.max(np.abs(f_before - null)), np.max(np.abs(f_at - null))))


def window_features(env_window: np.ndarray, outlier_k: Optional[float] = None) -> tuple:
    """(m, D) of one window of envelope samples; NaN if too few samples remain. outlier_k None:
    keep every sample > 0 (as nakagami_map.m in practice, see the module notes)."""
    x = np.asarray(env_window, dtype=float).ravel()
    x = x[x > 0] if outlier_k is None else remove_outliers(x, outlier_k)
    if x.size < 10:
        return np.nan, np.nan
    m, omega = nakagami_fit(x)
    if not np.isfinite(m):
        return np.nan, np.nan
    return m, ks_statistic(x, m, omega)


# ------------------------------------------------------------------------------------- the grid
def window_indices(mask: np.ndarray, axial_size: int, lateral_size: int, axial_overlap: float = OVERLAP,
                   lateral_overlap: float = OVERLAP, include_edge: bool = False) -> np.ndarray:
    """Sliding windows over a mask (port of roi_indices.m, 0-based): (n, 4) array of
    top, bottom, left, right (inclusive). A window is kept if more than 99.9 % of it is in the mask
    (1 % with include_edge)."""
    mask = np.asarray(mask, dtype=bool)
    ax_inc = max(int(np.round(axial_size * (1 - axial_overlap))), 1)
    lat_inc = max(int(np.round(lateral_size * (1 - lateral_overlap))), 1)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return np.zeros((0, 4), dtype=int)
    ax_start, ax_end = rows[0], min(rows[-1], mask.shape[0] - axial_size)
    lat_start, lat_end = cols[0], min(cols[-1], mask.shape[1] - lateral_size)
    threshold = 0.01 if include_edge else 0.999
    out = []
    for a in range(ax_start, ax_end + 1, ax_inc):
        for l in range(lat_start, lat_end + 1, lat_inc):
            if mask[a:a + axial_size, l:l + lateral_size].mean() > threshold:
                out.append((a, a + axial_size - 1, l, l + lateral_size - 1))
    return np.asarray(out, dtype=int).reshape(-1, 4)


@dataclass
class NakagamiConfidence:
    """confidence: (samples, lines) KS D map (NaN where no window); m_map: Nakagami m map;
    relative_intensity_db: mean window intensity relative to the median window at the same depth (dB);
    windows: (n, 4) top, bottom, left, right; features: (n, 3) m, D, relative intensity (dB) per window."""
    confidence: np.ndarray
    m_map: np.ndarray
    relative_intensity_db: np.ndarray
    windows: np.ndarray
    features: np.ndarray
    axial_size: int
    lateral_size: int
    window_mm: float

    def vessel_mask(self, threshold: float = VESSEL_THRESHOLD, darker_only: bool = True) -> np.ndarray:
        """Pixels with a poor Nakagami fit (KS D > threshold); darker_only: and darker than the
        median at their depth (vessel lumens, not bright reflectors)."""
        with np.errstate(invalid="ignore"):
            mask = np.nan_to_num(self.confidence, nan=0.0) > threshold
            if darker_only:
                mask &= np.nan_to_num(self.relative_intensity_db, nan=0.0) < 0
        return mask

    def bright_mask(self, threshold: float = VESSEL_THRESHOLD) -> np.ndarray:
        """Pixels with a poor Nakagami fit (KS D > threshold) that are brighter than the median at
        their depth: strong / organised reflectors (vessel walls, septa, ligaments), which break the
        diffuse-scattering assumption the other way from a lumen."""
        with np.errstate(invalid="ignore"):
            return (np.nan_to_num(self.confidence, nan=0.0) > threshold) & \
                   (np.nan_to_num(self.relative_intensity_db, nan=0.0) > 0)


def nakagami_confidence(rf: np.ndarray, axial_size: int, lateral_size: int, mask: Optional[np.ndarray] = None,
                        axial_overlap: float = OVERLAP, lateral_overlap: float = OVERLAP,
                        window_mm: float = np.nan) -> NakagamiConfidence:
    """Nakagami confidence (KS D) and m maps of one RF frame (samples, lines)."""
    rf = np.asarray(rf, dtype=float)
    env = np.abs(hilbert(rf, axis=0))
    if mask is None:
        mask = np.ones(rf.shape, dtype=bool)
    win = window_indices(mask, axial_size, lateral_size, axial_overlap, lateral_overlap)
    feats = np.array([window_features(env[t:b + 1, l:r + 1]) for t, b, l, r in win]).reshape(-1, 2)
    # mean intensity (omega, dB) of each window relative to the median window on the same row
    with np.errstate(divide="ignore"):
        omega_db = np.array([10 * np.log10(np.mean(env[t:b + 1, l:r + 1] ** 2)) for t, b, l, r in win])
    rel = np.full(len(win), np.nan)
    for top in np.unique(win[:, 0]) if len(win) else []:
        k = win[:, 0] == top
        rel[k] = omega_db[k] - np.median(omega_db[k])
    feats = np.c_[feats, rel] if len(win) else np.zeros((0, 3))
    maps = []
    for k in range(3):
        acc = np.zeros(rf.shape)
        cnt = np.zeros(rf.shape)
        for (t, b, l, r), v in zip(win, feats[:, k]):
            if np.isfinite(v):
                acc[t:b + 1, l:r + 1] += v
                cnt[t:b + 1, l:r + 1] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            maps.append(acc / cnt)
    return NakagamiConfidence(confidence=maps[1], m_map=maps[0], relative_intensity_db=maps[2], windows=win,
                              features=feats,
                              axial_size=int(axial_size), lateral_size=int(lateral_size), window_mm=float(window_mm))


def nakagami_confidence_from_image(image_data, frame: int = 0, window_wavelengths: float = WINDOW_WAVELENGTHS,
                                   center_freq_mhz: Optional[float] = None, overlap: float = OVERLAP,
                                   mask: Optional[np.ndarray] = None) -> NakagamiConfidence:
    """nakagami_confidence on a loaded QuantUS image's sample frame: square window of
    window_wavelengths wavelengths at the probe's centre frequency (rf.yml transmit frequency);
    axial size from the sample spacing, lateral size from the mean A-line spacing of the image."""
    from ..correlation_length.correlation import positions_from_image
    from ..correlation_length.first_pass import probe_center_frequency_mhz

    rf = image_data.rf_data
    rf = rf[frame] if rf.ndim == 3 else rf
    fc = center_freq_mhz or probe_center_frequency_mhz(image_data)
    if not fc:
        raise ValueError("The probe's centre frequency is unknown (no transmit frequency in rf.yml); "
                         "pass center_freq_mhz")
    axial, lateral = positions_from_image(image_data, *rf.shape)
    mid = rf.shape[1] // 2
    dz = float(np.median(np.diff(axial[:, mid])))
    dy = float(np.mean(np.hypot(np.diff(lateral, axis=1), np.diff(axial, axis=1))))   # mean A-line spacing
    window_mm = window_wavelengths * SPEED_OF_SOUND / (fc * 1e6) * 1e3
    ax_px = int(np.ceil(window_mm / dz))
    lat_px = max(int(np.ceil(window_mm / dy)), 2)
    return nakagami_confidence(rf, ax_px, lat_px, mask=mask, axial_overlap=overlap, lateral_overlap=overlap,
                               window_mm=window_mm)
