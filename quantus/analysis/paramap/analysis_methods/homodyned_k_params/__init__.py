import weakref
from pathlib import Path
from typing import List

import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

from ...decorators import *
from .....data_objs.analysis_config import RfAnalysisConfig
from .....data_objs.analysis import Window
from .....data_objs.image import UltrasoundRfImage

# Bundled alongside this plugin (not the user's config) since it's a universal constant,
# not a per-scan calibration file -- see quantus/analysis/README.md's folder-plugin section.
_RSK_DATA_PATH = Path(__file__).parent / "RSK_data_501.mat"
_rsk_cache = {}

# Hilbert is an FFT-based, whole-signal transform, so it must run on the full RF array
# before windowing -- computing it per-window instead measurably biases mu near each
# window's edges. Cache it per image_data (a WeakKeyDictionary evicts each scan's entry
# once nothing else references it) so every window reuses the same envelope rather than
# recomputing it.
_env_cache = weakref.WeakKeyDictionary()


def _load_rsk_data() -> dict:
    """Lazily load and cache RSK_curves/k_values/mu_values/nu_values -- load once, reuse
    across every window."""
    key = str(_RSK_DATA_PATH)
    if key not in _rsk_cache:
        raw = loadmat(str(_RSK_DATA_PATH))
        _rsk_cache[key] = {
            "RSK_curves": raw["RSK_curves"],      # (501, 501, 2, 3): (k, mu, nu, classifier[R,S,K])
            "k_values": raw["k_values"].ravel(),   # (501,)
            "mu_values": raw["mu_values"].ravel(), # (501,)
            "nu_values": raw["nu_values"].ravel(), # (2,) -- moment orders, e.g. [0.72, 0.88]
        }
    return _rsk_cache[key]


def _get_full_envelope(image_data: UltrasoundRfImage) -> np.ndarray:
    """abs(hilbert(image_data.rf_data, axis=0)), computed once per scan and cached."""
    if image_data not in _env_cache:
        _env_cache[image_data] = np.abs(hilbert(image_data.rf_data, axis=0))
    return _env_cache[image_data]


def _level_curve_points(grid: np.ndarray, target: float) -> np.ndarray:
    """0-indexed (l, m) grid points approximating the level curve grid == target, found by
    searching both rows and columns for sign changes and keeping whichever of the two
    bracketing grid points is numerically closer to target (ties go to the higher index).
    """
    sign = np.sign(grid - target)
    pts = []

    # vertical search: for each column, find sign changes down the rows (varying l)
    crossings = np.diff(sign, axis=0) != 0
    ls, ms = np.nonzero(crossings)
    if ls.size:
        d0 = np.abs(grid[ls, ms] - target)
        d1 = np.abs(grid[ls + 1, ms] - target)
        chosen_l = np.where(d0 < d1, ls, ls + 1)
        pts.append(np.stack([chosen_l, ms], axis=1))

    # horizontal search: for each row, find sign changes across the columns (varying m)
    crossings = np.diff(sign, axis=1) != 0
    ls2, ms2 = np.nonzero(crossings)
    if ls2.size:
        d0 = np.abs(grid[ls2, ms2] - target)
        d1 = np.abs(grid[ls2, ms2 + 1] - target)
        chosen_m = np.where(d0 < d1, ms2, ms2 + 1)
        pts.append(np.stack([ls2, chosen_m], axis=1))

    if not pts:
        return np.empty((0, 2), dtype=int)
    return np.unique(np.concatenate(pts, axis=0), axis=0)


def _level_curve_dist(candidates_0idx: np.ndarray, curve_pts: List[np.ndarray]) -> np.ndarray:
    """Total squared distance from each candidate (P, 2) 0-indexed grid point to all level
    curves (sum, over curves, of the min squared distance to that curve)."""
    total = np.zeros(candidates_0idx.shape[0])
    for curve in curve_pts:
        diffs = candidates_0idx[:, None, :] - curve[None, :, :]
        total += np.sum(diffs ** 2, axis=2).min(axis=1)
    return total


@supported_spatial_dims(2)
@output_vars("hk_k", "hk_mu", "hk_err")
def homodyned_k_params(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
                        window: Window, config: RfAnalysisConfig,
                        image_data: UltrasoundRfImage, **kwargs) -> None:
    """Estimate homodyned K-distribution parameters (k = structure/periodicity parameter,
    mu = effective scatterer number density, err = fit residual) from the window's envelope
    samples, via the fractional-moment SNR/skewness/kurtosis level-curve grid search.

    Source: D. P. Hruska, "Envelope Statistics ... Homodyned K Distribution" (BRL, UIUC,
    2008-2009); moment orders nu=0.72, 0.88. Uses the bundled precomputed RSK_data_501.mat
    lookup table (501x501 grid over k in [0,5], mu in [0.001,100]).

    Only uses scan_rf_window -- envelope statistics are computed on the scan alone, not
    the phantom.

    On failure (a classifier falling outside the precomputed table's range), writes NaN
    for all three outputs.
    """
    rsk = _load_rsk_data()
    RSK_curves = rsk["RSK_curves"]
    k_values = rsk["k_values"]
    mu_values = rsk["mu_values"]
    nu_values = rsk["nu_values"]
    grid_size = len(k_values)

    env_full = _get_full_envelope(image_data)
    env = env_full[window.ax_min: window.ax_max + 1, window.lat_min: window.lat_max + 1].ravel()

    curve_pts = []
    for n, nu in enumerate(nu_values):
        roi_nu = env ** nu
        std = np.std(roi_nu, ddof=1)
        classifiers = (
            np.mean(roi_nu) / std,                       # R: SNR
            skew(roi_nu, bias=False),                     # S: bias-corrected skewness
            kurtosis(roi_nu, fisher=False, bias=False),    # K: bias-corrected (non-excess) kurtosis
        )
        for p, value in enumerate(classifiers):
            grid = RSK_curves[:, :, n, p]
            if not (np.isfinite(value) and grid.min() <= value <= grid.max()):
                window.results.hk_k = np.nan
                window.results.hk_mu = np.nan
                window.results.hk_err = np.nan
                return
            pts = _level_curve_points(grid, value)
            if pts.size == 0:
                window.results.hk_k = np.nan
                window.results.hk_mu = np.nan
                window.results.hk_err = np.nan
                return
            curve_pts.append(pts)

    # Coarse-to-fine grid search for the (l, m) grid point minimizing total squared distance
    # to all 6 level curves. Kept 1-indexed internally so the edge-clamping arithmetic below
    # reads naturally; only converted to 0-indexed array lookups at the end.
    num_grid_pts = 11
    l_min, l_max = 1, grid_size
    m_min, m_max = 1, grid_size
    l = m = err = None

    while True:
        l_vals = np.round(np.linspace(l_min, l_max, num_grid_pts)).astype(int)
        m_vals = np.round(np.linspace(m_min, m_max, num_grid_pts)).astype(int)
        l_grid, m_grid = np.meshgrid(l_vals, m_vals)
        candidates = np.stack([l_grid.ravel(), m_grid.ravel()], axis=1)

        dist2 = _level_curve_dist(candidates - 1, curve_pts)
        idx = np.argmin(dist2)
        err = dist2[idx]
        l, m = candidates[idx]

        if np.ptp(l_vals) <= num_grid_pts and np.ptp(m_vals) <= num_grid_pts:
            break

        new_box_size = round(np.ptp(l_vals) / 4)

        if l <= new_box_size:
            l_min, l_max = 1, 2 * new_box_size + 1
        elif l + new_box_size >= grid_size:
            l_min, l_max = grid_size - 2 * new_box_size, grid_size
        else:
            l_min, l_max = l - new_box_size, l + new_box_size

        if m <= new_box_size:
            m_min, m_max = 1, 2 * new_box_size + 1
        elif m + new_box_size >= grid_size:
            m_min, m_max = grid_size - 2 * new_box_size, grid_size
        else:
            m_min, m_max = m - new_box_size, m + new_box_size

    window.results.hk_k = k_values[l - 1]
    window.results.hk_mu = mu_values[m - 1]
    window.results.hk_err = float(err)
