"""Multitaper power spectra on a grid of windows (port of the S1Study MATLAB functions, versions of
2026-11-12/13):

  CreateDPSS                          -> create_dpss
  PowerSpectrumMTF                    -> power_spectrum_mtf     (Thomson adaptive weights)
  DefinePowerSpectrumLocations        -> define_power_spectrum_locations
  ComputePowerSpectra_MTM_Grid(_Reference) -> grid_power_spectra

Kept index-for-index faithful to the MATLAB (1-based rules translated to 0-based indices) so the
two can be checked against each other on the same RF; see README.md. The zoom-FFT comparison
periodogram ("Prev") the MATLAB also computes is not ported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.signal import detrend
from scipy.signal.windows import dpss as _scipy_dpss


def _mround(x):
    """MATLAB round (half away from zero), scalar or array."""
    x = np.asarray(x, dtype=float)
    r = np.sign(x) * np.floor(np.abs(x) + 0.5)
    return int(r) if r.ndim == 0 else r.astype(int)


# ---------------------------------------------------------------------------------------------
# CreateDPSS
# ---------------------------------------------------------------------------------------------
def n_tapers(nw: float) -> int:
    """Tapers used: max(1, floor(2 NW - 1)) -- well concentrated inside +/-W (CreateDPSS)."""
    return max(1, int(np.floor(2 * nw - 1)))


def dpss_tapers(n: int, nw: float):
    """(E, V): n x K Slepian sequences (unit energy) and their concentration eigenvalues, as
    MATLAB [E, V] = dpss(n, NW, K) with K = n_tapers(NW)."""
    k = n_tapers(nw)
    e, v = _scipy_dpss(n, nw, Kmax=k, sym=True, norm=2, return_ratios=True)
    return np.atleast_2d(e).T, np.atleast_1d(v)


def create_dpss(max_n: int, nw: float) -> dict:
    """{N: (E, V)} for N = max_n down to round(max_n / 2), as CreateDPSS (which stores a cell
    array indexed from max_n downwards; PowerSpectrumMTF picks the entry for the data length)."""
    return {n: dpss_tapers(n, nw) for n in range(max_n, _mround(max_n / 2) - 1, -1)}


# ---------------------------------------------------------------------------------------------
# PowerSpectrumMTF
# ---------------------------------------------------------------------------------------------
def power_spectrum_mtf(data: np.ndarray, tapers, fs_mhz: float, nfft: int = 1024,
                       max_iter: int = 1000) -> np.ndarray:
    """Adaptive multitaper power spectrum of every column (A-line) of data, as PowerSpectrumMTF
    ('Normal', one-sided).

    data: (n_samples, n_lines). tapers: (E, V) for n_samples (dpss_tapers). Each line is
    detrended and its mean removed; S_k = |FFT(E_k x, nfft)|^2; Thomson's adaptive weights are
    iterated from S = (S_1 + S_2)/2 until sum|S - S_prev| / nfft <= 0.0005 * (x'x/N) / nfft;
    then one-sided (bins 1 .. nfft/2 - 1 doubled) and scaled as PSD x fs, i.e. no fs factor.
    Returns (nfft // 2 + 1, n_lines).
    """
    e, v = tapers
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n, n_lines = x.shape
    if e.shape[0] != n:
        raise ValueError(f"Tapers are for {e.shape[0]} samples, data has {n}")
    k = e.shape[1]
    x = detrend(x, axis=0, type="linear")
    x = x - x.mean(axis=0, keepdims=True)
    sk = np.abs(np.fft.fft(e[:, :, None] * x[:, None, :], nfft, axis=0)) ** 2   # (nfft, K, lines)
    sig2 = np.sum(x * x, axis=0) / n                                         # (lines,)
    s = sk[:, 0, :].copy() if k < 2 else (sk[:, 0, :] + sk[:, 1, :]) / 2     # (nfft, lines)
    s1 = np.zeros_like(s)
    tol = 0.0005 * sig2 / nfft
    a = sig2[None, :] * (1 - v)[:, None]                                     # (K, lines)
    # Each line iterates until its own criterion is met (as the MATLAB per-line loop does);
    # converged lines are frozen.
    active = np.sum(np.abs(s - s1), axis=0) / nfft > tol
    it = 0
    while active.any() and it < max_iter:
        sa, ska, aa = s[:, active], sk[:, :, active], a[:, active]
        b = sa[:, None, :] / (sa[:, None, :] * v[None, :, None] + aa[None, :, :])   # (nfft, K, act)
        wk = b ** 2 * v[None, :, None]
        s_new = np.sum(wk * ska, axis=1) / np.sum(wk, axis=1)
        s1[:, active] = sa
        s[:, active] = s_new
        active[active] = np.sum(np.abs(s_new - sa), axis=0) / nfft > tol[active]
        it += 1
    one = s[:nfft // 2 + 1].copy()
    one[1:-1] *= 2              # computepsd 'onesided' (even nfft): DC and Nyquist not doubled
    return one                  # PSD * SampFreq: computepsd's 1/Fs undone


def mtm_freq_axis(fs_mhz: float, nfft: int = 1024) -> np.ndarray:
    """psdfreqvec('npts', nfft, 'Fs', fs)(1 : nfft/2 + 1): 0 .. fs/2 (MHz)."""
    return np.arange(nfft // 2 + 1) * fs_mhz / nfft


# ---------------------------------------------------------------------------------------------
# DefinePowerSpectrumLocations
# ---------------------------------------------------------------------------------------------
@dataclass
class SpectrumGrid:
    """Window grid (DefinePowerSpectrumLocations outputs). Indices are 0-based and inclusive:
    window (i, j) covers samples start_ax[i] .. end_ax[i] and lines start_lat[i, j] .. end_lat[i, j]."""
    position_row: np.ndarray    # (n_ax, n_lat) window centre sample
    start_ax: np.ndarray        # (n_ax, n_lat)
    end_ax: np.ndarray
    position_col: np.ndarray    # (n_ax, n_lat) window centre line
    start_lat: np.ndarray
    end_lat: np.ndarray
    ax_lengths: np.ndarray      # (n_ax,) samples per window (curPERNumAxSamples + 1)
    lat_widths: np.ndarray      # (n_ax,) lines per window
    delta_lats: np.ndarray      # (n_ax,) mean line spacing (mm) at each row
    delta_ax: float
    sample_spacing: int
    line_spacing: int
    ax_overlap: np.ndarray      # (n_ax,) achieved axial overlap fraction
    lat_overlap: np.ndarray
    nw: float
    dpss_n: int                 # taper length (window samples)

    @property
    def shape(self):
        return self.position_row.shape

    def depths_mm(self, axial: np.ndarray) -> np.ndarray:
        """(n_ax,) depth of each row's window centre on the centre line."""
        mid0 = _mround(axial.shape[1] / 2) - 1
        return axial[self.position_row[:, 0], mid0]

    def centres_mm(self, axial: np.ndarray, lateral: np.ndarray):
        """(x, z) Cartesian mm of every window centre, (n_ax, n_lat) each."""
        return lateral[self.position_row, self.position_col], axial[self.position_row, self.position_col]


def define_power_spectrum_locations(axial_position_mm: np.ndarray, lateral_position_mm: np.ndarray,
                                    start_ax_mm: float, end_ax_mm: float, start_lat_line: int, end_lat_line: int,
                                    ax_window_mm: float, lat_window_mm: float, nw: float,
                                    min_overlap_ax: Optional[float] = 0.9, min_overlap_lat: Optional[float] = 0.85,
                                    num_pos_row: Optional[int] = None, num_pos_col: Optional[int] = None,
                                    verbose: bool = False) -> SpectrumGrid:
    """Grid of power-spectrum windows (port of DefinePowerSpectrumLocations.m, S1Study 2026-11-13).

    start_ax_mm / end_ax_mm: depth range of the window CENTRES (nearest sample on the centre
    line). start_lat_line / end_lat_line: range of window-centre lines (1-based, inclusive, as
    the MATLAB StartLatLine / EndLatLine). Windows are round(ax_window_mm / dz) + 1 samples and
    round(lat_window_mm / line spacing at that row) + 1 lines. Spacing: so that neighbouring
    windows overlap by at least min_overlap_ax / min_overlap_lat (the lateral one at the row with
    the fewest lines), else num_pos_row / num_pos_col positions. Windows that would run outside
    the image are dropped, never shifted (a lateral position is kept only if it fits in every row).
    """
    axial = np.asarray(axial_position_mm, dtype=float)
    lateral = np.asarray(lateral_position_mm, dtype=float)
    n_rows, n_cols = axial.shape
    if min_overlap_ax is None and num_pos_row is None:
        raise ValueError("Set min_overlap_ax (or num_pos_row)")
    if min_overlap_lat is None and num_pos_col is None:
        raise ValueError("Set min_overlap_lat (or num_pos_col)")

    mid_lat_1 = _mround(n_cols / 2)
    start_lat, end_lat = int(start_lat_line), int(end_lat_line)
    num_lines = len(range(start_lat, end_lat + 1))

    centre = axial[:, mid_lat_1 - 1]
    delta_ax = centre[2] - centre[1]
    start_ax_1 = int(np.argmin(np.abs(centre - start_ax_mm))) + 1
    end_ax_1 = int(np.argmin(np.abs(centre - end_ax_mm))) + 1
    num_samples = len(range(start_ax_1, end_ax_1 + 1))

    win_ax = _mround(ax_window_mm / delta_ax)          # curPERNumAxSamples
    half_ax = win_ax // 2
    if min_overlap_ax is not None:
        sample_spacing = max(1, int(np.floor((1 - min_overlap_ax) * (win_ax + 1))))
    else:
        sample_spacing = max(1, _mround(num_samples / num_pos_row))
    ax_locs = np.arange(start_ax_1, end_ax_1 + 1, sample_spacing)

    s_all = ax_locs - half_ax
    e_all = s_all + win_ax
    keep_ax = (s_all >= 1) & (e_all <= n_rows)
    if verbose and np.any(~keep_ax):
        print(f"  dropped {np.sum(~keep_ax)} of {keep_ax.size} axial positions (window outside the image)")
    ax_locs = ax_locs[keep_ax]
    n_ax = ax_locs.size

    span = np.arange(max(1, start_lat), min(n_cols, max(end_lat, start_lat + 1)) + 1)   # 1-based
    row_dlat = np.zeros(n_ax)
    row_lines = np.zeros(n_ax, dtype=int)
    for i, loc in enumerate(ax_locs):
        s_ax = loc - half_ax
        e_ax = s_ax + win_ax
        mid_ax = _mround((s_ax + e_ax) / 2)
        row_dlat[i] = np.mean(np.diff(lateral[mid_ax - 1, span - 1]))
        row_lines[i] = _mround(lat_window_mm / row_dlat[i])

    if min_overlap_lat is not None:
        min_lat_window = int(np.min(row_lines)) + 1 if n_ax else 1
        line_spacing = max(1, int(np.floor((1 - min_overlap_lat) * min_lat_window)))
    else:
        line_spacing = max(1, _mround(num_lines / num_pos_col))
    lat_locs = np.arange(start_lat, end_lat + 1, line_spacing)

    keep_lat = np.ones(lat_locs.size, dtype=bool)
    for i in range(n_ax):
        s_lat = lat_locs - row_lines[i] // 2
        e_lat = s_lat + row_lines[i]
        keep_lat &= (s_lat >= 1) & (e_lat <= n_cols)
    if verbose and np.any(~keep_lat):
        print(f"  dropped {np.sum(~keep_lat)} of {keep_lat.size} lateral positions (window outside the image)")
    lat_locs = lat_locs[keep_lat]
    n_lat = lat_locs.size
    if n_ax == 0 or n_lat == 0:
        raise ValueError("No power-spectrum windows fit inside the image for this ROI.")

    shape = (n_ax, n_lat)
    pos_row, s_ax_m, e_ax_m = (np.zeros(shape, dtype=int) for _ in range(3))
    pos_col, s_lat_m, e_lat_m = (np.zeros(shape, dtype=int) for _ in range(3))
    for i, loc in enumerate(ax_locs):
        s_ax = loc - half_ax
        e_ax = s_ax + win_ax
        mid_ax = _mround((s_ax + e_ax) / 2)
        half_lat = row_lines[i] // 2
        for j, lloc in enumerate(lat_locs):
            s_lat = lloc - half_lat
            e_lat = s_lat + row_lines[i]
            pos_col[i, j] = _mround((s_lat + e_lat) / 2)
            s_lat_m[i, j], e_lat_m[i, j] = s_lat, e_lat
            pos_row[i, j], s_ax_m[i, j], e_ax_m[i, j] = mid_ax, s_ax, e_ax

    ax_lengths = np.full(n_ax, win_ax + 1)
    lat_widths = row_lines + 1
    return SpectrumGrid(
        position_row=pos_row - 1, start_ax=s_ax_m - 1, end_ax=e_ax_m - 1,
        position_col=pos_col - 1, start_lat=s_lat_m - 1, end_lat=e_lat_m - 1,
        ax_lengths=ax_lengths, lat_widths=lat_widths, delta_lats=row_dlat, delta_ax=float(delta_ax),
        sample_spacing=sample_spacing, line_spacing=line_spacing,
        ax_overlap=(ax_lengths - sample_spacing) / ax_lengths, lat_overlap=(lat_widths - line_spacing) / lat_widths,
        nw=float(nw), dpss_n=int(_mround(np.mean(ax_lengths))),
    )


# ---------------------------------------------------------------------------------------------
# ComputePowerSpectra_MTM_Grid(_Reference)
# ---------------------------------------------------------------------------------------------
def grid_power_spectra(rf_frames: np.ndarray, grid: SpectrumGrid, fs_mhz: float, nfft: int = 1024,
                       tapers=None, progress=None) -> np.ndarray:
    """Multitaper spectrum of every grid window, averaged over its lines, then over the frames
    (ComputePowerSpectra_MTM_Grid_Reference's MeanRefPS; with one frame, the sample's
    SampPowerSpectra_MTF). rf_frames: (n_frames, n_samples, n_lines) or (n_samples, n_lines).
    Returns (n_ax, n_lat, nfft // 2 + 1), linear."""
    rf = np.asarray(rf_frames, dtype=float)
    if rf.ndim == 2:
        rf = rf[None]
    n_ax, n_lat = grid.shape
    if tapers is None:
        tapers = dpss_tapers(int(grid.ax_lengths[0]), grid.nw)
    out = np.zeros((n_ax, n_lat, nfft // 2 + 1))
    n_steps = rf.shape[0] * n_ax
    for f_i, frame in enumerate(rf):
        for i in range(n_ax):
            if progress is not None:
                progress(f_i * n_ax + i, n_steps)    # progress(done, total): e.g. a GUI progress bar
            # every window of a row covers the same samples, and a line's spectrum depends only
            # on that line: compute each line once per row, then average over each window's lines
            c0, c1 = int(grid.start_lat[i].min()), int(grid.end_lat[i].max())
            rows = frame[grid.start_ax[i, 0]:grid.end_ax[i, 0] + 1, c0:c1 + 1]
            line_ps = power_spectrum_mtf(rows, tapers, fs_mhz, nfft)                  # (nfreq, lines)
            for j in range(n_lat):
                out[i, j] += line_ps[:, grid.start_lat[i, j] - c0:grid.end_lat[i, j] - c0 + 1].mean(axis=1)
    return out / rf.shape[0]
