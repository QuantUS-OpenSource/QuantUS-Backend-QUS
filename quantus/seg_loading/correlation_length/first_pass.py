"""First-pass power check: where in the image is there enough signal for spectral analysis?

Port of the S1Study / ClariusExVivoVersionTest MATLAB chain (Code_Liver/LiverFirstPassROI.m):

  CalcFirstPassPowerSpectra  -> calc_first_pass        Hanning-windowed spectra at every depth,
                                                        averaged over all lines (depth profile) and
                                                        over lateral blocks (2-D map); band edges and
                                                        spectral SNR for each.
  FindBWEdges                -> find_bw_edges           band edges = noise floor + 10 dB, walking out
                                                        from the peak, stopping at notches.
  IntersectFirstPass         -> intersect_first_pass    blocks usable in BOTH sample and reference.
  ROIBandCheck               -> roi_band_check          band of a given ROI from block spectra
                                                        averaged across the ROI at each depth.

(AutoROIFromFirstPass, the automatic ROI choice, is deliberately not included.)

Kept index-for-index faithful to the MATLAB (1-based rules translated to 0-based indices) so the
two can be checked against each other on the same RF; tests/test_first_pass.py and
README_first_pass.md list what was verified and the known differences.

Usable (MATLAB rule of thumb): a band exists and signal power = peak - (noise floor + 10 dB)
>= 5 dB, i.e. peak - noise floor >= 15 dB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.signal import find_peaks

C_MM_US = 1.540            # speed of sound used by the MATLAB to get fs from the sample spacing
ABOVE_NOISE_DB = 10.0      # band edges sit this far above the noise floor
MIN_SIGNAL_POWER_DB = 5.0  # peak must clear the band level by this much -> usable


class NyquistError(ValueError):
    """The signal band runs up to Nyquist: there is no out-of-band noise above it, so the noise
    floor, SNR and band edges of the first pass are not reliable. Do not use this data."""


class NyquistWarning(UserWarning):
    """As NyquistError, when calc_first_pass(..., on_nyquist="warn")."""


def _mround(x):
    """MATLAB round (half away from zero), scalar or array."""
    x = np.asarray(x, dtype=float)
    r = np.sign(x) * np.floor(np.abs(x) + 0.5)
    return int(r) if r.ndim == 0 else r.astype(int)


def matlab_hanning(n: int) -> np.ndarray:
    """MATLAB hanning(n): symmetric, WITHOUT the zero end points (numpy.hanning includes them)."""
    k = np.arange(1, n + 1)
    return 0.5 * (1 - np.cos(2 * np.pi * k / (n + 1)))


# ---------------------------------------------------------------------------------------------
# FindBWEdges
# ---------------------------------------------------------------------------------------------
def find_bw_edges(ps: np.ndarray, freq: np.ndarray, above_noise_db: float = ABOVE_NOISE_DB,
                  dip_prominence: float = 6.0, guard: float = 0.0, avoid_freq: Sequence[float] = (),
                  drop_edge_bins: bool = True, search_above: float = -np.inf,
                  noise_method: str = "plateau", noise_band: Optional[Sequence[float]] = None,
                  noise_pctl: float = 20.0, noise_tol_db: float = 3.0, min_noise_bins: int = 5):
    """Band edges of one power spectrum (port of FindBWEdges.m, S1Study 2026-11-13).

    ps: power spectrum, LINEAR units; freq: MHz. Returns (f_low, f_high, info) with info keys
    peak_freq, noise_floor_db, threshold_db, signal_power_db (peak - threshold), dips_mhz,
    low_limited_by, high_limited_by, peak_loc (0-based, after dropping the edge bins),
    noise_method_used, noise_band_mhz, num_noise_bins, noise_flat.

    Noise floor ('plateau', default): the flat out-of-band region above the signal band -- the
    bins on the high side (peak -> Nyquist) within noise_tol_db of their noise_pctl percentile,
    floor = mean POWER of those bins; falls back to 'legacy' if fewer than min_noise_bins.
    'band': mean power inside noise_band; 'legacy': min of the low or high side, chosen by a
    clipping test. Edges: walk out from the peak while >= floor + above_noise_db, stopping at a
    notch (local minimum >= dip_prominence dB deep); then kept guard MHz clear of notches and
    avoid_freq. NaN edges if no band remains.
    """
    ps = np.asarray(ps, dtype=float).ravel()
    freq = np.asarray(freq, dtype=float).ravel()
    if drop_edge_bins:
        ps, freq = ps[1:-1], freq[1:-1]
    with np.errstate(divide="ignore"):
        s = 10 * np.log10(ps)
    n = s.size

    # ---- peak search
    s_search = np.where(freq >= search_above, s, -np.inf)
    ip = int(np.argmax(s_search))  # first maximum, as MATLAB max
    peak_val = s_search[ip]

    # ---- noise floor
    method = noise_method.lower()
    noise_idx = np.array([], dtype=int)
    if method == "band":
        if noise_band is None or len(noise_band) != 2:
            raise ValueError("noise_method 'band' needs noise_band = (f1, f2) in MHz")
        noise_idx = np.nonzero((freq >= noise_band[0]) & (freq <= noise_band[1]))[0]
        if noise_idx.size == 0:
            raise ValueError("No frequency bins inside noise_band")
        method_used = "band"
    elif method == "plateau":
        high_side = np.arange(ip, n)
        v = np.sort(s[high_side])
        level = v[max(1, _mround(noise_pctl / 100 * v.size)) - 1]
        noise_idx = high_side[s[high_side] <= level + noise_tol_db]
        if noise_idx.size >= min_noise_bins:
            method_used = "plateau"
        else:
            noise_idx = np.array([], dtype=int)
            method_used = "legacy (no high-side plateau)"
    elif method == "legacy":
        method_used = "legacy"
    else:
        raise ValueError("noise_method must be 'plateau', 'band' or 'legacy'")

    if noise_idx.size:
        noise_floor = 10 * np.log10(np.mean(ps[noise_idx]))
        noise_band_used = (float(np.min(freq[noise_idx])), float(np.max(freq[noise_idx])))
        if noise_idx.size > 2 and noise_band_used[1] > noise_band_used[0]:
            slope = np.polyfit(freq[noise_idx], s[noise_idx], 1)[0]
            noise_flat = bool(abs(slope) * (noise_band_used[1] - noise_band_used[0]) < noise_tol_db)
        else:
            noise_flat = True
    else:
        low_nf = np.min(s[:ip + 1])
        high_nf = np.min(s[ip:])
        curve = (s - s.min()) / (s.max() - s.min())
        edge = min(10, n // 4)
        with np.errstate(invalid="ignore"):
            clipped_low = bool(np.mean(curve[:edge]) > 0.1) if edge > 0 else False
            clipped_high = bool(np.mean(curve[n - edge:]) > 0.1) if edge > 0 else False
        if clipped_high and not clipped_low:
            noise_floor = low_nf
        elif clipped_low and not clipped_high:
            noise_floor = high_nf
        else:
            noise_floor = max(low_nf, high_nf)
        noise_band_used = (np.nan, np.nan)
        noise_flat = False
    thr = noise_floor + above_noise_db

    # ---- notches: local minima at least dip_prominence dB deep (findpeaks on -s)
    dip_idx = np.array([], dtype=int)
    if np.isfinite(dip_prominence) and n > 3:
        dip_idx, _ = find_peaks(-s, prominence=dip_prominence)
    dips = freq[dip_idx]
    dip_set = set(int(i) for i in dip_idx)

    # ---- walk outward from the peak
    lo, low_by = ip, "threshold"
    while lo > 0 and s[lo - 1] >= thr:
        lo -= 1
        if lo in dip_set:
            low_by = "notch"
            break
    if lo == 0:
        low_by = "spectrum edge"
    hi, high_by = ip, "threshold"
    while hi < n - 1 and s[hi + 1] >= thr:
        hi += 1
        if hi in dip_set:
            high_by = "notch"
            break
    if hi == n - 1:
        high_by = "spectrum edge"
    f_low, f_high = float(freq[lo]), float(freq[hi])

    # ---- guard band from every notch / avoid_freq
    for a in np.unique(np.concatenate([dips, np.asarray(avoid_freq, dtype=float).ravel()])):
        if a < freq[ip] and f_low < a + guard:
            f_low, low_by = float(a + guard), f"notch {a:.2f} MHz + guard"
        elif a > freq[ip] and f_high > a - guard:
            f_high, high_by = float(a - guard), f"notch {a:.2f} MHz - guard"
    if f_high <= f_low:
        f_low = f_high = np.nan

    info = dict(peak_freq=float(freq[ip]), noise_floor_db=float(noise_floor), threshold_db=float(thr),
                signal_power_db=float(peak_val - thr), dips_mhz=dips, low_limited_by=low_by,
                high_limited_by=high_by, peak_loc=ip, noise_method_used=method_used,
                noise_band_mhz=noise_band_used, num_noise_bins=int(noise_idx.size), noise_flat=noise_flat)
    return f_low, f_high, info


def _run_bw(ps, freq_axis, avoid_freq, above_noise_db, search_above, with_method=False):
    """CalcFirstPassPowerSpectra RunBW: (f_low, f_high, peak_freq, noise_floor, snr = peak - floor)
    [+ the noise-floor method used, with_method=True]."""
    ps = np.asarray(ps, dtype=float).ravel()
    if np.any(np.isnan(ps)):
        return (np.nan,) * 5 + ((False,) if with_method else ())
    fl, fh, info = find_bw_edges(ps, freq_axis, above_noise_db=above_noise_db, avoid_freq=avoid_freq,
                                 search_above=search_above)
    snr = info["signal_power_db"] + (info["threshold_db"] - info["noise_floor_db"])
    out = (fl, fh, info["peak_freq"], info["noise_floor_db"], snr)
    # Nyquist-limited: no out-of-band noise plateau above the peak AND the band runs into the top
    # bin (a noise-only spectrum also lacks a plateau, but has no band reaching the edge)
    nyq = info["noise_method_used"].startswith("legacy (no high-side plateau)") and \
        info["high_limited_by"] == "spectrum edge" and not np.isnan(fh)
    return out + ((nyq,) if with_method else ())


# ---------------------------------------------------------------------------------------------
# CalcFirstPassPowerSpectra
# ---------------------------------------------------------------------------------------------
@dataclass
class FirstPass:
    """Output of calc_first_pass (the variables CalcFirstPassPowerSpectra saves).

    Depth profile (n_depths,): mean_ps (n_depths, n_freq), f_low, f_high, peak_freq,
    noise_floor_db, snr_db, signal_power_db, usable. Lateral blocks (n_depths, n_blocks):
    snr_2d, signal_power_2d, usable_2d, f_low_2d, f_high_2d, peak_freq_2d, block_x, block_z
    (Cartesian mm of each block centre), block_ps (n_depths, n_blocks, n_freq, linear).
    Indices (rows, block_centre / block_start) are 0-based.
    """
    mean_ps: np.ndarray
    freq_axis: np.ndarray
    depths: np.ndarray
    rows: np.ndarray
    f_low: np.ndarray
    f_high: np.ndarray
    peak_freq: np.ndarray
    noise_floor_db: np.ndarray
    snr_db: np.ndarray
    signal_power_db: np.ndarray
    usable: np.ndarray
    snr_2d: np.ndarray
    signal_power_2d: np.ndarray
    usable_2d: np.ndarray
    f_low_2d: np.ndarray
    f_high_2d: np.ndarray
    peak_freq_2d: np.ndarray
    block_x: np.ndarray
    block_z: np.ndarray
    block_ps: np.ndarray
    block_start: np.ndarray
    block_centre: np.ndarray
    block_lines: int
    block_width_mm: float
    window_length_mm: float
    window_samples: int
    fs_mhz: float
    delta_ax_mm: float
    num_frames: int
    num_lines: int
    avoid_freq: tuple = ()
    search_above: float = -np.inf
    above_noise_db: float = ABOVE_NOISE_DB
    min_signal_power_db: float = MIN_SIGNAL_POWER_DB
    # (n_depths,) True where the depth-profile spectrum has no out-of-band noise plateau above the
    # signal and its band runs into the top bin (signal up to Nyquist); see NyquistError
    nyquist_limited: Optional[np.ndarray] = None

    @property
    def snr_threshold_db(self) -> float:
        return self.above_noise_db + self.min_signal_power_db

    @property
    def bandwidth_2d(self) -> np.ndarray:
        return self.f_high_2d - self.f_low_2d


def calc_first_pass(rf_frames: np.ndarray, axial_position_mm: np.ndarray, lateral_position_mm: np.ndarray,
                    window_length_mm: float, block_width_mm: float,
                    ax_start_mm: Optional[float] = None, ax_end_mm: Optional[float] = None,
                    n_ax_locations: Optional[int] = None, avoid_freq: Sequence[float] = (),
                    search_above: Optional[float] = None, ax_step_mm: Optional[float] = None,
                    sampling_frequency_mhz: Optional[float] = None, on_nyquist: str = "error",
                    block_lines: Optional[int] = None) -> FirstPass:
    """First-pass power spectra: depth profile and lateral-block map (port of CalcFirstPassPowerSpectra.m).

    rf_frames: (n_frames, n_samples, n_lines) or (n_samples, n_lines); every frame's spectra are
        averaged (e.g. all captures of a reference phantom).
    axial_position_mm, lateral_position_mm: (n_samples, n_lines) Cartesian position of every RF
        sample (positions_from_image), as AxialPosition / LateralPosition in RFData.mat.
    window_length_mm: axial window (Hanning, odd number of samples, unit energy; FFT length =
        window length). block_width_mm: lateral block width, converted to a fixed number of lines
        with the line spacing at the middle of the depth range (blocks overlap by half).
    ax_start_mm / ax_end_mm: shallowest / deepest window centre (None: first / last full window).
    ax_step_mm: spacing of the window centres; otherwise n_ax_locations depths (one is needed).
    avoid_freq: known notch frequencies (MHz) for find_bw_edges. search_above: restrict the peak
        search to frequencies >= this (MHz), e.g. 7.5 for the L20 13.0.0 upper lobe.
    sampling_frequency_mhz: None (as MATLAB): fs = 1.540 / 2 / axial sample spacing, i.e. from
        AxialPosition with c = 1540 m/s. Pass the true fs to use it for the frequency axis
        instead (the two differ when the loader's axial spacing used another sound speed).
    on_nyquist: what to do if the signal band runs up to Nyquist at any depth (the depth-profile
        spectrum has no out-of-band noise plateau above the signal, so the noise floor, SNR and
        band edges are unreliable -- e.g. Clarius L15, fs 20 MHz, imaging depth past ~25 mm):
        "error" (default) raises NyquistError, "warn" issues a NyquistWarning, "ignore" does
        neither. The affected depths are in result.nyquist_limited either way.
    block_lines: lines per lateral block; None (as MATLAB) -> block_width_mm / the line spacing at
        the middle of this image's depth range. Pass the reference's block_lines for the sample
        when the two were acquired at different imaging depths on a sector probe (their mid-depth
        line spacings differ, which would otherwise give different block grids).
    ax_step_mm takes precedence over n_ax_locations if both are given.
    """
    if on_nyquist not in ("error", "warn", "ignore"):
        raise ValueError("on_nyquist must be 'error', 'warn' or 'ignore'")
    rf = np.asarray(rf_frames, dtype=float)
    if rf.ndim == 2:
        rf = rf[None]
    n_frames, n_samples, n_lines = rf.shape
    axial = np.asarray(axial_position_mm, dtype=float)
    lateral = np.asarray(lateral_position_mm, dtype=float)
    if axial.shape != (n_samples, n_lines) or lateral.shape != (n_samples, n_lines):
        raise ValueError("Position arrays must match the RF frame shape")
    search = -np.inf if search_above is None else float(search_above)
    avoid = tuple(np.asarray(avoid_freq, dtype=float).ravel())

    # ---- spatial calibration (centre line)
    mid0 = _mround(n_lines / 2) - 1
    centre_ax = axial[:, mid0]
    d_ax = centre_ax[2] - centre_ax[1]
    fs = C_MM_US / 2 / d_ax if sampling_frequency_mhz is None else float(sampling_frequency_mhz)

    # ---- Hanning window (odd), unit energy
    win = _mround(window_length_mm / d_ax)
    if win % 2 == 0:
        win += 1
    half = (win - 1) // 2
    w = matlab_hanning(win)
    w = w / np.sqrt(np.sum(w ** 2))
    nfft = win
    npos = nfft // 2 + 1
    freq_axis = np.arange(npos) * fs / nfft

    # ---- window centres (1-based, as MATLAB)
    start_1 = half + 1 if ax_start_mm is None else max(int(np.argmin(np.abs(centre_ax - ax_start_mm))) + 1, half + 1)
    end_1 = n_samples - half if ax_end_mm is None else min(int(np.argmin(np.abs(centre_ax - ax_end_mm))) + 1,
                                                           n_samples - half)
    start_mm, end_mm = centre_ax[start_1 - 1], centre_ax[end_1 - 1]
    if ax_step_mm is not None:
        spacing = max(1, _mround(ax_step_mm / d_ax))
    elif n_ax_locations is not None:
        spacing = max(1, _mround(len(range(start_1, end_1 + 1)) / n_ax_locations))
    else:
        raise ValueError("Give ax_step_mm or n_ax_locations")
    locs_1 = np.arange(start_1, end_1 + 1, spacing)
    if locs_1.size == 0:
        raise ValueError(f"No full {window_length_mm:g} mm window fits in the requested depth range "
                         f"({n_samples}-sample frame); shorten the window or widen the range.")
    rows = locs_1 - 1
    depths = centre_ax[rows]
    n_depths = rows.size

    # ---- lateral blocks: fixed number of lines, half overlap
    if block_lines is None:
        mid_ax = int(np.argmin(np.abs(centre_ax - (start_mm + end_mm) / 2)))
        d_lat = abs(lateral[mid_ax, mid0 + 1] - lateral[mid_ax, mid0])
        block_lines = max(1, _mround(block_width_mm / d_lat))
    block_lines = int(block_lines)
    if block_lines > n_lines:
        raise ValueError(f"A {block_width_mm:g} mm block needs {block_lines} A-lines but the image has {n_lines}; "
                         "use a narrower block.")
    block_step = max(1, _mround(block_lines / 2))
    block_start_1 = np.arange(1, n_lines - block_lines + 2, block_step)
    block_end_1 = block_start_1 + block_lines - 1
    block_centre_1 = _mround((block_start_1 + block_end_1) / 2)
    n_blocks = block_start_1.size
    block_x = lateral[np.ix_(rows, block_centre_1 - 1)]
    block_z = axial[np.ix_(rows, block_centre_1 - 1)]

    # ---- spectra: all-line average and lateral-block averages
    mean_ps = np.full((n_depths, npos), np.nan)
    block_ps = np.full((n_depths, n_blocks, npos), np.nan)
    for d, loc_1 in enumerate(locs_1):
        ws, we = loc_1 - half, loc_1 + half          # 1-based inclusive
        if ws < 1 or we > n_samples:
            continue
        seg = rf[:, ws - 1:we, :]                   # (frames, win, lines)
        seg = seg - seg.mean(axis=1, keepdims=True)
        x = np.fft.fft(seg * w[None, :, None], nfft, axis=1)
        p = np.abs(x[:, :npos, :]) ** 2
        p[:, 1:-1, :] *= 2                           # fold negative frequencies (as MATLAB 2:end-1)
        p_line = p.mean(axis=0)                      # (npos, lines), frame-averaged
        mean_ps[d] = p_line.mean(axis=1)
        for b in range(n_blocks):
            block_ps[d, b] = p_line[:, block_start_1[b] - 1:block_end_1[b]].mean(axis=1)

    # ---- band edges and SNR
    prof_m = [_run_bw(mean_ps[d], freq_axis, avoid, ABOVE_NOISE_DB, search, with_method=True)
              for d in range(n_depths)]
    nyquist_limited = np.array([bool(m[5]) for m in prof_m])
    if nyquist_limited.any() and on_nyquist != "ignore":
        dl = depths[nyquist_limited]
        msg = (f"Signal reaches Nyquist ({fs / 2:g} MHz) at {nyquist_limited.sum()} of {n_depths} depths "
               f"({dl.min():.1f}-{dl.max():.1f} mm): there is no out-of-band noise above the signal band, so "
               f"the noise floor, SNR and band edges are not reliable. Do not use this data for spectral "
               f"analysis (known for Clarius L15, fs 20 MHz, with imaging depth past ~25 mm).")
        if on_nyquist == "error":
            raise NyquistError(msg)
        import warnings
        warnings.warn(msg, NyquistWarning, stacklevel=2)
    prof = np.array([m[:5] for m in prof_m])
    blk = np.array([[_run_bw(block_ps[d, b], freq_axis, avoid, ABOVE_NOISE_DB, search) for b in range(n_blocks)]
                    for d in range(n_depths)])
    f_low, f_high, peak_freq, noise_floor, snr = (prof[:, k] for k in range(5))
    f_low_2d, f_high_2d, peak_freq_2d, _, snr_2d = (blk[..., k] for k in range(5))

    signal_power = snr - ABOVE_NOISE_DB
    signal_power_2d = snr_2d - ABOVE_NOISE_DB
    with np.errstate(invalid="ignore"):
        usable = (signal_power >= MIN_SIGNAL_POWER_DB) & ~np.isnan(f_low)
        usable_2d = (signal_power_2d >= MIN_SIGNAL_POWER_DB) & ~np.isnan(f_low_2d)

    return FirstPass(
        mean_ps=mean_ps, freq_axis=freq_axis, depths=depths, rows=rows, f_low=f_low, f_high=f_high,
        peak_freq=peak_freq, noise_floor_db=noise_floor, snr_db=snr, signal_power_db=signal_power,
        usable=usable, snr_2d=snr_2d, signal_power_2d=signal_power_2d, usable_2d=usable_2d,
        f_low_2d=f_low_2d, f_high_2d=f_high_2d, peak_freq_2d=peak_freq_2d, block_x=block_x, block_z=block_z,
        block_ps=block_ps, block_start=block_start_1 - 1, block_centre=block_centre_1 - 1,
        block_lines=block_lines, block_width_mm=block_width_mm, window_length_mm=window_length_mm,
        window_samples=win, fs_mhz=float(fs), delta_ax_mm=float(d_ax), num_frames=n_frames, num_lines=n_lines,
        avoid_freq=avoid, search_above=search, nyquist_limited=nyquist_limited,
    )


def calc_first_pass_from_image(image_data, source: str = "phantom", frames: Optional[Sequence[int]] = None,
                               **kwargs) -> FirstPass:
    """calc_first_pass() on a loaded QuantUS UltrasoundRfImage ("phantom": every reference frame,
    "scan": the scan's frames), with positions from positions_from_image()."""
    from .correlation import _frames_from_image, positions_from_image

    rf = _frames_from_image(image_data, source, frames)
    axial, lateral = positions_from_image(image_data, rf.shape[1], rf.shape[2])
    return calc_first_pass(rf, axial, lateral, **kwargs)


# ---------------------------------------------------------------------------------------------
# IntersectFirstPass
# ---------------------------------------------------------------------------------------------
@dataclass
class FirstPassIntersect:
    """Blocks usable in both the sample (S1) and the reference (L1), as IntersectFirstPass."""
    usable_both_2d: np.ndarray
    signal_power_min_2d: np.ndarray
    f_low_both_2d: np.ndarray
    f_high_both_2d: np.ndarray
    usable_both: np.ndarray
    f_low_both: np.ndarray
    f_high_both: np.ndarray
    sample: FirstPass = field(repr=False)
    reference: FirstPass = field(repr=False)

    # what roi_band_check needs, from the (depth-trimmed) sample grid
    @property
    def block_x(self):
        return self.sample.block_x

    @property
    def block_z(self):
        return self.sample.block_z

    @property
    def depths(self):
        return self.sample.depths

    @property
    def freq_axis(self):
        return self.sample.freq_axis


def _nanmax(a, b):
    c = np.fmax(a, b)
    return np.where(np.isnan(a) | np.isnan(b), np.nan, c)


def _nanmin(a, b):
    c = np.fmin(a, b)
    return np.where(np.isnan(a) | np.isnan(b), np.nan, c)


# FirstPass fields with one row per depth (trimmed together; block_start / block_centre are per
# block and must not be touched even when n_blocks happens to equal n_depths)
_PER_DEPTH_FIELDS = ("mean_ps", "depths", "rows", "f_low", "f_high", "peak_freq", "noise_floor_db", "snr_db",
                     "signal_power_db", "usable", "snr_2d", "signal_power_2d", "usable_2d", "f_low_2d", "f_high_2d",
                     "peak_freq_2d", "block_x", "block_z", "block_ps", "nyquist_limited")


def _trim_depths(fp: FirstPass, n: int) -> FirstPass:
    import dataclasses

    updates = {name: getattr(fp, name)[:n] for name in _PER_DEPTH_FIELDS if getattr(fp, name) is not None}
    return dataclasses.replace(fp, **updates)


def intersect_first_pass(sample: FirstPass, reference: FirstPass) -> FirstPassIntersect:
    """Combine sample and reference first passes block by block (port of IntersectFirstPass.m).

    Both must come from calc_first_pass with the same window / block settings. If they cover
    different depths (different imaging depth), the shared leading depth rows are kept; on a
    sector probe compute the sample with block_lines=reference.block_lines so the block grids
    match (windows_and_first_pass does this). Sample and reference must start at the same depth
    (no RF delay offset).
    usable_both_2d = usable in both; f_low_both_2d = max of the two f_low, f_high_both_2d = min of
    the two f_high (NaN if either is NaN); signal_power_min_2d = the smaller signal power.
    """
    ns, nl = sample.depths.size, reference.depths.size
    if ns != nl:
        n = min(ns, nl)
        if np.max(np.abs(sample.depths[:n] - reference.depths[:n])) > 1e-6:
            raise ValueError("Sample and reference depth rows differ")
        sample, reference = _trim_depths(sample, n), _trim_depths(reference, n)
    if sample.block_x.shape != reference.block_x.shape:
        raise ValueError("Sample and reference block grids differ; use the same window / block settings")
    dx = np.abs(sample.block_x - reference.block_x)
    if (np.any(dx[~np.isnan(dx)] > 1e-6) or np.max(np.abs(sample.depths - reference.depths)) > 1e-6
            or sample.freq_axis.size != reference.freq_axis.size):
        raise ValueError("Sample and reference block grids differ; use the same window / block settings / depth range")
    return FirstPassIntersect(
        usable_both_2d=sample.usable_2d & reference.usable_2d,
        signal_power_min_2d=np.fmin(sample.signal_power_2d, reference.signal_power_2d),
        f_low_both_2d=_nanmax(sample.f_low_2d, reference.f_low_2d),
        f_high_both_2d=_nanmin(sample.f_high_2d, reference.f_high_2d),
        usable_both=sample.usable & reference.usable,
        f_low_both=_nanmax(sample.f_low, reference.f_low),
        f_high_both=_nanmin(sample.f_high, reference.f_high),
        sample=sample, reference=reference,
    )


# ---------------------------------------------------------------------------------------------
# ROIBandCheck
# ---------------------------------------------------------------------------------------------
def inpolygon(x, y, xv, yv, tol: float = 1e-9) -> np.ndarray:
    """MATLAB inpolygon (in OR on the boundary) for points x, y and polygon vertices xv, yv."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    px, py = x.ravel(), y.ravel()
    xv = np.asarray(xv, dtype=float).ravel()
    yv = np.asarray(yv, dtype=float).ravel()
    x0, y0 = xv, yv
    x1, y1 = np.roll(xv, -1), np.roll(yv, -1)
    inside = np.zeros(px.size, dtype=bool)
    on = np.zeros(px.size, dtype=bool)
    scale = max(np.ptp(xv), np.ptp(yv), 1.0)
    for a, b, c, d in zip(x0, y0, x1, y1):
        # crossing number (ray to +x)
        cond = (b > py) != (d > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = a + (py - b) * (c - a) / (d - b)
        inside ^= cond & (px < xc)
        # on the segment
        cross = (c - a) * (py - b) - (d - b) * (px - a)
        within = ((px >= min(a, c) - tol * scale) & (px <= max(a, c) + tol * scale)
                  & (py >= min(b, d) - tol * scale) & (py <= max(b, d) + tol * scale))
        on |= within & (np.abs(cross) <= tol * scale * max(np.hypot(c - a, d - b), 1e-300))
    return (inside | on).reshape(x.shape)


@dataclass
class ROIBand:
    """Output of roi_band_check (ROIBandCheck R struct)."""
    depths: np.ndarray
    num_blocks: np.ndarray
    f_low_s1: np.ndarray
    f_high_s1: np.ndarray
    f_low_l1: np.ndarray
    f_high_l1: np.ndarray
    signal_power_s1: np.ndarray
    signal_power_l1: np.ndarray
    f_low_row: np.ndarray
    f_high_row: np.ndarray
    row_usable: np.ndarray
    all_rows_usable: bool
    num_blocks_in_roi: int
    f_low_roi: float
    f_high_roi: float
    bw_mhz: float
    f_low_limit_depth: float
    f_high_limit_depth: float
    rows: np.ndarray  # first-pass depth rows (0-based) with block centres inside the ROI


def roi_band_check(intersect: FirstPassIntersect, x_poly, z_poly) -> ROIBand:
    """Usable band of an ROI valid in both sample and reference (port of ROIBandCheck.m).

    For each depth row with block centres inside the polygon (Cartesian mm, inside or on the
    boundary), the block spectra inside are averaged (separately for sample and reference) and
    find_bw_edges is run on each average. Shared band per row = [max f_low, min f_high]; usable =
    a band exists and signal power >= 5 dB in both. ROI band = [max shared f_low, min shared
    f_high] over the usable rows.
    """
    s, l = intersect.sample, intersect.reference
    inside = inpolygon(s.block_x, s.block_z, x_poly, z_poly)
    rows = np.nonzero(inside.any(axis=1))[0]
    if rows.size == 0:
        raise ValueError("No block centres inside the ROI - draw a larger ROI.")
    k = rows.size
    out = {name: np.full(k, np.nan) for name in ("f_low_s1", "f_high_s1", "f_low_l1", "f_high_l1",
                                                  "signal_power_s1", "signal_power_l1")}
    num_blocks = np.zeros(k, dtype=int)

    def run(ps):
        ps = np.asarray(ps).ravel()
        if np.any(np.isnan(ps)):
            return np.nan, np.nan, np.nan
        fl, fh, info = find_bw_edges(ps, s.freq_axis, above_noise_db=s.above_noise_db,
                                     avoid_freq=s.avoid_freq, search_above=s.search_above)
        return fl, fh, info["signal_power_db"]

    for i, r in enumerate(rows):
        idx = np.nonzero(inside[r])[0]
        num_blocks[i] = idx.size
        out["f_low_s1"][i], out["f_high_s1"][i], out["signal_power_s1"][i] = run(s.block_ps[r, idx].mean(axis=0))
        out["f_low_l1"][i], out["f_high_l1"][i], out["signal_power_l1"][i] = run(l.block_ps[r, idx].mean(axis=0))

    f_low_row = np.fmax(out["f_low_s1"], out["f_low_l1"])
    f_high_row = np.fmin(out["f_high_s1"], out["f_high_l1"])
    no_band = (np.isnan(out["f_low_s1"]) | np.isnan(out["f_low_l1"]) | np.isnan(out["f_high_s1"])
               | np.isnan(out["f_high_l1"]))
    f_low_row[no_band] = np.nan
    f_high_row[no_band] = np.nan
    with np.errstate(invalid="ignore"):
        f_low_row[f_high_row <= f_low_row] = np.nan
    f_high_row[np.isnan(f_low_row)] = np.nan
    with np.errstate(invalid="ignore"):
        row_usable = (~np.isnan(f_low_row) & (out["signal_power_s1"] >= s.min_signal_power_db)
                      & (out["signal_power_l1"] >= s.min_signal_power_db))

    f_low_roi = f_high_roi = bw = d_low = d_high = np.nan
    depths = s.depths[rows]
    if row_usable.any():
        u = np.nonzero(row_usable)[0]
        il, ih = int(np.argmax(f_low_row[u])), int(np.argmin(f_high_row[u]))
        f_low_roi, f_high_roi = float(f_low_row[u][il]), float(f_high_row[u][ih])
        d_low, d_high = float(depths[u[il]]), float(depths[u[ih]])
        bw = f_high_roi - f_low_roi
        if bw <= 0:
            bw = np.nan
    return ROIBand(depths=depths, num_blocks=num_blocks, f_low_row=f_low_row, f_high_row=f_high_row,
                   row_usable=row_usable, all_rows_usable=bool(row_usable.all()),
                   num_blocks_in_roi=int(inside.sum()), f_low_roi=f_low_roi, f_high_roi=f_high_roi, bw_mhz=bw,
                   f_low_limit_depth=d_low, f_high_limit_depth=d_high, rows=rows, **out)
