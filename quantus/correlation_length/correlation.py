"""Axial and lateral speckle correlation lengths vs. depth, used to choose QUS window sizes.

Port of CalcAxLatCorrelationCoeffCurves.m (S1Study, version of 2026-11-16: window centres from
AxialPosition) -- kept index-for-index faithful to the MATLAB so the two can be checked against
each other; see README.md for the known differences we may want to change.

Method, per frame and per depth:
  1. Detrend each A-line of the RF frame (linear), take the Hilbert envelope.
  2. Take a window (window_length_mm deep x window_width_mm wide) centred on the middle line.
  3. Detrend each A-line of the window envelope, add its mean back, square -> intensity.
  4. Lateral: correlation coefficient between A-line c and A-line c+shift (over depth),
     averaged over c. Axial: same between row r and row r+shift (over A-lines).
  5. First shift where the curve drops below `threshold`, interpolated linearly between the
     two neighbouring shifts -> correlation length in mm. The axial length is doubled: the
     pulse-echo axial correlation length is the full width (lateral stays one-sided).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.signal import detrend, hilbert

def _mround(x: float) -> int:
    """MATLAB round (half away from zero); Python's round() is banker's rounding."""
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


@dataclass
class CorrelationLengths:
    depths_mm: np.ndarray          # (n_depths,) depth of each window centre
    ax_corr_mm: np.ndarray         # (n_depths, n_frames) full axial correlation length
    lat_corr_mm: np.ndarray        # (n_depths, n_frames) lateral correlation width
    ax_spacing_mm: float           # axial sample spacing
    lat_spacing_mm: np.ndarray     # (n_depths,) A-line spacing at each depth
    window_samples: np.ndarray     # (n_depths,) axial samples in the measurement window
    window_lines: np.ndarray       # (n_depths,) A-lines in the measurement window
    threshold: float
    window_length_mm: float
    window_width_mm: float
    ax_window_factor: float = 10.0  # recommended axial window = factor x mean axial length
    lat_window_factor: float = 5.0  # recommended lateral window = factor x mean lateral width
    lat_corr_curves: Optional[np.ndarray] = field(default=None, repr=False)  # (n_depths, n_frames, n_shifts+1)
    ax_corr_curves: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_frames(self) -> int:
        return self.ax_corr_mm.shape[1]

    @property
    def ax_mean_mm(self) -> np.ndarray:
        return np.nanmean(self.ax_corr_mm, axis=1)

    @property
    def ax_std_mm(self) -> np.ndarray:
        return np.nanstd(self.ax_corr_mm, axis=1, ddof=1) if self.n_frames > 1 else np.zeros(len(self.depths_mm))

    @property
    def lat_mean_mm(self) -> np.ndarray:
        return np.nanmean(self.lat_corr_mm, axis=1)

    @property
    def lat_std_mm(self) -> np.ndarray:
        return np.nanstd(self.lat_corr_mm, axis=1, ddof=1) if self.n_frames > 1 else np.zeros(len(self.depths_mm))

    # Lengths in samples (axial) and A-lines (lateral): the units the QUS windows are built in.
    @property
    def ax_corr_samples(self) -> np.ndarray:
        """(n_depths, n_frames) full axial correlation length in samples."""
        return self.ax_corr_mm / self.ax_spacing_mm

    @property
    def lat_corr_lines(self) -> np.ndarray:
        """(n_depths, n_frames) lateral correlation width in A-lines (line spacing at that depth)."""
        return self.lat_corr_mm / self.lat_spacing_mm[:, None]

    @property
    def ax_mean_samples(self) -> np.ndarray:
        return np.nanmean(self.ax_corr_samples, axis=1)

    @property
    def ax_std_samples(self) -> np.ndarray:
        return self.ax_std_mm / self.ax_spacing_mm

    @property
    def lat_mean_lines(self) -> np.ndarray:
        return np.nanmean(self.lat_corr_lines, axis=1)

    @property
    def lat_std_lines(self) -> np.ndarray:
        return self.lat_std_mm / self.lat_spacing_mm

    # Recommended QUS window at each depth: factor x mean length, rounded UP to a whole number
    # of samples / A-lines (NaN where the length is undefined). The mm values are those whole
    # numbers times the spacing, so mm and counts always describe the same window.
    # ax_window_factor / lat_window_factor are inputs (correlation_lengths kwargs) and can also
    # be changed on the result afterwards; these recompute.
    @property
    def recommended_ax_window_samples(self) -> np.ndarray:
        return _ceil_count(self.ax_window_factor * self.ax_mean_samples)

    @property
    def recommended_lat_window_lines(self) -> np.ndarray:
        return _ceil_count(self.lat_window_factor * self.lat_mean_lines)

    @property
    def recommended_ax_window_mm(self) -> np.ndarray:
        return self.recommended_ax_window_samples * self.ax_spacing_mm

    @property
    def recommended_lat_window_mm(self) -> np.ndarray:
        return self.recommended_lat_window_lines * self.lat_spacing_mm

    def recommended_window(self) -> dict:
        """One window for the whole checked depth range: the largest per-depth recommendation.

        Axial in samples (and mm), lateral in A-lines (and mm at the depth where the maximum
        occurs; on a curvilinear probe the line spacing grows with depth), plus those depths.
        """
        def largest(counts, spacing_mm):
            if not np.any(np.isfinite(counts)):
                return np.nan, np.nan, np.nan
            i = int(np.nanargmax(counts))
            spacing = spacing_mm[i] if np.ndim(spacing_mm) else spacing_mm
            return float(counts[i]), float(counts[i] * spacing), float(self.depths_mm[i])

        ax_n, ax_mm, ax_depth = largest(self.recommended_ax_window_samples, self.ax_spacing_mm)
        lat_n, lat_mm, lat_depth = largest(self.recommended_lat_window_lines, self.lat_spacing_mm)
        return {
            "ax_window_samples": ax_n, "ax_window_mm": ax_mm, "ax_at_depth_mm": ax_depth,
            "lat_window_lines": lat_n, "lat_window_mm": lat_mm, "lat_at_depth_mm": lat_depth,
            "depth_range_mm": (float(np.min(self.depths_mm)), float(np.max(self.depths_mm))),
            "ax_window_factor": self.ax_window_factor, "lat_window_factor": self.lat_window_factor,
        }

    def describe_window(self) -> str:
        """One-line statement of the whole-range window (see recommended_window)."""
        w = self.recommended_window()
        d0, d1 = w["depth_range_mm"]
        return (f"Window for {d0:.1f}-{d1:.1f} mm ({self.n_frames} frames, threshold {self.threshold:g}): "
                f"axial {w['ax_window_samples']:.0f} samples ({w['ax_window_mm']:.2f} mm, "
                f"{w['ax_window_factor']:g}x length, max at {w['ax_at_depth_mm']:.1f} mm) x "
                f"lateral {w['lat_window_lines']:.0f} A-lines ({w['lat_window_mm']:.2f} mm, "
                f"{w['lat_window_factor']:g}x width, max at {w['lat_at_depth_mm']:.1f} mm)")

    def to_table(self):
        """One row per depth: lengths and recommended windows in samples / A-lines and mm (pandas DataFrame)."""
        import pandas as pd

        return pd.DataFrame({
            "depth_mm": self.depths_mm,
            "ax_length_samples": self.ax_mean_samples,
            "ax_length_sd_samples": self.ax_std_samples,
            "ax_length_mm": self.ax_mean_mm,
            "lat_width_lines": self.lat_mean_lines,
            "lat_width_sd_lines": self.lat_std_lines,
            "lat_width_mm": self.lat_mean_mm,
            "line_spacing_mm": self.lat_spacing_mm,
            f"ax_window_samples ({self.ax_window_factor:g}x)": self.recommended_ax_window_samples,
            "ax_window_mm": self.recommended_ax_window_mm,
            f"lat_window_lines ({self.lat_window_factor:g}x)": self.recommended_lat_window_lines,
            "lat_window_mm": self.recommended_lat_window_mm,
            "n_frames": np.sum(np.isfinite(self.lat_corr_mm), axis=1),
        })


def _ceil_count(x: np.ndarray) -> np.ndarray:
    """Round up to whole samples / lines, keeping NaN (tolerates float noise like 5.0000000001)."""
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), np.ceil(x - 1e-9), np.nan)


def envelope_frames(rf_frames: np.ndarray) -> np.ndarray:
    """Detrend each A-line, remove its mean, and take the Hilbert envelope.

    rf_frames: (n_frames, n_samples, n_lines). Returns the same shape.
    """
    det = detrend(np.asarray(rf_frames, dtype=np.float64), axis=1, type="linear")
    det = det - det.mean(axis=1, keepdims=True)
    return np.abs(hilbert(det, axis=1))


def _neighbour_corr_curve(window: np.ndarray, n_half: int, n_shifts: int, along_rows: bool) -> np.ndarray:
    """Mean correlation coefficient between element i and i+shift, for shift = 0..n_shifts.

    along_rows=False: elements are columns (A-lines), correlated over depth -> lateral curve.
    along_rows=True:  elements are rows (depths), correlated over A-lines -> axial curve.
    Elements i = 0..n_half-1, matching the MATLAB loops.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(window, rowvar=along_rows)
    c = np.atleast_2d(c)
    i = np.arange(n_half)
    return np.array([np.mean(c[i, i + s]) for s in range(n_shifts + 1)])


def _first_crossing_shift(curve: np.ndarray, threshold: float) -> Optional[float]:
    """Shift (in samples / lines, fractional) where the curve first drops below `threshold`.

    Solved exactly on the straight segment between the last shift at/above the threshold and
    the first one below it -- the same linear interpolation MATLAB does on a 0.001-shift grid,
    without the grid (MATLAB also reports one grid step late from its 1-based index; up to
    0.002 shift in all). None if the curve never drops below the threshold.
    """
    with np.errstate(invalid="ignore"):
        below = np.nonzero(curve < threshold)[0]
    if below.size == 0:
        return None
    k = int(below[0])
    if k == 0:
        return 0.0
    y0, y1 = curve[k - 1], curve[k]
    if not np.isfinite(y0):
        return None
    return (k - 1) + (y0 - threshold) / (y0 - y1)


def _distance_mm(axial: np.ndarray, lateral: np.ndarray, a, b) -> float:
    """Straight-line distance between samples a=(row, col) and b=(row, col): correct for linear,
    curvilinear and tilted geometry alike."""
    return float(np.hypot(axial[a] - axial[b], lateral[a] - lateral[b]))


def _sample_spacings_mm(axial: np.ndarray, lateral: np.ndarray, row: int, col: int):
    """(along-line, line-to-line) spacing at sample (row, col), 0-based."""
    r2 = row + 1 if row + 1 < axial.shape[0] else row - 1
    c2 = col - 1 if col > 0 else col + 1
    return _distance_mm(axial, lateral, (row, col), (r2, col)), _distance_mm(axial, lateral, (row, col), (row, c2))


def _odd_count(length_mm: float, spacing_mm: float) -> int:
    """Samples / lines in a window of length_mm, made odd so there is a central one."""
    n = _mround(length_mm / spacing_mm)
    return n + 1 if n % 2 == 0 else n


def _axial_centres(centre_ax: np.ndarray, n_samples: int, half_ax: int, window_length_mm: float,
                   ax_start_mm: Optional[float], ax_end_mm: Optional[float],
                   n_ax_locations: Optional[int]) -> np.ndarray:
    """Window-centre rows (1-based, as MATLAB): the sample nearest each requested depth on the
    centre line, kept inside the image so a full window fits; None -> first / last full window.
    n_ax_locations None -> a centre every half window; spacing at least 1 sample."""
    first_1, last_1 = half_ax + 1, n_samples - half_ax
    start_1 = first_1 if ax_start_mm is None else max(int(np.argmin(np.abs(centre_ax - ax_start_mm))) + 1, first_1)
    end_1 = last_1 if ax_end_mm is None else min(int(np.argmin(np.abs(centre_ax - ax_end_mm))) + 1, last_1)
    if end_1 < start_1:
        raise ValueError(f"No full {window_length_mm:g} mm window fits between the requested depths "
                         f"in a {n_samples}-sample frame; reduce window_length_mm or widen the depth range.")
    if n_ax_locations is None:
        step = max(1, half_ax)
    else:
        step = max(1, _mround((end_1 - start_1 + 1) / n_ax_locations))
    return np.arange(start_1, end_1 + 1, step)


def _measure_window(env: np.ndarray, r0: int, r1: int, c0: int, c1: int, d_ax: float, d_lat: float,
                    threshold: float):
    """Correlation lengths of one measurement window env[:, r0:r1, c0:c1] in every frame.

    Detrend each A-line of the window envelope, add its mean back, square to intensity; then the
    lateral (A-line to A-line) and axial (row to row) mean correlation curves and their threshold
    crossings. Returns (lat_mm, ax_half_mm, lat_curves, ax_curves), the first two (n_frames,)
    with NaN where a curve never drops below the threshold. The axial value is the one-sided
    (half) length; callers double it.
    """
    win_env = env[:, r0:r1, c0:c1]
    win_mean = win_env.mean(axis=1, keepdims=True)
    windows = (detrend(win_env, axis=1, type="linear") + win_mean) ** 2
    half_ax, half_lat = (r1 - r0 - 1) // 2, (c1 - c0 - 1) // 2
    n_frames = env.shape[0]
    lat_mm, ax_mm = np.full(n_frames, np.nan), np.full(n_frames, np.nan)
    lat_curves, ax_curves = [], []
    for f in range(n_frames):
        lat_curve = _neighbour_corr_curve(windows[f], half_lat, half_lat + 1, along_rows=False)
        ax_curve = _neighbour_corr_curve(windows[f], half_ax, half_ax + 1, along_rows=True)
        s = _first_crossing_shift(lat_curve, threshold)
        lat_mm[f] = np.nan if s is None else s * d_lat
        s = _first_crossing_shift(ax_curve, threshold)
        # Never below the threshold -> NaN (MATLAB instead returned max(curve) * spacing / 1000,
        # a correlation value, not a distance).
        ax_mm[f] = np.nan if s is None else s * d_ax
        lat_curves.append(lat_curve)
        ax_curves.append(ax_curve)
    return lat_mm, ax_mm, lat_curves, ax_curves


def _prepare(rf_frames: np.ndarray, axial_position_mm: np.ndarray, lateral_position_mm: np.ndarray):
    rf = np.asarray(rf_frames, dtype=np.float64)
    if rf.ndim == 2:
        rf = rf[None]
    if rf.ndim != 3:
        raise ValueError(f"rf_frames must be (samples, lines) or (frames, samples, lines), got {rf.shape}")
    n_samples, n_lines = rf.shape[1:]
    axial = np.asarray(axial_position_mm, dtype=float)
    lateral = np.asarray(lateral_position_mm, dtype=float)
    if axial.shape != (n_samples, n_lines) or lateral.shape != (n_samples, n_lines):
        raise ValueError("axial_position_mm and lateral_position_mm must match the RF frame shape "
                         f"{(n_samples, n_lines)}")
    return envelope_frames(rf), axial, lateral


def correlation_lengths(rf_frames: np.ndarray, axial_position_mm: np.ndarray, lateral_position_mm: np.ndarray,
                        window_length_mm: float, window_width_mm: float,
                        ax_start_mm: Optional[float] = None, ax_end_mm: Optional[float] = None,
                        n_ax_locations: Optional[int] = None,
                        threshold: float = 0.5, ax_window_factor: float = 10.0, lat_window_factor: float = 5.0,
                        keep_curves: bool = False) -> CorrelationLengths:
    """Axial and lateral correlation lengths vs. depth on the centre A-line (port of
    CalcAxLatCorrelationCoeffCurves.m). correlation_length_map() does the same across the image.

    Args:
        rf_frames: RF data, (n_samples, n_lines) or (n_frames, n_samples, n_lines), pre-scan-conversion.
        axial_position_mm, lateral_position_mm: (n_samples, n_lines) physical position of every
            RF sample in mm (as in the MATLAB RFData.mat). Spacings are straight-line distances
            between neighbouring samples (along the centre line, and from the centre line to its
            left neighbour at each window's depth), so linear, curvilinear and tilted geometry
            are all handled. See positions_from_image() to build these for a QuantUS image.
        window_length_mm, window_width_mm: size of the measurement window at each depth.
        ax_start_mm, ax_end_mm: depth range of the window centres (mm, as in axial_position_mm,
            i.e. including any RF start delay). Each snaps to the nearest sample on the centre
            line and is kept far enough inside the image for a full window. None -> the
            first / last full window in the image.
        n_ax_locations: approximate number of depths; None -> a window centre every half window.
        threshold: correlation coefficient that defines the length (default 0.5).
        ax_window_factor, lat_window_factor: multipliers for the recommended QUS window
            (MATLAB: axial 10x, lateral 0.5*10 = 5x).
        keep_curves: also return the mean correlation-vs-shift curve for every depth/frame.
    """
    env, axial, lateral = _prepare(rf_frames, axial_position_mm, lateral_position_mm)
    n_frames, n_samples, n_lines = env.shape

    # MATLAB indices are 1-based; *_1 variables keep that convention, *0 are 0-based.
    mid0 = _mround(n_lines / 2) - 1
    centre_ax = axial[:, mid0]  # depth of every sample on the centre line
    d_ax = _distance_mm(axial, lateral, (2, mid0), (1, mid0))  # MATLAB: CentreAx(3) - CentreAx(2)

    win_ax = _odd_count(window_length_mm, d_ax)
    half_ax = (win_ax - 1) // 2
    ax_locs_1 = _axial_centres(centre_ax, n_samples, half_ax, window_length_mm,
                               ax_start_mm, ax_end_mm, n_ax_locations)

    n_depths = len(ax_locs_1)
    ax_corr = np.full((n_depths, n_frames), np.nan)
    lat_corr = np.full((n_depths, n_frames), np.nan)
    lat_spacing = np.full(n_depths, np.nan)
    win_lines_all = np.zeros(n_depths, dtype=int)
    lat_curves, ax_curves = [], []

    for k, loc_1 in enumerate(ax_locs_1):
        row = loc_1 - 1
        d_lat = _distance_mm(axial, lateral, (row, mid0), (row, mid0 - 1))
        lat_spacing[k] = d_lat
        win_lat = _odd_count(window_width_mm, d_lat)
        win_lines_all[k] = win_lat
        half_lat = (win_lat - 1) // 2
        c0, c1 = mid0 - half_lat, mid0 + half_lat + 1
        if c0 < 0 or c1 > n_lines:
            raise ValueError(f"Window at depth index {loc_1} needs {win_lat} A-lines but only {n_lines} exist; "
                             "reduce window_width_mm.")
        lat_corr[k], ax_corr[k], lat_k, ax_k = _measure_window(
            env, row - half_ax, row + half_ax + 1, c0, c1, d_ax, d_lat, threshold)
        if keep_curves:
            lat_curves.append(lat_k)
            ax_curves.append(ax_k)

    return CorrelationLengths(
        depths_mm=centre_ax[ax_locs_1 - 1],  # depth of each window centre (MATLAB: CentreAx(AxialLocations))
        ax_corr_mm=ax_corr * 2,  # pulse-echo: the axial length is the full width (2x the one-sided crossing)
        lat_corr_mm=lat_corr,
        ax_spacing_mm=float(d_ax),
        lat_spacing_mm=lat_spacing,
        window_samples=np.full(n_depths, win_ax),
        window_lines=win_lines_all,
        threshold=threshold,
        window_length_mm=window_length_mm,
        window_width_mm=window_width_mm,
        ax_window_factor=ax_window_factor,
        lat_window_factor=lat_window_factor,
        lat_corr_curves=np.array(lat_curves, dtype=object) if keep_curves else None,
        ax_corr_curves=np.array(ax_curves, dtype=object) if keep_curves else None,
    )


def positions_from_image(image_data, n_samples: int, n_lines: int):
    """Physical (axial, lateral) position in mm of every pre-scan-conversion RF sample.

    - Sector / curvilinear data (image_data.width set, in degrees): lines fan out over `width`
      degrees about `tilt`, with sample i at radius start_depth + i*axial_res from the apex.
      start_depth is in metres, measured from the sector apex (the loaders already add the
      probe radius / apex offset: Clarius startDepth1 + radius, VEVO F2 apex_offset + depth).
      Axial position is measured from the first sample on the centre line.
    - Otherwise linear: image_data.axial / image_data.lateral (mm) if the loader provides them,
      else axial_res / lateral_res spacing.
    """
    ax_res = float(image_data.axial_res)
    width = getattr(image_data, "width", None)
    start_depth = getattr(image_data, "start_depth", None)
    if width and start_depth is not None:
        r = start_depth * 1000 + np.arange(n_samples) * ax_res          # mm from the apex
        tilt = getattr(image_data, "tilt", 0) or 0
        theta = np.deg2rad(tilt + width * (np.arange(n_lines) / (n_lines - 1) - 0.5))
        axial = r[:, None] * np.cos(theta)[None, :] - start_depth * 1000
        lateral = r[:, None] * np.sin(theta)[None, :]
        return axial, lateral

    axial_1d = getattr(image_data, "axial", None)
    axial_1d = np.asarray(axial_1d, float) if axial_1d is not None else np.arange(n_samples) * ax_res
    lateral_1d = getattr(image_data, "lateral", None)
    if lateral_1d is not None and np.size(lateral_1d) == n_lines:
        lateral_1d = np.asarray(lateral_1d, float)
    else:
        lateral_1d = (np.arange(n_lines) - (n_lines - 1) / 2) * float(image_data.lateral_res)
    return np.repeat(axial_1d[:n_samples, None], n_lines, axis=1), np.repeat(lateral_1d[None, :], n_samples, axis=0)


def correlation_lengths_from_image(image_data, source: str = "phantom", frames: Optional[Sequence[int]] = None,
                                   **kwargs) -> CorrelationLengths:
    """correlation_lengths() on a loaded QuantUS UltrasoundRfImage.

    source: "phantom" (reference phantom RF, the usual choice for picking window sizes) or "scan".
    frames: which frames to use (default all available; for "phantom" this indexes the frames of
        all reference files, in upload order).
    Remaining kwargs go to correlation_lengths (window_length_mm, window_width_mm, ax_start_mm,
    ax_end_mm, n_ax_locations, threshold, ...).
    """
    rf = _frames_from_image(image_data, source, frames)
    axial, lateral = positions_from_image(image_data, rf.shape[1], rf.shape[2])
    return correlation_lengths(rf, axial, lateral, **kwargs)


def _frames_from_image(image_data, source: str, frames: Optional[Sequence[int]]) -> np.ndarray:
    """(n_frames, samples, lines) RF from a loaded image: "phantom" = every frame of every
    reference file, "scan" = the scan's frames; `frames` picks a subset."""
    if source == "phantom":
        rf = image_data.get_phantom_frames()
    elif source == "scan":
        rf = image_data.rf_data
    else:
        raise ValueError("source must be 'phantom' or 'scan'")
    rf = np.asarray(rf)
    if rf.ndim == 2:
        rf = rf[None]
    if frames is not None:
        rf = rf[list(frames)]
    return rf


def plot_correlation_lengths(result: CorrelationLengths, title: str = "Correlation lengths", fig=None):
    """Mean +/- 1 SD correlation lengths (A-lines / samples) and the recommended whole-number
    window sizes vs. depth. result.to_table() has the same values in mm too."""
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=(10, 8))
    axs = fig.subplots(2, 2)
    d = result.depths_mm
    label = f"Mean ({result.n_frames} frames)"
    for ax, mean, sd, color, name in (
        (axs[0, 0], result.lat_mean_lines, result.lat_std_lines, "goldenrod", "Lateral correlation width (A-lines)"),
        (axs[0, 1], result.ax_mean_samples, result.ax_std_samples, "darkcyan", "Axial correlation length (samples)"),
    ):
        ax.plot(d, mean, color=color, marker=".", label=label)
        ax.fill_between(d, mean - sd, mean + sd, color=color, alpha=0.25, linewidth=0, label="Mean ± 1 SD")
        ax.set_ylabel(name)
        ax.legend(loc="best", fontsize=8)
    axs[0, 0].axhline(1, color="grey", linestyle=":", linewidth=1)  # one line spacing
    w = result.recommended_window()
    axs[1, 0].plot(d, result.recommended_lat_window_lines, color="goldenrod", marker="o", drawstyle="steps-mid")
    axs[1, 0].set_ylabel(f"Lateral window (A-lines) = ⌈{result.lat_window_factor:g} × width⌉")
    axs[1, 1].plot(d, result.recommended_ax_window_samples, color="darkcyan", marker="o", drawstyle="steps-mid")
    axs[1, 1].set_ylabel(f"Axial window (samples) = ⌈{result.ax_window_factor:g} × length⌉")
    for ax, n, mm, unit in ((axs[1, 0], w["lat_window_lines"], w["lat_window_mm"], "A-lines"),
                            (axs[1, 1], w["ax_window_samples"], w["ax_window_mm"], "samples")):
        if np.isfinite(n):  # the single whole-range window (largest per-depth recommendation)
            ax.axhline(n, color="black", linestyle="--", linewidth=1,
                       label=f"Whole range: {n:.0f} {unit} ({mm:.2f} mm)")
            ax.legend(loc="lower right", fontsize=8)
    for ax in axs.flat:
        ax.set_xlabel("Depth (mm)")
        ax.grid(True)
    fig.suptitle(f"{title} (threshold {result.threshold:g})")
    fig.tight_layout()
    return fig
