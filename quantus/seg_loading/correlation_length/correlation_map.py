"""2D correlation-length map: the CalcAxLatCorrelationCoeffCurves measurement repeated on a grid
of window positions across the whole image (not just the centre A-line), so the QUS window can be
chosen from the largest lengths anywhere in the image.

Window centres: rows as in correlation_lengths() (ax_start_mm / ax_end_mm / n_ax_locations, snapped
on the centre line); columns every `lat_step_lines` A-lines, symmetric about (and always including)
the centre line, so the map's centre column equals correlation_lengths(). At each row the lateral
window size comes from the line spacing there; a column is measured only where the full window
fits in the image (NaN elsewhere). Each window's depth and lateral position are its centre
sample's physical position, so curvilinear maps come out fan-shaped.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .correlation import (
    _apply_preset, _axial_centres, _ceil_count, _central_region, _round_mm, _distance_mm, _measure_window, _mround,
    _odd_count, _prepare, _sample_spacings_mm, positions_from_image, _frames_from_image,
)


@dataclass
class CorrelationLengthMap:
    depth_mm: np.ndarray        # (n_rows, n_cols) depth of each window centre
    lateral_mm: np.ndarray      # (n_rows, n_cols) lateral position of each window centre
    rows: np.ndarray            # (n_rows,) 0-based centre sample index
    cols: np.ndarray            # (n_cols,) 0-based centre A-line index
    ax_corr_mm: np.ndarray      # (n_rows, n_cols, n_frames) full axial correlation length (pulse-echo: 2x one-sided)
    lat_corr_mm: np.ndarray     # (n_rows, n_cols, n_frames) lateral correlation width (one-sided)
    ax_spacing_mm: np.ndarray   # (n_rows, n_cols) along-line sample spacing
    lat_spacing_mm: np.ndarray  # (n_rows, n_cols) line-to-line spacing
    window_samples: int         # axial samples in the measurement window
    window_lines: np.ndarray    # (n_rows,) A-lines in the measurement window
    threshold: float
    window_length_mm: float
    window_width_mm: float
    ax_window_factor: float = 10.0
    lat_window_factor: float = 5.0
    lateral_fraction: Optional[float] = None  # windows limited to the central fraction of the A-lines

    @property
    def n_frames(self) -> int:
        return self.ax_corr_mm.shape[-1]

    @staticmethod
    def _mean(x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells -> NaN
            return np.nanmean(x, axis=-1)

    def _std(self, x):
        if self.n_frames < 2:
            return np.where(np.isfinite(self._mean(x)), 0.0, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanstd(x, axis=-1, ddof=1)

    @property
    def ax_mean_samples(self) -> np.ndarray:
        return self._mean(self.ax_corr_mm / self.ax_spacing_mm[..., None])

    @property
    def ax_std_samples(self) -> np.ndarray:
        return self._std(self.ax_corr_mm / self.ax_spacing_mm[..., None])

    @property
    def lat_mean_lines(self) -> np.ndarray:
        return self._mean(self.lat_corr_mm / self.lat_spacing_mm[..., None])

    @property
    def lat_std_lines(self) -> np.ndarray:
        return self._std(self.lat_corr_mm / self.lat_spacing_mm[..., None])

    @property
    def ax_mean_mm(self) -> np.ndarray:
        return self._mean(self.ax_corr_mm)

    @property
    def lat_mean_mm(self) -> np.ndarray:
        return self._mean(self.lat_corr_mm)

    @property
    def recommended_ax_window_samples(self) -> np.ndarray:
        """(n_rows, n_cols) ceil(ax_window_factor x mean axial length) in samples."""
        return _ceil_count(self.ax_window_factor * self.ax_mean_samples)

    @property
    def recommended_lat_window_lines(self) -> np.ndarray:
        """(n_rows, n_cols) ceil(lat_window_factor x mean lateral width) in A-lines."""
        return _ceil_count(self.lat_window_factor * self.lat_mean_lines)

    def region_mask(self, depth_range_mm=None, lateral_range_mm=None) -> np.ndarray:
        """(n_rows, n_cols) True for windows whose centre lies in the given (min, max) ranges (None = no limit)."""
        mask = np.ones(self.depth_mm.shape, dtype=bool)
        for values, rng in ((self.depth_mm, depth_range_mm), (self.lateral_mm, lateral_range_mm)):
            if rng is not None:
                lo, hi = rng
                mask &= (values >= (-np.inf if lo is None else lo)) & (values <= (np.inf if hi is None else hi))
        return mask

    def recommended_window(self, depth_range_mm=None, lateral_range_mm=None) -> dict:
        """One window for the image (or a region of it): the largest per-window recommendation,
        with where it occurs (depth and lateral position of that measurement window).

        depth_range_mm / lateral_range_mm: (min, max) of the window centres to consider, e.g. the
        speckle region, to leave out walls, near-field reverberation or targets; None = everything.
        """
        region = self.region_mask(depth_range_mm, lateral_range_mm)

        def largest(counts, length_mm, factor):
            # the largest whole-number window, its mm size (factor x mean length, nearest 0.1 mm)
            # and where it is
            counts = np.where(region, counts, np.nan)
            if not np.any(np.isfinite(counts)):
                return np.nan, np.nan, np.nan, np.nan
            i = np.unravel_index(int(np.nanargmax(counts)), counts.shape)
            return (float(counts[i]), float(_round_mm(factor * length_mm[i])),
                    float(self.depth_mm[i]), float(self.lateral_mm[i]))

        ax = largest(self.recommended_ax_window_samples, self.ax_mean_mm, self.ax_window_factor)
        lat = largest(self.recommended_lat_window_lines, self.lat_mean_mm, self.lat_window_factor)
        measured = region & (np.isfinite(self.ax_mean_samples) | np.isfinite(self.lat_mean_lines))
        if not measured.any():
            raise ValueError("No measured windows in the requested region.")
        return {
            "ax_window_samples": ax[0], "ax_window_mm": ax[1], "ax_at_depth_mm": ax[2], "ax_at_lateral_mm": ax[3],
            "lat_window_lines": lat[0], "lat_window_mm": lat[1], "lat_at_depth_mm": lat[2], "lat_at_lateral_mm": lat[3],
            "depth_range_mm": (float(np.nanmin(self.depth_mm[measured])), float(np.nanmax(self.depth_mm[measured]))),
            "lateral_range_mm": (float(np.nanmin(self.lateral_mm[measured])), float(np.nanmax(self.lateral_mm[measured]))),
            "ax_window_factor": self.ax_window_factor, "lat_window_factor": self.lat_window_factor,
        }

    def describe_window(self, depth_range_mm=None, lateral_range_mm=None) -> str:
        w = self.recommended_window(depth_range_mm, lateral_range_mm)
        (d0, d1), (x0, x1) = w["depth_range_mm"], w["lateral_range_mm"]
        where = "the whole map" if depth_range_mm is None and lateral_range_mm is None else "the region"
        return (f"Window for {where} (depth {d0:.1f}-{d1:.1f} mm, lateral {x0:.1f} to {x1:.1f} mm, "
                f"{self.n_frames} frames, threshold {self.threshold:g}): "
                f"axial {w['ax_window_samples']:.0f} samples ({w['ax_window_mm']:.1f} mm, "
                f"{w['ax_window_factor']:g}x length, max at depth {w['ax_at_depth_mm']:.1f} mm / "
                f"lateral {w['ax_at_lateral_mm']:.1f} mm) x "
                f"lateral {w['lat_window_lines']:.0f} A-lines ({w['lat_window_mm']:.1f} mm, "
                f"{w['lat_window_factor']:g}x width, max at depth {w['lat_at_depth_mm']:.1f} mm / "
                f"lateral {w['lat_at_lateral_mm']:.1f} mm)")

    def to_table(self):
        """One row per measured window (pandas DataFrame)."""
        import pandas as pd

        rr, cc = np.meshgrid(np.arange(len(self.rows)), np.arange(len(self.cols)), indexing="ij")
        keep = np.isfinite(self.ax_mean_samples) | np.isfinite(self.lat_mean_lines)
        cols = {
            "depth_mm": self.depth_mm, "lateral_mm": self.lateral_mm,
            "row": self.rows[rr], "line": self.cols[cc],
            "ax_length_samples": self.ax_mean_samples, "ax_length_sd_samples": self.ax_std_samples,
            "lat_width_lines": self.lat_mean_lines, "lat_width_sd_lines": self.lat_std_lines,
            f"ax_window_samples ({self.ax_window_factor:g}x)": self.recommended_ax_window_samples,
            f"lat_window_lines ({self.lat_window_factor:g}x)": self.recommended_lat_window_lines,
        }
        return pd.DataFrame({k: np.asarray(v)[keep] for k, v in cols.items()})


def correlation_length_map(rf_frames: np.ndarray, axial_position_mm: np.ndarray, lateral_position_mm: np.ndarray,
                           window_length_mm: float, window_width_mm: float,
                           ax_start_mm: Optional[float] = None, ax_end_mm: Optional[float] = None,
                           n_ax_locations: Optional[int] = None, lat_step_lines: Optional[int] = None,
                           threshold: float = 0.5, ax_window_factor: float = 10.0,
                           lat_window_factor: float = 5.0,
                           lateral_fraction: Optional[float] = None, depth_range_mm=None) -> CorrelationLengthMap:
    """Correlation lengths on a grid of window positions across the image.

    Same inputs as correlation_lengths(), plus:
        lat_step_lines: A-lines between window centres laterally; None -> half the narrowest
            measurement window (in lines) over the depths, i.e. a centre every half window.
        lateral_fraction: only measure windows lying entirely inside the central fraction of
            the A-lines (e.g. 0.5); None -> anywhere the window fits in the image. Unlike
            correlation_lengths() (which always keeps the centre window), a region narrower
            than one window leaves the map empty.
        depth_range_mm: (top, bottom) window-edge depth limits on the centre line, as in
            correlation_lengths() (rows are shared by all columns).
    """
    env, axial, lateral = _prepare(rf_frames, axial_position_mm, lateral_position_mm)
    n_frames, n_samples, n_lines = env.shape
    mid0 = _mround(n_lines / 2) - 1

    d_ax_centre = _distance_mm(axial, lateral, (2, mid0), (1, mid0))
    win_ax = _odd_count(window_length_mm, d_ax_centre)
    half_ax = (win_ax - 1) // 2
    rows = _axial_centres(axial[:, mid0], n_samples, half_ax, window_length_mm,
                          ax_start_mm, ax_end_mm, n_ax_locations, depth_range_mm) - 1

    # Lateral window size per row (line spacing at that depth on the centre line).
    win_lines = np.array([_odd_count(window_width_mm, _distance_mm(axial, lateral, (r, mid0), (r, mid0 - 1)))
                          for r in rows])
    half_lat = (win_lines - 1) // 2
    if np.all(2 * half_lat + 1 > n_lines):
        raise ValueError(f"A {window_width_mm:g} mm window needs more A-lines than the {n_lines} available "
                         "at every depth; reduce window_width_mm.")
    step = lat_step_lines if lat_step_lines is not None else max(1, int(np.min(half_lat)))
    if step < 1:
        raise ValueError("lat_step_lines must be >= 1")
    first, last = _central_region(n_lines, mid0, lateral_fraction)
    cols = np.concatenate([np.arange(mid0, first - 1, -step)[::-1], np.arange(mid0 + step, last + 1, step)])

    shape = (len(rows), len(cols))
    ax_corr = np.full(shape + (n_frames,), np.nan)
    lat_corr = np.full(shape + (n_frames,), np.nan)
    ax_spacing = np.full(shape, np.nan)
    lat_spacing = np.full(shape, np.nan)
    for i, (row, h) in enumerate(zip(rows, half_lat)):
        for j, col in enumerate(cols):
            ax_spacing[i, j], lat_spacing[i, j] = _sample_spacings_mm(axial, lateral, row, col)
            if col - h < first or col + h > last:
                continue  # full window doesn't fit in the image / central region at this position
            lat_corr[i, j], ax_half, _, _ = _measure_window(
                env, row - half_ax, row + half_ax + 1, col - h, col + h + 1,
                ax_spacing[i, j], lat_spacing[i, j], threshold)
            ax_corr[i, j] = 2 * ax_half  # pulse-echo full width

    return CorrelationLengthMap(
        depth_mm=axial[np.ix_(rows, cols)], lateral_mm=lateral[np.ix_(rows, cols)], rows=rows, cols=cols,
        ax_corr_mm=ax_corr, lat_corr_mm=lat_corr, ax_spacing_mm=ax_spacing, lat_spacing_mm=lat_spacing,
        window_samples=win_ax, window_lines=win_lines, threshold=threshold,
        window_length_mm=window_length_mm, window_width_mm=window_width_mm,
        ax_window_factor=ax_window_factor, lat_window_factor=lat_window_factor,
        lateral_fraction=lateral_fraction,
    )


def correlation_length_map_from_image(image_data, source: str = "phantom", frames: Optional[Sequence[int]] = None,
                                      preset: Optional[str] = None, **kwargs) -> CorrelationLengthMap:
    """correlation_length_map() on a loaded QuantUS UltrasoundRfImage (source / frames / preset as
    in correlation_lengths_from_image)."""
    kwargs = _apply_preset(preset, kwargs)
    rf = _frames_from_image(image_data, source, frames)
    axial, lateral = positions_from_image(image_data, rf.shape[1], rf.shape[2])
    return correlation_length_map(rf, axial, lateral, **kwargs)


def plot_correlation_length_map(result: CorrelationLengthMap, title: str = "Correlation-length map", fig=None,
                                depth_range_mm=None, lateral_range_mm=None):
    """Colour maps (depth vs lateral position) of the axial length (samples), lateral width
    (A-lines) and the recommended windows. The cell setting the window (over the whole map, or
    over depth_range_mm / lateral_range_mm, drawn as a dashed box) is circled."""
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=(12, 9))
    axs = fig.subplots(2, 2)
    w = result.recommended_window(depth_range_mm, lateral_range_mm)
    region = result.region_mask(depth_range_mm, lateral_range_mm)
    boxed = depth_range_mm is not None or lateral_range_mm is not None
    panels = (
        (axs[0, 0], result.ax_mean_samples, "Axial correlation length (samples)", None, "viridis"),
        (axs[0, 1], result.lat_mean_lines, "Lateral correlation width (A-lines)", None, "magma"),
        (axs[1, 0], result.recommended_ax_window_samples,
         f"Axial window (samples) = ⌈{result.ax_window_factor:g} × length⌉", (w["ax_at_lateral_mm"], w["ax_at_depth_mm"]), "viridis"),
        (axs[1, 1], result.recommended_lat_window_lines,
         f"Lateral window (A-lines) = ⌈{result.lat_window_factor:g} × width⌉", (w["lat_at_lateral_mm"], w["lat_at_depth_mm"]), "magma"),
    )
    for ax, values, name, peak, cmap in panels:
        data = np.ma.masked_invalid(values)
        mesh = ax.pcolormesh(result.lateral_mm, result.depth_mm, data, shading="nearest", cmap=cmap)
        fig.colorbar(mesh, ax=ax)
        if boxed:
            d0, d1 = depth_range_mm if depth_range_mm is not None else (None, None)
            x0, x1 = lateral_range_mm if lateral_range_mm is not None else (None, None)
            d0 = np.nanmin(result.depth_mm) if d0 is None else d0
            d1 = np.nanmax(result.depth_mm) if d1 is None else d1
            x0 = np.nanmin(result.lateral_mm) if x0 is None else x0
            x1 = np.nanmax(result.lateral_mm) if x1 is None else x1
            ax.plot([x0, x1, x1, x0, x0], [d0, d0, d1, d1, d0], color="white", linestyle="--", linewidth=1.2)
        if peak is not None and np.all(np.isfinite(peak)):
            ax.plot(*peak, marker="o", markersize=14, markerfacecolor="none", markeredgecolor="red", markeredgewidth=2)
            label = "region" if boxed else "whole image"
            ax.set_title(f"{name}\n{label}: {np.nanmax(np.where(region, values, np.nan)):.0f}", fontsize=10)
        else:
            ax.set_title(name, fontsize=10)
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Depth (mm)")
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
    check = "" if result.lateral_fraction is None else f"; central {result.lateral_fraction:g} of the image"
    fig.suptitle(f"{title} ({result.n_frames} frames, threshold {result.threshold:g}; "
                 f"measurement window {result.window_length_mm:g} x {result.window_width_mm:g} mm{check})")
    fig.tight_layout()
    return fig
