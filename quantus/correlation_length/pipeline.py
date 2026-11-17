"""Window sizes and first-pass power check for one sample + reference, probe-independent:

  1. correlation_lengths_from_image on the reference phantom -> recommended window (mm)
  2. calc_first_pass on the reference (all captures) and on the sample, with that window as the
     first-pass axial window and lateral block width
  3. intersect_first_pass -> blocks usable in both

Every step's settings can be overridden; probe presets (PROBE_PRESETS) only fill in defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .correlation import CorrelationLengths, correlation_lengths_from_image
from .first_pass import FirstPass, FirstPassIntersect, calc_first_pass_from_image, intersect_first_pass


@dataclass
class WindowsAndFirstPass:
    correlation: CorrelationLengths
    ax_window_mm: float
    lat_window_mm: float
    reference: FirstPass
    sample: FirstPass
    intersect: FirstPassIntersect

    def summary(self) -> str:
        return "\n".join([
            self.correlation.describe_window(),
            f"First pass: {self.ax_window_mm:.1f} mm axial window, {self.lat_window_mm:.1f} mm blocks "
            f"({self.sample.block_lines} lines), {self.sample.depths.size} depths x "
            f"{self.sample.block_x.shape[1]} blocks, fs {self.sample.fs_mhz:g} MHz; usable in reference "
            f"{self.reference.usable_2d.mean():.0%}, sample {self.sample.usable_2d.mean():.0%}, "
            f"both {self.intersect.usable_both_2d.mean():.0%}",
        ])


def windows_and_first_pass(image, preset: Optional[str] = None, correlation_kwargs: Optional[dict] = None,
                           ax_step_mm: Optional[float] = None, use_image_sampling_frequency: bool = True,
                           avoid_freq: Sequence[float] = (), search_above: Optional[float] = None,
                           on_nyquist: str = "error") -> WindowsAndFirstPass:
    """Correlation-length windows -> first pass on reference and sample -> blocks usable in both.

    preset / correlation_kwargs: correlation-length settings (PROBE_PRESETS name and overrides).
    ax_step_mm: first-pass depth step (default 0.2 x the axial window).
    use_image_sampling_frequency: True -> first-pass frequency axis from the image's sampling
        frequency; False -> from the axial spacing with c = 1540 m/s, as the MATLAB.
    on_nyquist: passed to calc_first_pass ("error" stops if the signal reaches Nyquist).
    """
    corr = correlation_lengths_from_image(image, source="phantom", preset=preset, **dict(correlation_kwargs or {}))
    w = corr.recommended_window()
    ax_mm, lat_mm = w["ax_window_mm"], w["lat_window_mm"]
    if not (np.isfinite(ax_mm) and np.isfinite(lat_mm)):
        raise ValueError("No recommended window from the correlation lengths")
    fs = getattr(image, "sampling_frequency", None)
    fp_kw = dict(window_length_mm=ax_mm, block_width_mm=lat_mm,
                 ax_step_mm=0.2 * ax_mm if ax_step_mm is None else ax_step_mm,
                 avoid_freq=avoid_freq, search_above=search_above, on_nyquist=on_nyquist,
                 sampling_frequency_mhz=float(fs) / 1e6 if (use_image_sampling_frequency and fs) else None)
    ref = calc_first_pass_from_image(image, source="phantom", **fp_kw)
    # same lateral blocks as the reference, even if the sample's imaging depth differs (sector probes)
    smp = calc_first_pass_from_image(image, source="scan", block_lines=ref.block_lines, **fp_kw)
    return WindowsAndFirstPass(correlation=corr, ax_window_mm=ax_mm, lat_window_mm=lat_mm, reference=ref,
                               sample=smp, intersect=intersect_first_pass(smp, ref))


def plot_first_pass(result: WindowsAndFirstPass, image=None, title: str = "", fig=None):
    """Sample B-mode with blocks usable in both (green) / not (red), and the sample / reference
    spectral SNR maps with the 15 dB usable contour."""
    import matplotlib.pyplot as plt

    from .correlation import _frames_from_image, envelope_frames, positions_from_image

    if fig is None:
        fig = plt.figure(figsize=(16, 5.5))
    axs = fig.subplots(1, 3)
    s, inter = result.sample, result.intersect
    x, z = s.block_x, s.block_z
    if image is not None:
        rf = _frames_from_image(image, "scan", [0])
        ax_p, lat_p = positions_from_image(image, rf.shape[1], rf.shape[2])
        env = envelope_frames(rf)[0]
        axs[0].pcolormesh(lat_p, ax_p, 20 * np.log10(env / env.max() + 1e-12), cmap="gray", vmin=-60, vmax=0,
                          shading="nearest", rasterized=True)
    axs[0].pcolormesh(x, z, inter.usable_both_2d.astype(float), cmap="RdYlGn", vmin=0, vmax=1, alpha=0.3,
                      shading="nearest")
    axs[0].set_title("Sample: usable in both (green) / not (red)", fontsize=10)
    snr_max = np.nanmax([np.nanmax(s.snr_2d), np.nanmax(result.reference.snr_2d)])
    for ax, fp, name in ((axs[1], s, "Sample"), (axs[2], result.reference, "Reference")):
        m = ax.pcolormesh(x, z, fp.snr_2d, cmap="turbo", vmin=0, vmax=snr_max, shading="nearest")
        ax.contour(x, z, fp.snr_2d, [fp.snr_threshold_db], colors="w", linewidths=1.2)
        fig.colorbar(m, ax=ax, label="Peak - noise floor (dB)")
        ax.set_title(f"{name} spectral SNR, {fp.snr_threshold_db:g} dB contour", fontsize=10)
    for ax in axs:
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Depth (mm)")
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
    fig.suptitle(f"{title} {result.ax_window_mm:.1f} x {result.lat_window_mm:.1f} mm windows".strip())
    fig.tight_layout()
    return fig
