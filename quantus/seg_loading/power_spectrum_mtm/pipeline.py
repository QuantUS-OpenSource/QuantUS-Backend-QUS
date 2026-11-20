"""Power spectral estimation for an accepted first-pass ROI (S1Study DetermineSpectralParameters ->
ComputePowerSpectra -> DetermineBandwidth, over the whole ROI):

  1. correlation lengths of the reference phantom inside the ROI (its A-lines and depth range)
     -> final axial / lateral window (mm)
  2. NW from the ROI's first-pass band and that window (determine_spectral_parameters)
  3. window grid: centres from the ROI depth range rounded outward to 0.1 mm and the ROI's lines,
     MinOverlapAx 0.92, MinOverlapLat 0.85 (ComputePowerSpectra.m)
  4. multitaper spectra of every window: mean over the reference frames, and the sample frame
  5. band in every window, shared by sample and reference, over the ROI (roi_bandwidth)

As in the MATLAB grid, windows are centred from the ROI's first to last line / depth, so edge
windows reach half a window outside the ROI. DetermineInclusionPSERs / DetermineAttenuationPER
(which would restrict the windows) are not used: every grid window is evaluated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..correlation_length.correlation import (
    CorrelationLengths, _frames_from_image, correlation_lengths, positions_from_image, _apply_preset,
)
from .bandwidth import ROIBandwidth, bw_options, roi_bandwidth
from .multitaper import SpectrumGrid, define_power_spectrum_locations, grid_power_spectra, mtm_freq_axis
from .parameters import KAPPA, SpectralParameters, determine_spectral_parameters

MIN_OVERLAP_AX = 0.92
MIN_OVERLAP_LAT = 0.85


@dataclass
class PowerSpectralEstimate:
    correlation: CorrelationLengths
    ax_window_mm: float
    lat_window_mm: float
    parameters: SpectralParameters
    grid: SpectrumGrid
    start_ax_mm: float
    end_ax_mm: float
    lines: tuple                         # ROI A-lines (0-based, inclusive) = window-centre lines
    freq: np.ndarray
    reference_spectra: np.ndarray        # (n_ax, n_lat, n_freq), mean over the reference frames
    sample_spectra: np.ndarray
    bandwidth: ROIBandwidth
    depths_mm: np.ndarray                # (n_ax,) window-centre depth on the centre line
    timings_s: dict = field(default_factory=dict)

    def summary(self) -> str:
        g = self.grid
        return "\n".join([
            "ROI " + self.correlation.describe_window(),
            self.parameters.describe(),
            f"Grid: depths {self.start_ax_mm:.1f} - {self.end_ax_mm:.1f} mm, lines {self.lines[0]}-{self.lines[1]}: "
            f"{g.shape[0]} x {g.shape[1]} windows of {int(g.ax_lengths[0])} samples x "
            f"{int(g.lat_widths.min())}-{int(g.lat_widths.max())} lines (overlap ax {g.ax_overlap.min():.2f}, "
            f"lat {g.lat_overlap.min():.2f})",
            self.bandwidth.describe(),
        ])


def power_spectral_estimation(rf_reference: np.ndarray, rf_sample: np.ndarray, axial_mm: np.ndarray,
                              lateral_mm: np.ndarray, fs_mhz: float, roi_lines: Sequence[int],
                              roi_depth_range_mm: Sequence[float], roi_band_mhz: Sequence[float],
                              correlation_kwargs: Optional[dict] = None, kappa: float = KAPPA,
                              avoid_freq: Sequence[float] = (), min_overlap_ax: float = MIN_OVERLAP_AX,
                              min_overlap_lat: float = MIN_OVERLAP_LAT, nfft: int = 1024,
                              verbose: bool = False, progress=None,
                              center_freq_mhz: Optional[float] = None) -> PowerSpectralEstimate:
    """Steps 1-5 above on arrays. rf_reference: (n_frames, samples, lines); rf_sample: one frame
    (samples, lines). roi_lines: (first, last) A-lines, 0-based inclusive. roi_depth_range_mm:
    ROI top / bottom on the centre line. roi_band_mhz: the ROI's first-pass band (fLow, fHigh).
    correlation_kwargs: correlation_lengths settings (window_length_mm, window_width_mm,
    threshold, ...; preset values via _apply_preset). progress: optional callable(fraction 0-1,
    message), e.g. for a GUI progress bar."""
    def report(frac, msg):
        if progress is not None:
            progress(float(frac), msg)

    t = {}
    report(0.0, "Correlation lengths inside the ROI")
    t0 = time.time()
    rf_ref = np.asarray(rf_reference, dtype=float)
    if rf_ref.ndim == 2:
        rf_ref = rf_ref[None]
    l1, l2 = (int(v) for v in roi_lines)
    d0, d1 = sorted(float(v) for v in roi_depth_range_mm)

    ckw = dict(correlation_kwargs or {})
    ckw.pop("lateral_fraction", None)
    ckw.pop("depth_range_mm", None)
    corr = correlation_lengths(rf_ref, axial_mm, lateral_mm, depth_range_mm=(d0, d1), line_range=(l1, l2), **ckw)
    w = corr.recommended_window()
    ax_mm, lat_mm = w["ax_window_mm"], w["lat_window_mm"]
    if not (np.isfinite(ax_mm) and np.isfinite(lat_mm)):
        raise ValueError("No correlation-length window inside the ROI (ROI smaller than the measurement "
                         "window, or the correlation never drops below the threshold).")
    t["correlation"] = time.time() - t0

    params = determine_spectral_parameters(roi_band_mhz, ax_mm, axial_mm, fs_mhz, kappa=kappa,
                                           avoid_freq=avoid_freq)
    if verbose:
        print(corr.describe_window())
        print(params.describe())

    start_ax = float(np.floor(d0 * 10 + 1e-9) / 10)
    end_ax = float(np.ceil(d1 * 10 - 1e-9) / 10)
    grid = define_power_spectrum_locations(axial_mm, lateral_mm, start_ax, end_ax, l1 + 1, l2 + 1, ax_mm, lat_mm,
                                           params.nw, min_overlap_ax=min_overlap_ax, min_overlap_lat=min_overlap_lat,
                                           verbose=verbose)
    # As in the MATLAB: NW is chosen for N = round(L/dz) made odd (DetermineSpectralParameters),
    # the grid windows are round(L/dz) + 1 samples (DefinePowerSpectrumLocations); they differ by
    # one sample when round(L/dz) is odd, which changes W by < 1% and NW (0.5 steps) rarely.
    if verbose and int(grid.ax_lengths[0]) != params.n:
        print(f"  note: grid windows are {int(grid.ax_lengths[0])} samples, NW chosen for N = {params.n}")

    # progress: spectra take nearly all the time, one share per frame (reference frames + sample)
    n_ref = rf_ref.shape[0]
    share = 0.97 / (n_ref + 1)
    t0 = time.time()
    ref_ps = grid_power_spectra(rf_ref, grid, fs_mhz, nfft, progress=lambda i, n: report(
        0.02 + share * n_ref * i / n, f"Reference spectra ({grid.shape[0]} x {grid.shape[1]} windows, {n_ref} frames)"))
    t["reference_spectra"] = time.time() - t0
    t0 = time.time()
    smp_ps = grid_power_spectra(np.asarray(rf_sample, dtype=float), grid, fs_mhz, nfft, progress=lambda i, n: report(
        0.02 + share * (n_ref + i / n), "Sample spectra"))
    t["sample_spectra"] = time.time() - t0
    report(0.99, "Bandwidth in every window")
    freq = mtm_freq_axis(fs_mhz, nfft)

    t0 = time.time()
    bw = roi_bandwidth(smp_ps, ref_ps, freq, **bw_options(params.avoid_freq, params.w_mhz, center_freq_mhz))
    t["bandwidth"] = time.time() - t0
    return PowerSpectralEstimate(
        correlation=corr, ax_window_mm=ax_mm, lat_window_mm=lat_mm, parameters=params, grid=grid,
        start_ax_mm=start_ax, end_ax_mm=end_ax, lines=(l1, l2), freq=freq, reference_spectra=ref_ps,
        sample_spectra=smp_ps, bandwidth=bw, depths_mm=grid.depths_mm(np.asarray(axial_mm, float)), timings_s=t)


def power_spectral_estimation_from_image(image, roi, roi_band_mhz: Optional[Sequence[float]] = None,
                                         preset: Optional[str] = None, correlation_kwargs: Optional[dict] = None,
                                         sample_frame: int = 0, use_image_sampling_frequency: bool = True,
                                         **kwargs) -> PowerSpectralEstimate:
    """power_spectral_estimation on a loaded QuantUS UltrasoundRfImage and a first-pass SectorROI
    (correlation_length.roi). roi_band_mhz defaults to the ROI's first-pass band (roi.band).
    preset: PROBE_PRESETS name for the correlation measurement window and threshold."""
    if roi_band_mhz is None:
        if roi.band is None:
            raise ValueError("The ROI has no first-pass band; pass roi_band_mhz")
        roi_band_mhz = (roi.band.f_low_roi, roi.band.f_high_roi)
    ref = _frames_from_image(image, "phantom", None)
    smp = _frames_from_image(image, "scan", [sample_frame])[0]
    axial, lateral = positions_from_image(image, ref.shape[1], ref.shape[2])
    if smp.shape != ref.shape[1:]:
        raise ValueError(f"Sample {smp.shape} and reference {ref.shape[1:]} RF sizes differ")
    fs = getattr(image, "sampling_frequency", None)
    if use_image_sampling_frequency and fs:
        fs_mhz = float(fs) / 1e6
    else:
        dz = float(axial[2, axial.shape[1] // 2] - axial[1, axial.shape[1] // 2])
        fs_mhz = 1540.0 / (2 * dz * 1e-3) / 1e6
    ckw = _apply_preset(preset, dict(correlation_kwargs or {}))
    from ..correlation_length.first_pass import probe_center_frequency_mhz
    kwargs.setdefault("center_freq_mhz", probe_center_frequency_mhz(image))
    return power_spectral_estimation(ref, smp, axial, lateral, fs_mhz, roi.lines, roi.depth_range_mm,
                                     roi_band_mhz, correlation_kwargs=ckw, **kwargs)
