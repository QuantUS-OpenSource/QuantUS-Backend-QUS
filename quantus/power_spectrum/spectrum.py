"""ROI power-spectrum inspection of a sample vs. a reference phantom.

Used before segmentation / analysis config to check the shape of both spectra and pick the
analysis band: the band where BOTH the sample and the reference stay above a dB cutoff.

Ported from showPowerF2.m (VEVO_F2_QUS repo, itself from showPower/findBandwidth_Callback in
the UIUC BRL Estimator Toolbox) and verified against it to float precision. Scanner-independent:
it only needs 2D pre-scan-conversion RF frames [samples, lines] and the sampling frequency.

ROIs are (row_start, row_end, col_start, col_end) in pre-scan-conversion pixel indices,
0-based with exclusive ends (i.e. rf[row_start:row_end, col_start:col_end]).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

Roi = Tuple[int, int, int, int]

# First bin kept (MATLAB lowFreqIdxStart = 15, 1-based): skips near-DC bins.
LOW_FREQ_IDX_START = 14
# Spectra are displayed from their peak down to this level.
DISPLAY_FLOOR_DB = -15.0


@dataclass
class SpectrumBandwidth:
    f_min: float         # Hz, lower edge of the sample/reference overlap band (NaN if none)
    f_max: float         # Hz, upper edge
    f_center: float      # Hz, midpoint of [f_min, f_max]
    cutoff_db: float     # dB threshold used
    f_peak_sample: float     # Hz, sample spectrum peak
    f_peak_reference: float  # Hz, reference spectrum peak

    @property
    def is_valid(self) -> bool:
        return not np.isnan(self.f_min)


@dataclass
class SpectrumComparison:
    freqs: np.ndarray        # Hz, over the display range
    sample_db: np.ndarray    # dB, normalised to its own peak
    reference_db: np.ndarray # dB, normalised to its own peak
    bandwidth: SpectrumBandwidth
    sample_roi: Roi
    reference_roi: Roi


def default_roi(shape: Tuple[int, int]) -> Roi:
    """Central ROI (rows 1/6..5/6, lines 1/4..3/4), as showPowerF2.m used when nothing was clicked."""
    n_rows, n_cols = shape[:2]
    return (int(np.round(n_rows / 6)) - 1, int(np.round(n_rows * 5 / 6)),
            int(np.round(n_cols / 4)) - 1, int(np.round(n_cols * 3 / 4)))


def _validate_roi(roi: Roi, shape: Tuple[int, int]) -> Roi:
    r0, r1, c0, c1 = (int(v) for v in roi)
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, shape[0]), min(c1, shape[1])
    if r1 - r0 < 2 or c1 - c0 < 1:
        raise ValueError(f"ROI {roi} is empty for an RF frame of shape {shape}")
    return r0, r1, c0, c1


def roi_power_spectrum_db(rf: np.ndarray, sampling_frequency: float, roi: Optional[Roi] = None,
                          n_fft: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Line-averaged power spectrum of an RF ROI, normalised to its peak.

    Args:
        rf: 2D RF frame [samples, lines], or a stack of frames [frames, samples, lines]
            (pre-scan-conversion); for a stack the same ROI is taken from every frame and the
            power is averaged over lines and frames.
        sampling_frequency: Hz.
        roi: see module docstring; None uses default_roi.
        n_fft: zero-padded FFT length (default max(8192, next pow2 of the frame depth)).

    Returns:
        freqs [Hz] and power [dB re peak], from bin LOW_FREQ_IDX_START up to Nyquist.
    """
    rf = np.asarray(rf, dtype=np.float64)
    if rf.ndim == 2:
        rf = rf[None]
    if rf.ndim != 3:
        raise ValueError(f"Expected RF [samples, lines] or [frames, samples, lines], got shape {rf.shape}")
    frame_shape = rf.shape[1:]
    r0, r1, c0, c1 = _validate_roi(default_roi(frame_shape) if roi is None else roi, frame_shape)
    if n_fft is None:
        n_fft = max(8192, 1 << int(np.ceil(np.log2(frame_shape[0]))))
    # frames side by side -> (samples, frames * lines), then one line-average covers all frames
    roi_rf = rf[:, r0:r1, c0:c1].transpose(1, 0, 2).reshape(r1 - r0, -1)
    mag = np.abs(np.fft.fft(roi_rf, n=n_fft, axis=0))
    keep = slice(LOW_FREQ_IDX_START, n_fft // 2)
    with np.errstate(divide="ignore"):
        p = 10 * np.log10(np.mean(mag[keep] ** 2, axis=1))
    freqs = (np.arange(n_fft) / n_fft * sampling_frequency)[keep]
    return freqs, p - p.max()


def _first_below(p: np.ndarray, start: int, thresh: float) -> int:
    hits = np.nonzero(p[start:] < thresh)[0]
    return start + int(hits[0]) if hits.size else len(p) - 1


def _last_below(p: np.ndarray, stop: int, thresh: float) -> int:
    hits = np.nonzero(p[:stop + 1] < thresh)[0]
    return int(hits[-1]) if hits.size else 0


def compare_spectra(sample_rf: np.ndarray, reference_rf: np.ndarray, sampling_frequency: float,
                    sample_roi: Optional[Roi] = None, reference_roi: Optional[Roi] = None,
                    cutoff_db: float = -10.0) -> SpectrumComparison:
    """Sample vs. reference ROI spectra and their overlapping usable bandwidth.

    Args:
        sample_rf, reference_rf: 2D RF frames [samples, lines], pre-scan-conversion.
            reference_rf may also be a stack [frames, samples, lines]; its spectrum is then
            averaged over every frame (same ROI in each).
        sampling_frequency: Hz.
        sample_roi, reference_roi: see module docstring; None uses default_roi.
        cutoff_db: each spectrum's band is where it stays above this level (relative to its
            own peak); the analysis band is the intersection. Default -10 dB (the value
            showPowerF2.m actually used).
    """
    sample_rf = np.asarray(sample_rf)
    reference_rf = np.asarray(reference_rf)
    sample_roi = default_roi(sample_rf.shape) if sample_roi is None else tuple(sample_roi)
    reference_roi = default_roi(reference_rf.shape[-2:]) if reference_roi is None else tuple(reference_roi)
    n_fft = max(8192, 1 << int(np.ceil(np.log2(max(sample_rf.shape[-2], reference_rf.shape[-2])))))

    f, p1 = roi_power_spectrum_db(sample_rf, sampling_frequency, sample_roi, n_fft)
    _, p2 = roi_power_spectrum_db(reference_rf, sampling_frequency, reference_roi, n_fft)
    pk1, pk2 = int(np.argmax(p1)), int(np.argmax(p2))

    disp_lo = min(_last_below(p1, pk1, DISPLAY_FLOOR_DB), _last_below(p2, pk2, DISPLAY_FLOOR_DB))
    disp_hi = max(_first_below(p1, pk1, DISPLAY_FLOOR_DB), _first_below(p2, pk2, DISPLAY_FLOOR_DB))
    bw_lo = max(_last_below(p1, pk1, cutoff_db), _last_below(p2, pk2, cutoff_db))
    bw_hi = min(_first_below(p1, pk1, cutoff_db), _first_below(p2, pk2, cutoff_db))

    if bw_hi < bw_lo:
        warnings.warn(f"Sample and reference spectra do not overlap at the {cutoff_db:.1f} dB cutoff "
                      "-- no usable bandwidth found.")
        f_min = f_max = f_center = float("nan")
    else:
        f_min, f_max = float(f[bw_lo]), float(f[bw_hi])
        f_center = (f_min + f_max) / 2

    disp = slice(disp_lo, disp_hi + 1)
    return SpectrumComparison(
        freqs=f[disp], sample_db=p1[disp], reference_db=p2[disp],
        bandwidth=SpectrumBandwidth(f_min=f_min, f_max=f_max, f_center=f_center, cutoff_db=cutoff_db,
                                    f_peak_sample=float(f[pk1]), f_peak_reference=float(f[pk2])),
        sample_roi=sample_roi, reference_roi=reference_roi,
    )


def compare_image_spectra(image_data, frame: int = 0, sample_roi: Optional[Roi] = None,
                          reference_roi: Optional[Roi] = None, cutoff_db: float = -10.0,
                          sampling_frequency: Optional[float] = None) -> SpectrumComparison:
    """compare_spectra on a loaded UltrasoundRfImage (2D scans; `frame` picks the cine frame).

    The reference spectrum is averaged over every reference frame (image_data.get_phantom_frames()).

    sampling_frequency defaults to image_data.sampling_frequency, which loaders that know it
    (e.g. vevo_f2_rf) set; otherwise pass it explicitly.
    """
    fs = sampling_frequency if sampling_frequency is not None else getattr(image_data, "sampling_frequency", None)
    if not fs:
        raise ValueError("Sampling frequency unknown for this scan loader; pass sampling_frequency (Hz).")
    rf = image_data.rf_data if image_data.rf_data.ndim == 2 else image_data.rf_data[frame]
    ref = image_data.get_phantom_frames()
    return compare_spectra(rf, ref, fs, sample_roi, reference_roi, cutoff_db)


def apply_bandwidth_to_config(config, bandwidth: SpectrumBandwidth, set_center_frequency: bool = True):
    """Write an estimated band into an RfAnalysisConfig (analysis_freq_band, center_frequency)."""
    if not bandwidth.is_valid:
        raise ValueError("No usable bandwidth to apply (sample and reference spectra don't overlap).")
    config.analysis_freq_band = [bandwidth.f_min, bandwidth.f_max]
    if set_center_frequency:
        config.center_frequency = bandwidth.f_center
    return config


def plot_spectrum_comparison(result: SpectrumComparison, title: str = "Power Spectrum and Bandwidth", ax=None):
    """Matplotlib version of the showPowerF2.m figure. Returns the axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    f_mhz = result.freqs / 1e6
    bw = result.bandwidth
    ax.plot(f_mhz, result.sample_db, label="Sample")
    ax.plot(f_mhz, result.reference_db, label="Reference")
    if bw.is_valid:
        lo, hi, c = bw.f_min / 1e6, bw.f_max / 1e6, bw.f_center / 1e6
        ax.plot([lo, hi], [bw.cutoff_db] * 2, "r--")
        ax.plot([lo, hi], [bw.cutoff_db] * 2, "mx", markersize=15)
        ax.plot([c, c], [DISPLAY_FLOOR_DB, 0], "r")
        ax.text(lo + (hi - lo) / 5, bw.cutoff_db,
                f"$f_{{min}}$ = {lo:.3g} MHz\n$f_{{max}}$ = {hi:.3g} MHz\n$f_{{center}}$ = {c:.3g} MHz")
    ax.set_ylim(DISPLAY_FLOOR_DB, 0)
    ax.set_xlim(f_mhz[0], f_mhz[-1])
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Power [dB]")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return ax
