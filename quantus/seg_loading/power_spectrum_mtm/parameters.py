"""Multitaper settings from the window and the band (S1Study CheckMTMParameters.m and the NW rules
of DetermineSpectralParameters.m).

    T = N / fs (us)          W = NW / T (MHz)          K = max(1, floor(2 NW - 1))
    NW_recommended = kappa * B * T                     B = fHigh - fLow

NW to use: NW_recommended to the nearest 0.5, capped (down, to a 0.5 step) if a known spectral
feature (avoid_freq) would leak into [fLow - W, fHigh + W], flagged if below 1.5 (one taper).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .multitaper import _mround, n_tapers

KAPPA = 0.2                  # DetermineSpectralParameters (CheckMTMParameters' own default is 0.15)
SPEED_OF_SOUND = 1540.0


def round_band_inward(f_low: float, f_high: float, tol: float = 1e-9) -> tuple:
    """Band rounded inward to 0.1 MHz: fLow up, fHigh down (tol stops 1.70 -> 1.8 from round-off)."""
    return float(np.ceil(f_low * 10 - tol) / 10), float(np.floor(f_high * 10 + tol) / 10)


def window_samples(ax_window_mm: float, delta_ax_mm: float) -> int:
    """N = round(AxPERLength / deltaAx), made odd (DetermineSpectralParameters). Equal to the
    window length define_power_spectrum_locations uses (round(L / dz) + 1 when that is odd)."""
    n = _mround(ax_window_mm / delta_ax_mm)
    return n + 1 if n % 2 == 0 else n


def centre_line_delta_ax(axial_position_mm: np.ndarray) -> float:
    """Sample spacing on the centre line, AxialPosition(3, MidLat) - AxialPosition(2, MidLat)."""
    axial = np.asarray(axial_position_mm, dtype=float)
    mid0 = _mround(axial.shape[1] / 2) - 1
    return float(axial[2, mid0] - axial[1, mid0])


@dataclass
class MTMCheck:
    """CheckMTMParameters output (the per-NW table as arrays)."""
    n: int
    fs_mhz: float
    t_us: float
    l_mm: float
    band_mhz: tuple
    b_mhz: float
    rayleigh_mhz: float
    kappa: float
    w_target_mhz: float
    nw_recommended: float
    n_for_nw2: int
    l_mm_for_nw2: float
    w_max_before_leak_mhz: float
    nw_max_before_leak: float
    nw: np.ndarray                      # NW values evaluated
    k: np.ndarray
    w_mhz: np.ndarray
    w_over_b: np.ndarray
    eff_low_mhz: np.ndarray
    eff_high_mhz: np.ndarray
    status: list

    def report(self) -> str:
        lines = [f"Window: N = {self.n} samples at fs = {self.fs_mhz:g} MHz -> T = {self.t_us:.2f} us, "
                 f"L = {self.l_mm:.2f} mm",
                 f"Band: {self.band_mhz[0]:.2f} - {self.band_mhz[1]:.2f} MHz (B = {self.b_mhz:.2f} MHz)",
                 f"Ideal W = kappa*B = {self.kappa:g}*{self.b_mhz:.2f} = {self.w_target_mhz:.2f} MHz -> "
                 f"NW {self.nw_recommended:.2f}"]
        for nw, k, w, wb, s in zip(self.nw, self.k, self.w_mhz, self.w_over_b, self.status):
            lines.append(f"  NW {nw:g}: K {k}, W {w:.2f} MHz ({wb:.2f} B)  {s}")
        if np.isfinite(self.w_max_before_leak_mhz):
            lines.append(f"  keep W < {self.w_max_before_leak_mhz:.2f} MHz (NW < {self.nw_max_before_leak:.2f}) "
                         "to keep the avoided feature out of the band")
        return "\n".join(lines)


def check_mtm_parameters(n: int, band_mhz, fs_mhz: float, nw: Sequence[float] = (2, 3, 4),
                         kappa: float = 0.15, c: float = SPEED_OF_SOUND,
                         avoid_freq: Sequence[float] = ()) -> MTMCheck:
    """Port of CheckMTMParameters(N, BW, fs, 'NW', nw, 'Kappa', kappa, 'c', c, 'AvoidFreq', ...).
    band_mhz: (fLow, fHigh). Taper eigenvalues are not computed (not used for the choice)."""
    if n <= 1:
        raise ValueError("N must be > 1")
    f_low, f_high = float(min(band_mhz)), float(max(band_mhz))
    b = f_high - f_low
    t_us = n / fs_mhz
    l_mm = c * (t_us * 1e-6) / 2 * 1e3
    nws = np.atleast_1d(np.asarray(nw, dtype=float))
    w = nws / t_us
    k = np.maximum(1, np.floor(2 * nws - 1)).astype(int)
    eff_low, eff_high = f_low - w, f_high + w
    avoid = np.atleast_1d(np.asarray(avoid_freq, dtype=float)).ravel()
    leak_ok = np.array([np.all((avoid < lo) | (avoid > hi)) for lo, hi in zip(eff_low, eff_high)])
    status = []
    for i in range(nws.size):
        if nws[i] >= n / 2:
            status.append("INVALID (NW >= N/2)")
        elif not leak_ok[i]:
            status.append("LEAKS feature")
        elif w[i] / b > 0.5:
            status.append("TOO SMOOTH")
        elif w[i] / b > 2 * kappa:
            status.append("marginal")
        else:
            status.append("OK")
    w_target = kappa * b
    n_for_nw2 = int(np.ceil(2 * fs_mhz / w_target))
    if avoid.size:
        w_leak = float(np.min(np.abs(np.concatenate([f_low - avoid, avoid - f_high]))))
        nw_leak = w_leak * t_us
    else:
        w_leak = nw_leak = np.nan
    return MTMCheck(n=int(n), fs_mhz=float(fs_mhz), t_us=t_us, l_mm=l_mm, band_mhz=(f_low, f_high), b_mhz=b,
                    rayleigh_mhz=fs_mhz / n, kappa=kappa, w_target_mhz=w_target, nw_recommended=w_target * t_us,
                    n_for_nw2=n_for_nw2, l_mm_for_nw2=c * (n_for_nw2 / fs_mhz * 1e-6) / 2 * 1e3,
                    w_max_before_leak_mhz=w_leak, nw_max_before_leak=nw_leak, nw=nws, k=k, w_mhz=w,
                    w_over_b=w / b, eff_low_mhz=eff_low, eff_high_mhz=eff_high, status=status)


@dataclass
class SpectralParameters:
    """DetermineSpectralParameters result: the band (rounded inward), window and multitaper settings."""
    band_mhz: tuple                     # ROI band rounded inward to 0.1 MHz
    roi_band_mhz: tuple                 # ROI band as given
    n: int                              # window samples (odd)
    fs_mhz: float
    ax_window_mm: float
    kappa: float
    avoid_freq: tuple
    nw: float
    k: int
    w_mhz: float
    note: str
    check: MTMCheck = field(repr=False)

    def describe(self) -> str:
        b = self.band_mhz[1] - self.band_mhz[0]
        s = (f"Band {self.roi_band_mhz[0]:.2f} - {self.roi_band_mhz[1]:.2f} MHz -> {self.band_mhz[0]:.1f} - "
             f"{self.band_mhz[1]:.1f} MHz; window {self.ax_window_mm:g} mm = {self.n} samples; NW {self.nw:g} "
             f"(recommended {self.check.nw_recommended:.2f}), K = {self.k} tapers, W = {self.w_mhz:.2f} MHz "
             f"= {self.w_mhz / b:.2f} B")
        return s + (f"  [{self.note}]" if self.note else "")


def determine_spectral_parameters(roi_band_mhz, ax_window_mm: float, axial_position_mm: np.ndarray,
                                  fs_mhz: float, kappa: float = KAPPA,
                                  avoid_freq: Sequence[float] = ()) -> SpectralParameters:
    """NW for the multitaper spectra (DetermineSpectralParameters.m, one probe/version).

    roi_band_mhz: (fLow, fHigh) of the ROI from the first pass (ROIBand / DrawROIWithBand).
    ax_window_mm: axial window (AxPERLength). axial_position_mm: RF sample depths (for deltaAx).
    """
    f_low, f_high = (float(v) for v in roi_band_mhz)
    if not (np.isfinite(f_low) and np.isfinite(f_high)):
        raise ValueError("The ROI has no band (fLow / fHigh is NaN)")
    band = round_band_inward(f_low, f_high)
    if band[1] <= band[0]:
        raise ValueError(f"ROI band {f_low:.2f} - {f_high:.2f} MHz is empty after rounding to 0.1 MHz")
    n = window_samples(ax_window_mm, centre_line_delta_ax(axial_position_mm))
    avoid = tuple(np.atleast_1d(np.asarray(avoid_freq, dtype=float)).ravel())
    chk = check_mtm_parameters(n, band, fs_mhz, kappa=kappa, avoid_freq=avoid)

    nw_use = _mround(chk.nw_recommended * 2) / 2
    note = ""
    if np.isfinite(chk.nw_max_before_leak) and nw_use > chk.nw_max_before_leak:
        nw_use = np.floor(chk.nw_max_before_leak * 2) / 2
        note = "capped by AvoidFreq"
    if nw_use < 1.5:
        note = (note + " NW < 1.5: single taper - lengthen window or raise fLow").strip()
    return SpectralParameters(band_mhz=band, roi_band_mhz=(f_low, f_high), n=n, fs_mhz=float(fs_mhz),
                              ax_window_mm=float(ax_window_mm), kappa=kappa, avoid_freq=avoid,
                              nw=float(nw_use), k=n_tapers(nw_use), w_mhz=float(nw_use / chk.t_us), note=note,
                              check=chk)
