import numpy as np
from scipy.optimize import curve_fit

from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.analysis import Window
from ....data_objs.image import UltrasoundRfImage

@supported_spatial_dims(2, 3)
@output_vars("central_freq_shift", "central_freq_scan", "central_freq_ref")
@dependencies("compute_power_spectra")
def central_freq_shift(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """
    Compute the central frequency shift between scan and reference using Gaussian fit or peak method.

    Args:
        scan_rf_window (np.ndarray): RF data of the scan window.
        phantom_rf_window (np.ndarray): RF data of the phantom window.
        window (Window): Window object to store results.
        config (RfAnalysisConfig): Configuration object for analysis.
        image_data (UltrasoundRfImage): Image data object containing metadata.
        apply_gaussian_fit (bool, optional): If True (default), fit a Gaussian to the power spectrum to find central frequency. If False, use the peak frequency directly.
    """
    freq_arr = window.results.f
    scan_ps = window.results.ps # dB
    ref_ps = window.results.r_ps # dB

    # get non db power spectra
    scan_ps_non_db = 10 ** (scan_ps / 20)
    ref_ps_non_db = 10 ** (ref_ps / 20)
    
    # Find the peak (maximum) for initial guess
    scan_peak_idx = np.argmax(scan_ps_non_db)
    ref_peak_idx = np.argmax(ref_ps_non_db)
    scan_peak_freq = freq_arr[scan_peak_idx]
    ref_peak_freq = freq_arr[ref_peak_idx]

    # fit gaussian function to power spectra and find central frequency
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    
    # Initial guesses: amplitude, center, width
    scan_p0 = [np.max(scan_ps_non_db), scan_peak_freq, 1e6]
    ref_p0 = [np.max(ref_ps_non_db), ref_peak_freq, 1e6]

    # Check for kwarg to apply gaussian fit
    apply_gaussian_fit = kwargs.get('CF_gaussian_fit')
    
    if apply_gaussian_fit:
        # Fit Gaussian to scan and reference, with robust error handling
        try:
            popt_scan, _ = curve_fit(gaussian, freq_arr, scan_ps_non_db, p0=scan_p0)
            central_freq_scan = popt_scan[1]
        except Exception as e:
            central_freq_scan = scan_peak_freq

        try:
            popt_ref, _ = curve_fit(gaussian, freq_arr, ref_ps_non_db, p0=ref_p0)
            central_freq_ref = popt_ref[1]
        except Exception as e:
            central_freq_ref = ref_peak_freq
    else:
        # Use peak frequency directly
        central_freq_scan = scan_peak_freq
        central_freq_ref = ref_peak_freq

    # compute central frequency shift
    central_freq_shift = central_freq_scan - central_freq_ref

    # store results
    window.results.central_freq_scan = central_freq_scan
    window.results.central_freq_ref = central_freq_ref
    window.results.central_freq_shift = central_freq_shift