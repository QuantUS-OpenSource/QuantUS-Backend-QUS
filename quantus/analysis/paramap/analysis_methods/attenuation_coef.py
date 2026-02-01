import numpy as np

from ..transforms import compute_hanning_power_spec
from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.analysis import Window
from ....data_objs.image import UltrasoundRfImage

@supported_spatial_dims(2, 3)
@output_vars("att_coef")
@required_kwargs("ref_attenuation")
@default_kwarg_vals(0.7)
@dependencies("compute_power_spectra")
def attenuation_coef(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute the local attenuation coefficient of the ROI using the Spectral Difference
    Method for Local Attenuation Estimation. This method computes the attenuation coefficient
    for multiple frequencies and returns the slope of the attenuation as a function of frequency.

    Args:
        rf_data (np.ndarray): RF data of the ROI (n lines x m samples).
        ref_rf_data (np.ndarray): RF data of the phantom (n lines x m samples).
        overlap (float): Overlap percentage for analysis windows.
        window_depth (int): Depth of each window in samples.
    Updated and verified : Feb 2025 - IR
    """
    overlap = 50
    window_depth = min(100, scan_rf_window.shape[0] // 3)
    ref_attenuation = kwargs['ref_attenuation']  # Reference attenuation coefficient (dB/cm/MHz)
    
    sampling_frequency = config.sampling_frequency
    start_frequency = config.analysis_freq_band[0]
    end_frequency = config.analysis_freq_band[1]

    # Initialize arrays for storing intensities (log of power spectrum for each frequency)
    ps_sample = []  # ROI power spectra
    ps_ref = []     # Phantom power spectra

    start_idx = 0
    end_idx = window_depth
    window_center_indices = []
    counter = 0

    # Loop through the windows in the RF data
    while end_idx < scan_rf_window.shape[0]:
        sub_window_rf = scan_rf_window[start_idx:end_idx]
        f, ps = compute_hanning_power_spec(sub_window_rf, start_frequency, end_frequency, sampling_frequency)
        ps_sample.append(20 * np.log10(ps))  # Log scale intensity for the ROI

        ref_sub_window_rf = phantom_rf_window[start_idx:end_idx]
        ref_f, ref_ps = compute_hanning_power_spec(ref_sub_window_rf, start_frequency, end_frequency, sampling_frequency)
        ps_ref.append(20 * np.log10(ref_ps))  # Log scale intensity for the phantom

        window_center_indices.append((start_idx + end_idx) // 2)

        start_idx += int(window_depth * (1 - (overlap / 100)))
        end_idx = start_idx + window_depth
        counter += 1

    # Convert window depths to cm
    axial_res_cm = image_data.axial_res / 10
    window_depths_cm = np.array(window_center_indices) * axial_res_cm

    attenuation_coefficients = []  # One coefficient for each frequency

    f = f / 1e6
    ps_sample = np.array(ps_sample)
    ps_ref = np.array(ps_ref)

    mid_idx = f.shape[0] // 2
    start_idx = max(0, mid_idx - 25)
    end_idx = min(f.shape[0], mid_idx + 25)

    # Compute attenuation for each frequency
    for f_idx in range(start_idx, end_idx):
        normalized_intensities = np.subtract(ps_sample[:, f_idx], ps_ref[:, f_idx])
        p = np.polyfit(window_depths_cm, normalized_intensities, 1)
        local_attenuation = ref_attenuation * f[f_idx] - (1 / 4) * p[0]  # dB/cm
        attenuation_coefficients.append(local_attenuation / f[f_idx])  # dB/cm/MHz

    attenuation_coef = np.mean(attenuation_coefficients)
    
    # Fill in attributes defined in "output_vars" decorator
    window.results.att_coef = attenuation_coef # dB/cm/MHz