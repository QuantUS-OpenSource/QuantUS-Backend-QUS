import numpy as np

from ..transforms import compute_power_spec
from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.analysis import Window
from ....data_objs.image import UltrasoundRfImage

@supported_spatial_dims(2, 3)
@output_vars("att_coef")
@required_kwargs("ref_atten", "ref_atten_exp", "ref_atten_offset", "n_secs", "sec_overlap", "n_fft")
@default_kwarg_vals(0.7, 1., 0., 5, 0.50, 8192)
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
    if phantom_rf_window.ndim == 3:
        assert scan_rf_window.ndim == 2, "AC only supported for 2D data"
        # print("3D Phantom with shape", phantom_rf_window.shape)
        phantom_rf_window = phantom_rf_window[:,:,0]
        
    n_secs = kwargs['n_secs']
    sec_overlap = kwargs['sec_overlap']
    ref_atten = kwargs['ref_atten']
    ref_atten_exp = kwargs['ref_atten_exp']
    ref_atten_offset = kwargs['ref_atten_offset']
    n_fft = kwargs['n_fft']
    
    roi_len = scan_rf_window.shape[0]
    sec_len = round(roi_len / ((1-sec_overlap)*(n_secs-1) + 1))
    sec_offset = round(roi_len / ((1-sec_overlap)*(n_secs-1) + 1) * (1 - sec_overlap))
    sec_start_indices = np.arange(n_secs) * sec_offset
    sec_end_indices = sec_start_indices + sec_len

    sampling_frequency = config.sampling_frequency
    start_frequency = config.analysis_freq_band[0]
    end_frequency = config.analysis_freq_band[1]

    # Initialize arrays for storing intensities (log of power spectrum for each frequency)
    ps_sample = np.zeros((n_secs, n_fft))  # ROI power spectra
    ps_ref = np.zeros((n_secs, n_fft))     # Phantom power spectra
    window_center_indices = np.zeros(n_secs)

    for i, (sec_start_ix, sec_end_ix) in enumerate(zip(sec_start_indices, sec_end_indices)):
        sub_window_rf = scan_rf_window[sec_start_ix:sec_end_ix]
        f, ps = compute_power_spec(sub_window_rf, start_frequency, end_frequency, sampling_frequency, n_fft, use_hanning=False)
        ps_sample[i] = np.log(ps)  # Log scale intensity for the ROI

        ref_sub_window_rf = phantom_rf_window[sec_start_ix:sec_end_ix]
        ref_f, ref_ps = compute_power_spec(ref_sub_window_rf, start_frequency, end_frequency, sampling_frequency, n_fft, use_hanning=False)
        ps_ref[i] = np.log(ref_ps)  # Log scale intensity for the phantom
        
        window_center_indices[i] = (sec_start_ix + sec_end_ix) // 2
        
    depths = ((window_center_indices + window.ax_min) * image_data.axial_res)/10

    spectral_ratios = ps_sample - ps_ref

    att_temp = np.zeros((n_fft))
    nepers_to_db = 8.686

    for i in range(n_fft):
        p = np.polyfit(depths, spectral_ratios[:, i], 1)
        att_temp[i] = nepers_to_db * p[0] / -4
        
    att_ref = ref_atten*np.pow(f/1e6, ref_atten_exp) + ref_atten_offset
    att = att_temp + att_ref

    window.results.att_coef = np.mean(att/f*1e6) # db/MHz/cm
