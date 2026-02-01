import numpy as np

from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.analysis import Window
from ....data_objs.image import UltrasoundRfImage

@supported_spatial_dims(2, 3)
@output_vars("bsc")
@required_kwargs("ref_bsc")
@default_kwarg_vals(0.0006)
@dependencies("compute_power_spectra", "attenuation_coef")
def bsc(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
            window: Window, config: RfAnalysisConfig, 
            image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute the backscatter coefficient of the ROI using the reference phantom method.
    Assumes instrumentation and beam terms have the same effect on the signal from both 
    image and phantom. 

    Source: Yao et al. (1990): https://doi.org/10.1177/016173469001200105. PMID: 2184569

    Args:
        freq_arr (np.ndarray): Frequency array of power spectra (Hz).
        scan_ps (np.ndarray): Power spectrum of the analyzed scan at the current region.
        ref_ps (np.ndarray): Power spectrum of the reference phantom at the current region.
        att_coef (float): Attenuation coefficient of the current region (dB/cm/MHz).
        frequency (int): Frequency on which to compute backscatter coefficient (Hz).
        roi_depth (int): Depth of the start of the ROI in samples.
        
    Returns:
        float: Backscatter coefficient of the ROI for the central frequency (1/cm-sr).
        Updated and verified : Feb 2025 - IR
    """
    freq_arr = window.results.f
    scan_ps = window.results.ps
    ref_ps = window.results.r_ps
    att_coef = window.results.att_coef
    roi_depth = scan_rf_window.shape[0]
    
    # Required kwargs
    ref_attenuation = kwargs['ref_attenuation']
    ref_backscatter_coef = kwargs['ref_bsc']
    
    # Optional kwarg
    frequency = kwargs.get('bsc_freq', config.center_frequency)  # Frequency for backscatter coefficient calculation
    
    index = np.argmin(np.abs(freq_arr - frequency))
    ps_sample = scan_ps[index]
    ps_ref = ref_ps[index]
    s_ratio = ps_sample / ps_ref

    np_conversion_factor = np.log(10) / 20 
    converted_att_coef = att_coef * np_conversion_factor  # dB/cm/MHz -> Np/cm/MHz
    converted_ref_att_coef = ref_attenuation * np_conversion_factor  # dB/cm/MHz -> Np/cm        

    window_depth_cm = roi_depth * image_data.axial_res / 10  # cm
    converted_att_coef *= frequency / 1e6  # Np/cm
    converted_ref_att_coef *= frequency / 1e6  # Np/cm        

    att_comp = np.exp(4 * window_depth_cm * (converted_att_coef - converted_ref_att_coef)) 
    bsc = s_ratio * ref_backscatter_coef * att_comp

    # Fill in attributes defined in "output_vars" decorator
    window.results.bsc = bsc # 1/cm-sr