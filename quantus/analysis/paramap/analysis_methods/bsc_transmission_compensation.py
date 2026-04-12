import numpy as np

from ..decorators import *
from ..transforms import compute_power_spec
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.analysis import Window
from ....data_objs.image import UltrasoundRfImage

@supported_spatial_dims(2, 3)
@output_vars("bsc")
@required_kwargs("ref_bsc_path", "sample_atten", "sample_atten_exp", "sample_atten_offset",
                 "ref_atten", "ref_atten_exp", "ref_atten_offset",
                 "n_fft", "cirs_path")
@default_kwarg_vals("", 1., 1., 0., 1., 1., 0., 8192, "")
def bsc_transmission_compensation(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
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
    if phantom_rf_window.ndim == 3:
        assert scan_rf_window.ndim == 2, "BSC only supported for 2D data"
        # print("3D Phantom with shape", phantom_rf_window.shape)
        phantom_rf_window = phantom_rf_window[:,:,0]
    
    # Required kwargs
    ref_bsc_path = kwargs['ref_bsc_path']
    sample_atten = kwargs['sample_atten']
    sample_atten_exp = kwargs['sample_atten_exp']
    sample_atten_offset = kwargs['sample_atten_offset']
    ref_atten = kwargs['ref_atten']
    ref_atten_exp = kwargs['ref_atten_exp']
    ref_atten_offset = kwargs['ref_atten_offset']
    n_fft = kwargs['n_fft']
    cirs_path = kwargs['cirs_path']
    
    sampling_frequency = config.sampling_frequency
    start_frequency = config.analysis_freq_band[0]
    end_frequency = config.analysis_freq_band[1]
    
    f, ps = compute_power_spec(scan_rf_window, start_frequency, end_frequency, 
                               sampling_frequency, n_fft, use_hanning=True)
    _, ref_ps = compute_power_spec(phantom_rf_window, start_frequency, end_frequency, 
                               sampling_frequency, n_fft, use_hanning=True)
    
    f *= 1e-6 # Hz --> MHz
    
    # Attenuation compensation
    dB_to_Np = 1 / (20 * np.log10(np.exp(1)))  # same as MATLAB
    dist =  (((window.ax_min + window.ax_max)/2) * image_data.axial_res)/10
    
    def att_exp_factor(atten, atten_exp, atten_offset):
        alpha1 = (
            atten * dist * (f ** atten_exp)
            + dist * atten_offset
        )
        alpha = dB_to_Np * alpha1  # convert to Nepers
        exp_factor = np.exp(4 * alpha)[:, np.newaxis]
        return exp_factor
    
    att_comp_ps = ps * att_exp_factor(sample_atten, sample_atten_exp, sample_atten_offset)
    att_comp_ref_ps = ref_ps * att_exp_factor(ref_atten, ref_atten_exp, ref_atten_offset)
    
    # Transmission compensation
    xy = np.loadtxt(cirs_path)
    T = np.interp(f, xy[:, 0], xy[:, 1])
    T_sq = (T ** 2)[:, np.newaxis]  # shape (num_freq, 1)
    
    final_ps = att_comp_ps / T_sq
    final_ref_ps = att_comp_ref_ps / T_sq
    
    # Average final PS
    final_ps = final_ps.mean(axis=1)
    final_ref_ps = final_ref_ps.mean(axis=1)
    
    ref_bsc_data = np.loadtxt(ref_bsc_path)
    ref_bsc = np.interp(f, ref_bsc_data[:, 0], ref_bsc_data[:, 1])
    
    bsc = (final_ps / final_ref_ps) * ref_bsc
    
    c_freq_mhz = config.center_frequency / 1e6
    c_freq_ix = np.argmin(abs(f - c_freq_mhz))

    window.results.bsc = bsc[c_freq_ix]
