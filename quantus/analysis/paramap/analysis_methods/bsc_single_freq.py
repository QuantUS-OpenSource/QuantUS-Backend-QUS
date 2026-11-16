import numpy as np

from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.analysis import Window
from ....data_objs.image import UltrasoundRfImage

@supported_spatial_dims(2, 3)
@output_vars("bsc_single_freq")
@required_kwargs("ref_bsc", "ref_atten", "sample_atten")
@default_kwarg_vals(0.0006, 0.7, 0.7)
@dependencies("compute_power_spectra")
def bsc_single_freq(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
            window: Window, config: RfAnalysisConfig,
            image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute the backscatter coefficient of the ROI at a single frequency using the
    reference phantom method. A simplified starting point compared to
    "bsc_transmission_compensation": scalar reference BSC, attenuation linear in frequency,
    and no phantom transmission (membrane) compensation.
    Assumes instrumentation and beam terms have the same effect on the signal from both
    image and phantom.

    Source: Yao et al. (1990): https://doi.org/10.1177/016173469001200105. PMID: 2184569

    Kwargs:
        ref_bsc (float): Backscatter coefficient of the reference phantom at bsc_freq (1/cm-sr).
        ref_atten (float): Attenuation coefficient of the reference phantom (dB/cm/MHz).
        sample_atten (float): Assumed attenuation coefficient of the sample (dB/cm/MHz).
        bsc_freq (float, optional): Frequency at which to compute the BSC (Hz). Defaults
            to config.center_frequency.

    Returns:
        None. Writes window.results.bsc_single_freq (1/cm-sr).
    """
    freq_arr = window.results.f
    nps = window.results.nps  # 20*log10(scan power / phantom power)

    # Required kwargs
    ref_bsc = kwargs['ref_bsc']
    ref_atten = kwargs['ref_atten']
    sample_atten = kwargs['sample_atten']

    # Optional kwarg
    frequency = kwargs.get('bsc_freq', config.center_frequency)  # Frequency for backscatter coefficient calculation

    # Spectra are stored as 20*log10 of power, so undo that to get a linear power ratio
    index = np.argmin(np.abs(freq_arr - frequency))
    s_ratio = 10 ** (nps[index] / 20)

    # Attenuation at the analysis frequency, dB/cm/MHz -> Np/cm
    db_to_np = np.log(10) / 20
    freq_mhz = freq_arr[index] / 1e6
    sample_atten_np = sample_atten * freq_mhz * db_to_np
    ref_atten_np = ref_atten * freq_mhz * db_to_np

    # Attenuation accumulates along the full path from the transducer, so compensate
    # using the depth of the window center rather than the window's height
    if hasattr(image_data, 'axial'):
        axial_ix = round((window.ax_min+1 + window.ax_max+1)/2)-1
        depth_cm = image_data.axial[axial_ix] / 10
    else:
        depth_cm = ((window.ax_min + window.ax_max) / 2) * image_data.axial_res / 10

    att_comp = np.exp(4 * depth_cm * (sample_atten_np - ref_atten_np))
    bsc = s_ratio * ref_bsc * att_comp

    # Fill in attributes defined in "output_vars" decorator
    window.results.bsc_single_freq = bsc # 1/cm-sr
