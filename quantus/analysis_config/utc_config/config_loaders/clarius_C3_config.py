import logging

from ..decorators import *
from ....image_loading.utc_loaders.clarius_rf.parser import clarius_sampling_frequency
from ....data_objs.analysis_config import RfAnalysisConfig, WIN_SIZE_UNITS_WAVELENGTH, WIN_SIZE_UNITS_A_LINES

@extensions()
def clarius_C3_config(analysis_path: str, **kwargs) -> RfAnalysisConfig:
    """Function to load RF analysis configuration data for Clarius ultrasound data.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the analysis config file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the analysis config file name.
    """
    out = RfAnalysisConfig()
    
    # Frequency parameters
    out.analysis_freq_band = [1.5e6, 3.5e6]  # [lower, upper] (Hz), typical analysis range
    out.center_frequency = 2.5e6  # Hz, typical center frequency for Clarius probes
    # Sampling frequency from the scan's rf.yml (fallback 15e6 Hz if it can't be read),
    # and the transducer band up to Nyquist.
    try:
        out.sampling_frequency = clarius_sampling_frequency(kwargs["scan_path"])
    except Exception as e:  # e.g. called without a scan_path
        logging.warning(f"clarius_C3_config: could not read the sampling frequency from rf.yml ({e}); using 15e6 Hz")
        out.sampling_frequency = 15e6
    out.transducer_freq_band = [0, out.sampling_frequency / 2]  # [min, max] (Hz)
    out.speed_of_sound = 1540.0  # m/s, standard soft-tissue assumption

    # Windowing parameters
    # Axial in wavelengths rather than a hardcoded mm value, so window size scales with
    # frequency: wavelength = speed_of_sound/center_frequency.
    out.ax_win_size = 15  # axial length per window (wavelengths)
    out.ax_win_size_units = WIN_SIZE_UNITS_WAVELENGTH
    # Lateral in A-lines directly, since a curvilinear probe's raw beam spacing isn't a fixed
    # mm/beam value (beams fan out from the apex) -- xResRF's mm-based approximation made an
    # 8mm lateral window round up to 264 beams, more than this probe's 192 total, so no window
    # could ever fit. A-lines sidesteps that and always fits within the real line count
    # regardless of depth.
    out.lat_win_size = 15  # lateral length per window (A-lines)
    out.lat_win_size_units = WIN_SIZE_UNITS_A_LINES
    out.window_thresh = 0.95  # % of window area required to be considered in ROI
    out.axial_overlap = 0.5  # % of window overlap in axial direction
    out.lateral_overlap = 0.5  # % of window overlap in lateral direction
    
    out.cor_win_size = None  # coronal length per window (mm)
    out.coronal_overlap = None  # % of window overlap in coronal direction

    return out