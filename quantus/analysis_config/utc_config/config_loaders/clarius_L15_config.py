import logging

from ..decorators import *
from ....image_loading.utc_loaders.clarius_rf.parser import clarius_sampling_frequency
from ....data_objs.analysis_config import RfAnalysisConfig, WIN_SIZE_UNITS_MM

@extensions()
def clarius_L15_config(analysis_path: str, **kwargs) -> RfAnalysisConfig:
    """Function to load RF analysis configuration data for Clarius ultrasound data.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the analysis config file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the analysis config file name.
    """
    out = RfAnalysisConfig()

    # Frequency parameters
    out.analysis_freq_band = [2e6, 8e6]  # [lower, upper] (Hz), typical analysis range
    out.center_frequency = 5e6  # Hz, typical center frequency for Clarius probes
    # Sampling frequency from the scan's rf.yml (fallback 30e6 Hz if it can't be read),
    # and the transducer band up to Nyquist.
    try:
        out.sampling_frequency = clarius_sampling_frequency(kwargs["scan_path"])
    except Exception as e:  # e.g. called without a scan_path
        logging.warning(f"clarius_L15_config: could not read the sampling frequency from rf.yml ({e}); using 30e6 Hz")
        out.sampling_frequency = 30e6
    out.transducer_freq_band = [0, out.sampling_frequency / 2]  # [min, max] (Hz)
    out.speed_of_sound = 1540.0  # m/s, standard soft-tissue assumption

    # Windowing parameters - adjusted for smaller windows due to high frequency
    out.ax_win_size = 2  # axial length per window (mm)
    out.ax_win_size_units = WIN_SIZE_UNITS_MM
    out.lat_win_size = 4  # lateral length per window (mm)
    out.lat_win_size_units = WIN_SIZE_UNITS_MM
    out.window_thresh = 0.5  # % of window area required to be considered in ROI
    out.axial_overlap = 0.75  # % of window overlap in axial direction
    out.lateral_overlap = 0.75  # % of window overlap in lateral direction

    # 3D scan parameters
    out.cor_win_size = None  # coronal length per window (mm)
    out.coronal_overlap = None  # % of window overlap in coronal direction

    return out