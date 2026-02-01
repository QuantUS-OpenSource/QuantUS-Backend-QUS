from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig

@extensions()
def clarius_L15_config(analysis_path: str, **kwargs) -> RfAnalysisConfig:
    """Function to load RF analysis configuration data for Clarius ultrasound data.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the analysis config file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the analysis config file name.
    """
    out = RfAnalysisConfig()

    # Frequency parameters
    out.transducer_freq_band = [0, 15e6]  # [min, max] (Hz), up to 15MHz for high-frequency probes
    out.analysis_freq_band = [2e6, 8e6]  # [lower, upper] (Hz), typical analysis range
    out.center_frequency = 5e6  # Hz, typical center frequency for Clarius probes
    out.sampling_frequency = 30e6 # Hz, following Nyquist criterion

    # Windowing parameters - adjusted for smaller windows due to high frequency
    out.ax_win_size = 20  # axial length per window (mm) 
    out.lat_win_size = 40  # lateral length per window (mm)
    out.window_thresh = 0.5  # % of window area required to be considered in ROI
    out.axial_overlap = 0.75  # % of window overlap in axial direction
    out.lateral_overlap = 0.75  # % of window overlap in lateral direction

    # 3D scan parameters
    out.cor_win_size = None  # coronal length per window (mm)
    out.coronal_overlap = None  # % of window overlap in coronal direction

    return out