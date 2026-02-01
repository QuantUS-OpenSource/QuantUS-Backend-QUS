from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig

@extensions()
def clarius_C3_config(analysis_path: str, **kwargs) -> RfAnalysisConfig:
    """Function to load RF analysis configuration data for Clarius ultrasound data.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the analysis config file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the analysis config file name.
    """
    out = RfAnalysisConfig()
    
    # Frequency parameters
    out.transducer_freq_band = [0, 7.5e6]  # [min, max] (Hz), up to 15MHz for high-frequency probes
    out.analysis_freq_band = [1e6, 4e6]  # [lower, upper] (Hz), typical analysis range
    out.center_frequency = 2.5e6  # Hz, typical center frequency for Clarius probes
    out.sampling_frequency = 15e6  # Hz, following Nyquist criterion
    
    # Windowing parameters
    out.ax_win_size = 20  # axial length per window (mm)
    out.lat_win_size = 10  # lateral length per window (mm)
    out.window_thresh = 0.95  # % of window area required to be considered in ROI
    out.axial_overlap = 0.5  # % of window overlap in axial direction
    out.lateral_overlap = 0.5  # % of window overlap in lateral direction
    
    out.cor_win_size = None  # coronal length per window (mm)
    out.coronal_overlap = None  # % of window overlap in coronal direction

    return out