from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig

@extensions()
def philips_3d_config(analysis_path: str, **kwargs) -> RfAnalysisConfig:
    """Class to load RF analysis configuration data from a pickle file saved from the QuantUS UI.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the analysis config file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the analysis config file name.
    """
    out = RfAnalysisConfig()

    # Frequency parameters
    out.transducer_freq_band = [0, 7000000] # [min, max] (Hz)
    out.analysis_freq_band = [2500000, 5000000] # [lower, upper] (Hz)
    out.center_frequency = 4500000 # Hz
    out.sampling_frequency = 4*out.center_frequency # Hz

    # Windowing parameters
    out.ax_win_size = 10 # axial length per window (mm)
    out.lat_win_size = 10 # lateral length per window (mm)
    out.window_thresh = 0.95 # % of window area required to be considered in ROI
    out.axial_overlap = 0.5 # % of window overlap in axial direction
    out.lateral_overlap = 0.5 # % of window overlap in lateral direction
    
    # 3D scan parameters
    out.cor_win_size = 20 # coronal length per window (mm)
    out.coronal_overlap = 0.5 # % of window overlap in coronal direction

    return out