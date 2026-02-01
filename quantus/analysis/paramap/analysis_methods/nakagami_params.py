import numpy as np

from scipy.signal import hilbert
from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.analysis import Window
from ....data_objs.image import UltrasoundRfImage

@supported_spatial_dims(2)
@output_vars("nak_w", "nak_u")
def nakagami_params(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
            window: Window, config: RfAnalysisConfig, 
            image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute Nakagami parameters for the ROI.

    Source: Tsui, P. H., Wan, Y. L., Huang, C. C. & Wang, M. C. 
    Effect of adaptive threshold filtering on ultrasonic Nakagami 
    parameter to detect variation in scatterer concentration. Ultrason. 
    Imaging 32, 229–242 (2010). https://doi.org/10.1177%2F016173461003200403

    Args:
        rf_data (np.ndarray): RF data of the ROI (n lines x m samples).
        
    Returns:
        Tuple: Nakagami parameters (w, u) for the ROI.
    """
    r = np.abs(hilbert(scan_rf_window, axis=0))
    w = np.nanmean(r ** 2, axis=1)
    u = (w ** 2) / np.var(r ** 2, axis=1)

    # Averaging to get single parameter values
    w = np.nanmean(w)
    u = np.nanmean(u)
    
    # Fill in attributes defined in "output_vars" decorator
    window.results.nak_w = w
    window.results.nak_u = u