import numpy as np
from ...paramap.decorators import supported_spatial_dims, output_vars
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.analysis import Window
from ....data_objs.image import UltrasoundRfImage

@supported_spatial_dims(2, 3)
@output_vars("bmode_mean")
def bmode_intensity(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute B-Mode intensity (mean of log-compressed envelope).
    """
    # Envelope detection (simple Hilbert transform is usually done before, 
    # but here we might just take the magnitude if it's already IQ or Hilbert)
    # For RF data, we usually need the envelope.
    from scipy.signal import hilbert
    envelope = np.abs(hilbert(scan_rf_window, axis=0))
    
    # Log compression
    log_envelope = 20 * np.log10(envelope + 1e-10)
    
    # Mean intensity in the window
    window.results.bmode_mean = np.mean(log_envelope)