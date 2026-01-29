import numpy as np
from ...paramap.decorators import supported_spatial_dims, output_vars
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.analysis import Window
from ....data_objs.image import UltrasoundRfImage

@supported_spatial_dims(2, 3)
@output_vars("snr")
def bmode_snr(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute Signal-to-Noise Ratio (SNR) of the envelope.
    """
    from scipy.signal import hilbert
    envelope = np.abs(hilbert(scan_rf_window, axis=0))
    
    mean_val = np.mean(envelope)
    std_val = np.std(envelope)
    
    if std_val > 0:
        snr = mean_val / std_val
    else:
        snr = 0
        
    window.results.snr = snr