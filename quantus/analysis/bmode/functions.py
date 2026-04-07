import numpy as np
from scipy.signal import hilbert

from ..paramap.decorators import supported_spatial_dims, output_vars, dependencies
from ...data_objs.analysis_config import RfAnalysisConfig
from ...data_objs.analysis import Window
from ...data_objs.image import UltrasoundRfImage

# Import radiomics functions from dedicated module
from .radiomics import (
    bmode_radiomics_mean,
    bmode_radiomics_std,
    bmode_radiomics_median,
    bmode_radiomics_entropy,
    bmode_radiomics_energy,
    bmode_radiomics_iqr,
    bmode_glcm_contrast,
    bmode_glcm_homogeneity,
    bmode_glcm_correlation,
    bmode_glcm_energy,
    RADIOMICS_MAP,
)


# ------------------------------------------------------------------
# Original functions (unchanged)
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Radiomics functions are now imported from the radiomics module
# ------------------------------------------------------------------
# For radiomics-related features, see: radiomics.py
# - First-order features: bmode_radiomics_mean, bmode_radiomics_std, etc.
# - GLCM features: bmode_glcm_contrast, bmode_glcm_homogeneity, etc.
# ------------------------------------------------------------------
