import numpy as np
from scipy.signal import hilbert

from ..paramap.decorators import supported_spatial_dims, output_vars, dependencies
from ...data_objs.analysis_config import RfAnalysisConfig
from ...data_objs.analysis import Window
from ...data_objs.image import UltrasoundRfImage

# Import radiomics module for wrapper functions
from . import radiomics


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
# Radiomics wrapper functions - delegate to radiomics module
# ------------------------------------------------------------------
# Each wrapper calls the corresponding radiomics function and stores result


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_mean")
def bmode_radiomics_mean_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics first-order Mean feature."""
    value = radiomics.calc_radiomics_mean(scan_rf_window, phantom_rf_window)
    window.results.bmode_radiomics_mean = value


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_std")
def bmode_radiomics_std_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics first-order Standard Deviation feature."""
    value = radiomics.calc_radiomics_std(scan_rf_window, phantom_rf_window)
    window.results.bmode_radiomics_std = value


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_median")
def bmode_radiomics_median_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics first-order Median feature."""
    value = radiomics.calc_radiomics_median(scan_rf_window, phantom_rf_window)
    window.results.bmode_radiomics_median = value


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_entropy")
def bmode_radiomics_entropy_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics first-order Entropy feature."""
    value = radiomics.calc_radiomics_entropy(scan_rf_window, phantom_rf_window)
    window.results.bmode_radiomics_entropy = value


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_energy")
def bmode_radiomics_energy_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics first-order Energy feature."""
    value = radiomics.calc_radiomics_energy(scan_rf_window, phantom_rf_window)
    window.results.bmode_radiomics_energy = value


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_iqr")
def bmode_radiomics_iqr_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics first-order InterquartileRange feature."""
    value = radiomics.calc_radiomics_iqr(scan_rf_window, phantom_rf_window)
    window.results.bmode_radiomics_iqr = value


@supported_spatial_dims(2, 3)
@output_vars("bmode_glcm_contrast")
def bmode_glcm_contrast_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics GLCM Contrast feature."""
    value = radiomics.calc_glcm_contrast(scan_rf_window, phantom_rf_window)
    window.results.bmode_glcm_contrast = value


@supported_spatial_dims(2, 3)
@output_vars("bmode_glcm_homogeneity")
def bmode_glcm_homogeneity_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics GLCM Homogeneity feature."""
    value = radiomics.calc_glcm_homogeneity(scan_rf_window, phantom_rf_window)
    window.results.bmode_glcm_homogeneity = value


@supported_spatial_dims(2, 3)
@output_vars("bmode_glcm_correlation")
def bmode_glcm_correlation_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics GLCM Correlation feature."""
    value = radiomics.calc_glcm_correlation(scan_rf_window, phantom_rf_window)
    window.results.bmode_glcm_correlation = value


@supported_spatial_dims(2, 3)
@output_vars("bmode_glcm_energy")
def bmode_glcm_energy_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics GLCM Energy feature."""
    value = radiomics.calc_glcm_energy(scan_rf_window, phantom_rf_window)
    window.results.bmode_glcm_energy = value
