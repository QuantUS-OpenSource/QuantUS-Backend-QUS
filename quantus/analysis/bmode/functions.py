import logging
import numpy as np
from scipy.signal import hilbert

from ..paramap.decorators import supported_spatial_dims, output_vars, dependencies
from ...data_objs.analysis_config import RfAnalysisConfig
from ...data_objs.analysis import Window
from ...data_objs.image import UltrasoundRfImage

# Configure logging
logger = logging.getLogger(__name__)

# Import radiomics module for wrapper functions
from . import radiomics_utils as radiomics


# ------------------------------------------------------------------
# Original functions (unchanged)
# ------------------------------------------------------------------

@supported_spatial_dims(2, 3)
@output_vars("bmode_mean")
def bmode_intensity(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute B-Mode intensity (mean of log-compressed envelope), normalised by phantom."""
    logger.debug(f"Calculating B-Mode Intensity. Scan window shape: {scan_rf_window.shape}")
    
    # Calculate for scan
    scan_envelope = np.abs(hilbert(scan_rf_window, axis=0))
    scan_log_env = 20.0 * np.log10(scan_envelope + 1e-10)
    scan_intensity = np.mean(scan_log_env)
    
    # Calculate for phantom
    phantom_intensity = 0.0
    if phantom_rf_window is not None:
        phantom_envelope = np.abs(hilbert(phantom_rf_window, axis=0))
        phantom_log_env = 20.0 * np.log10(phantom_envelope + 1e-10)
        phantom_intensity = np.mean(phantom_log_env)
    
    # Apply phantom normalization
    result = scan_intensity / (phantom_intensity + 1e-10)
    window.results.bmode_mean = result
    logger.debug(f"B-Mode Intensity result: {result}")

@supported_spatial_dims(2, 3)
@output_vars("bmode_snr")
def bmode_snr(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute Signal-to-Noise Ratio (SNR) of the envelope, normalised by phantom."""
    logger.debug(f"Calculating B-Mode SNR. Scan window shape: {scan_rf_window.shape}")

    # Calculate SNR for scan
    scan_envelope = np.abs(hilbert(scan_rf_window, axis=0))
    scan_mean = np.mean(scan_envelope)
    scan_std = np.std(scan_envelope)
    scan_snr = scan_mean / (scan_std + 1e-10)
    
    # Calculate SNR for phantom
    phantom_snr = 1.0 # default to neutral
    if phantom_rf_window is not None:
        phantom_envelope = np.abs(hilbert(phantom_rf_window, axis=0))
        phantom_mean = np.mean(phantom_envelope)
        phantom_std = np.std(phantom_envelope)
        phantom_snr = phantom_mean / (phantom_std + 1e-10)
    
    # Apply phantom normalization
    result = scan_snr / (phantom_snr + 1e-10)
    window.results.bmode_snr = result
    logger.debug(f"B-Mode SNR result: {result}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_std")
def bmode_std(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute B-Mode standard deviation of log-compressed envelope, normalised by phantom."""
    logger.debug(f"Calculating B-Mode Std. Scan window shape: {scan_rf_window.shape}")
    
    # Calculate for scan
    scan_envelope = np.abs(hilbert(scan_rf_window, axis=0))
    scan_log_env = 20.0 * np.log10(scan_envelope + 1e-10)
    scan_std = np.std(scan_log_env)
    
    # Calculate for phantom
    phantom_std = 1.0
    if phantom_rf_window is not None:
        phantom_envelope = np.abs(hilbert(phantom_rf_window, axis=0))
        phantom_log_env = 20.0 * np.log10(phantom_envelope + 1e-10)
        phantom_std = np.std(phantom_log_env)
    
    # Apply phantom normalization
    result = scan_std / (phantom_std + 1e-10)
    window.results.bmode_std = result
    logger.debug(f"B-Mode Std result: {result}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_median")
def bmode_median(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute B-Mode median of log-compressed envelope, normalised by phantom."""
    logger.debug(f"Calculating B-Mode Median. Scan window shape: {scan_rf_window.shape}")
    
    # Calculate for scan
    scan_envelope = np.abs(hilbert(scan_rf_window, axis=0))
    scan_log_env = 20.0 * np.log10(scan_envelope + 1e-10)
    scan_median = np.median(scan_log_env)
    
    # Calculate for phantom
    phantom_median = 1.0
    if phantom_rf_window is not None:
        phantom_envelope = np.abs(hilbert(phantom_rf_window, axis=0))
        phantom_log_env = 20.0 * np.log10(phantom_envelope + 1e-10)
        phantom_median = np.median(phantom_log_env)
    
    # Apply phantom normalization
    result = scan_median / (phantom_median + 1e-10)
    window.results.bmode_median = result
    logger.debug(f"B-Mode Median result: {result}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_iqr")
def bmode_iqr(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute B-Mode Interquartile Range (IQR) of log-compressed envelope, normalised by phantom."""
    logger.debug(f"Calculating B-Mode IQR. Scan window shape: {scan_rf_window.shape}")
    
    # Calculate for scan
    scan_envelope = np.abs(hilbert(scan_rf_window, axis=0))
    scan_log_env = 20.0 * np.log10(scan_envelope + 1e-10)
    scan_iqr = np.percentile(scan_log_env, 75) - np.percentile(scan_log_env, 25)
    
    # Calculate for phantom
    phantom_iqr = 1.0
    if phantom_rf_window is not None:
        phantom_envelope = np.abs(hilbert(phantom_rf_window, axis=0))
        phantom_log_env = 20.0 * np.log10(phantom_envelope + 1e-10)
        phantom_iqr = np.percentile(phantom_log_env, 75) - np.percentile(phantom_log_env, 25)
    
    # Apply phantom normalization
    result = scan_iqr / (phantom_iqr + 1e-10)
    window.results.bmode_iqr = result
    logger.debug(f"B-Mode IQR result: {result}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_skewness")
def bmode_skewness(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute B-Mode skewness of log-compressed envelope, normalised by phantom."""
    from scipy.stats import skew
    logger.debug("Calculating B-Mode Skewness")
    
    scan_envelope = np.abs(hilbert(scan_rf_window, axis=0))
    scan_log_env = 20.0 * np.log10(scan_envelope + 1e-10)
    scan_skew = skew(scan_log_env.ravel())
    
    phantom_skew = 0.0
    if phantom_rf_window is not None:
        phantom_envelope = np.abs(hilbert(phantom_rf_window, axis=0))
        phantom_log_env = 20.0 * np.log10(phantom_envelope + 1e-10)
        phantom_skew = skew(phantom_log_env.ravel())
    
    result = scan_skew - phantom_skew
    window.results.bmode_skewness = result
    logger.debug(f"B-Mode Skewness result: {result}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_kurtosis")
def bmode_kurtosis(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute B-Mode kurtosis of log-compressed envelope, normalised by phantom."""
    from scipy.stats import kurtosis
    logger.debug("Calculating B-Mode Kurtosis")
    
    scan_envelope = np.abs(hilbert(scan_rf_window, axis=0))
    scan_log_env = 20.0 * np.log10(scan_envelope + 1e-10)
    scan_kurt = kurtosis(scan_log_env.ravel())
    
    phantom_kurt = 0.0
    if phantom_rf_window is not None:
        phantom_envelope = np.abs(hilbert(phantom_rf_window, axis=0))
        phantom_log_env = 20.0 * np.log10(phantom_envelope + 1e-10)
        phantom_kurt = kurtosis(phantom_log_env.ravel())
    
    result = scan_kurt - phantom_kurt
    window.results.bmode_kurtosis = result
    logger.debug(f"B-Mode Kurtosis result: {result}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_entropy")
def bmode_entropy(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute Shannon Entropy of the log-compressed envelope distribution."""
    logger.debug("Calculating B-Mode Entropy")
    
    def _calc_entropy(data):
        # 100-bin histogram for entropy estimation
        hist, _ = np.histogram(data, bins=100, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    scan_envelope = np.abs(hilbert(scan_rf_window, axis=0))
    scan_log_env = 20.0 * np.log10(scan_envelope + 1e-10)
    scan_ent = _calc_entropy(scan_log_env)
    
    phantom_ent = 1.0
    if phantom_rf_window is not None:
        phantom_envelope = np.abs(hilbert(phantom_rf_window, axis=0))
        phantom_log_env = 20.0 * np.log10(phantom_envelope + 1e-10)
        phantom_ent = _calc_entropy(phantom_log_env)
    
    result = scan_ent / (phantom_ent + 1e-10)
    window.results.bmode_entropy = result
    logger.debug(f"B-Mode Entropy result: {result}")


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
    logger.debug("Calculating Radiomics Mean")
    value = radiomics.calc_radiomics_mean(scan_rf_window, phantom_rf_window, image_data)
    window.results.bmode_radiomics_mean = value
    logger.debug(f"Radiomics Mean result: {value}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_std")
def bmode_radiomics_std_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics first-order Standard Deviation feature."""
    logger.debug("Calculating Radiomics Std")
    value = radiomics.calc_radiomics_std(scan_rf_window, phantom_rf_window, image_data)
    window.results.bmode_radiomics_std = value
    logger.debug(f"Radiomics Std result: {value}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_median")
def bmode_radiomics_median_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics first-order Median feature."""
    logger.debug("Calculating Radiomics Median")
    value = radiomics.calc_radiomics_median(scan_rf_window, phantom_rf_window, image_data)
    window.results.bmode_radiomics_median = value
    logger.debug(f"Radiomics Median result: {value}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_entropy")
def bmode_radiomics_entropy_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics first-order Entropy feature."""
    logger.debug("Calculating Radiomics Entropy")
    value = radiomics.calc_radiomics_entropy(scan_rf_window, phantom_rf_window, image_data)
    window.results.bmode_radiomics_entropy = value
    logger.debug(f"Radiomics Entropy result: {value}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_energy")
def bmode_radiomics_energy_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics first-order Energy feature."""
    logger.debug("Calculating Radiomics Energy")
    value = radiomics.calc_radiomics_energy(scan_rf_window, phantom_rf_window, image_data)
    window.results.bmode_radiomics_energy = value
    logger.debug(f"Radiomics Energy result: {value}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_iqr")
def bmode_radiomics_iqr_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics first-order InterquartileRange feature."""
    logger.debug("Calculating Radiomics IQR")
    value = radiomics.calc_radiomics_iqr(scan_rf_window, phantom_rf_window, image_data)
    window.results.bmode_radiomics_iqr = value
    logger.debug(f"Radiomics IQR result: {value}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_glcm_contrast")
def bmode_glcm_contrast_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics GLCM Contrast feature."""
    logger.debug("Calculating GLCM Contrast")
    value = radiomics.calc_glcm_contrast(scan_rf_window, phantom_rf_window, image_data)
    window.results.bmode_glcm_contrast = value
    logger.debug(f"GLCM Contrast result: {value}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_glcm_homogeneity")
def bmode_glcm_homogeneity_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics GLCM Homogeneity feature."""
    logger.debug("Calculating GLCM Homogeneity")
    value = radiomics.calc_glcm_homogeneity(scan_rf_window, phantom_rf_window, image_data)
    window.results.bmode_glcm_homogeneity = value
    logger.debug(f"GLCM Homogeneity result: {value}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_glcm_correlation")
def bmode_glcm_correlation_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics GLCM Correlation feature."""
    logger.debug("Calculating GLCM Correlation")
    value = radiomics.calc_glcm_correlation(scan_rf_window, phantom_rf_window, image_data)
    window.results.bmode_glcm_correlation = value
    logger.debug(f"GLCM Correlation result: {value}")


@supported_spatial_dims(2, 3)
@output_vars("bmode_glcm_energy")
def bmode_glcm_energy_wrapper(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """Wrapper for PyRadiomics GLCM Energy feature."""
    logger.debug("Calculating GLCM Energy")
    value = radiomics.calc_glcm_energy(scan_rf_window, phantom_rf_window, image_data)
    window.results.bmode_glcm_energy = value
    logger.debug(f"GLCM Energy result: {value}")
