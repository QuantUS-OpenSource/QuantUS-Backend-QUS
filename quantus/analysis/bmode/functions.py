import numpy as np
import SimpleITK as sitk
from scipy.signal import hilbert
from radiomics import featureextractor

from ..paramap.decorators import supported_spatial_dims, output_vars, dependencies
from ...data_objs.analysis_config import RfAnalysisConfig
from ...data_objs.analysis import Window
from ...data_objs.image import UltrasoundRfImage

# =============================================================================
# Radiomics feature mapping (PyRadiomics → our feature names)
# =============================================================================
RADIOMICS_MAP = {
    "firstorder": {
        "Mean": "Mean",
        "StandardDeviation": "Variance",        # we take sqrt later to get std
        "Median": "Median",
        "Entropy": "Entropy",
        "Energy": "Energy",
        "InterquartileRange": "InterquartileRange",
    },
    "glcm": {
        "Contrast": "Contrast",
        "Homogeneity": "InverseDifferenceMoment",   # standard GLCM homogeneity
        "Correlation": "Correlation",
        "JointEnergy": "JointEnergy",
    },
}


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _get_log_envelope(rf_window: np.ndarray) -> np.ndarray:
    """Envelope-detect an RF window and return log-compressed B-mode."""
    envelope = np.abs(hilbert(rf_window, axis=0))
    log_env = 20.0 * np.log10(envelope + 1e-10)
    return np.nan_to_num(log_env, nan=0.0, posinf=0.0, neginf=0.0)


def _build_radiomics_image_and_mask(log_env: np.ndarray):
    """
    Wrap a log-envelope array as a pair of SimpleITK objects suitable for
    PyRadiomics. The mask is all-ones (label = 1), covering the entire
    window — real spatial masking is handled upstream by the QuantUS toolbox.
    
    A single voxel is set to label 2 at position [0,0] as a technical 
    requirement: PyRadiomics internally validates that more than one label 
    value exists in the mask before recognising label 1 as a valid region. 
    This does not spatially restrict the computation in any way.
    """
    image = sitk.GetImageFromArray(log_env.astype(np.float32))
    mask_array = np.ones_like(log_env, dtype=np.uint8)
    mask_array.flat[0] = 2  # sentinel: satisfies PyRadiomics label validation
    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(image)
    return image, mask


def _make_extractor(feature_class: str, feature_names: list) -> featureextractor.RadiomicsFeatureExtractor:
    """
    Return a RadiomicsFeatureExtractor configured to compute only the
    requested features on label = 1.  Resampling is disabled to prevent
    mask/image geometry mismatches on small windowed arrays.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(**{feature_class: feature_names})
    extractor.settings["label"] = 1
    extractor.settings["resampledPixelSpacing"] = None
    extractor.settings["interpolator"] = sitk.sitkNearestNeighbor
    return extractor


def _extract_feature(extractor, rf_window: np.ndarray, key: str):
    """
    Run PyRadiomics on a single RF window and return the named feature value.
    Returns None if rf_window is None (e.g. phantom not provided by QuantUS).
    """
    if rf_window is None:
        return None
    log_env = _get_log_envelope(rf_window)
    image, mask = _build_radiomics_image_and_mask(log_env)
    return float(extractor.execute(image, mask)[key])


def _safe_ratio(scan_val: float, phantom_val, eps: float = 1e-10) -> float:
    """
    Divide scan feature by phantom feature, guarding against near-zero
    denominators.  If phantom_val is None (phantom window unavailable),
    returns the raw scan value instead of a normalised ratio.
    """
    if phantom_val is None:
        return float(scan_val)
    return float(scan_val / (phantom_val + eps))


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
# First-Order Radiomics features  (scan / phantom normalised)
# ------------------------------------------------------------------
# Each function:
#   1. Uses PyRadiomics firstorder features on both scan and phantom windows.
#   2. Stores scan_feature / phantom_feature in window.results.
#   3. Falls back to raw scan value if phantom window is unavailable.
# ------------------------------------------------------------------

@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_mean")
def bmode_radiomics_mean(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """PyRadiomics first-order Mean, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Mean"])
    key = "original_firstorder_Mean"
    window.results.bmode_radiomics_mean = _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_std")
def bmode_radiomics_std(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """PyRadiomics first-order Variance → std (normalised by phantom)."""
    ext = _make_extractor("firstorder", ["Variance"])          # ← changed here
    key = "original_firstorder_Variance"                       # ← changed here
    variance = _extract_feature(ext, scan_rf_window, key)
    phantom_variance = _extract_feature(ext, phantom_rf_window, key)
    
    # Convert variance → standard deviation
    std_scan = np.sqrt(variance) if variance is not None else None
    std_phantom = np.sqrt(phantom_variance) if phantom_variance is not None else None
    
    window.results.bmode_radiomics_std = _safe_ratio(std_scan, std_phantom)


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_median")
def bmode_radiomics_median(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """PyRadiomics first-order Median, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Median"])
    key = "original_firstorder_Median"
    window.results.bmode_radiomics_median = _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_entropy")
def bmode_radiomics_entropy(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """PyRadiomics first-order Entropy, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Entropy"])
    key = "original_firstorder_Entropy"
    window.results.bmode_radiomics_entropy = _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_energy")
def bmode_radiomics_energy(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """PyRadiomics first-order Energy, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Energy"])
    key = "original_firstorder_Energy"
    window.results.bmode_radiomics_energy = _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_iqr")
def bmode_radiomics_iqr(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """PyRadiomics first-order InterquartileRange, normalised by phantom."""
    ext = _make_extractor("firstorder", ["InterquartileRange"])
    key = "original_firstorder_InterquartileRange"
    window.results.bmode_radiomics_iqr = _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


# ------------------------------------------------------------------
# GLCM (second-order) features  (scan / phantom normalised)
# ------------------------------------------------------------------
# The whole-window all-ones mask means PyRadiomics operates on the
# entire windowed signal.  Real masking is handled upstream by QuantUS.
# ------------------------------------------------------------------

@supported_spatial_dims(2, 3)
@output_vars("bmode_glcm_contrast")
def bmode_glcm_contrast(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """PyRadiomics GLCM Contrast, normalised by phantom."""
    ext = _make_extractor("glcm", ["Contrast"])
    key = "original_glcm_Contrast"
    window.results.bmode_glcm_contrast = _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


@supported_spatial_dims(2, 3)
@output_vars("bmode_glcm_homogeneity")
def bmode_glcm_homogeneity(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """PyRadiomics GLCM Idm (Homogeneity), normalised by phantom."""
    ext = _make_extractor("glcm", ["Idm"])
    key = "original_glcm_Idm"
    window.results.bmode_glcm_homogeneity = _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


@supported_spatial_dims(2, 3)
@output_vars("bmode_glcm_correlation")
def bmode_glcm_correlation(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """PyRadiomics GLCM Correlation, normalised by phantom."""
    ext = _make_extractor("glcm", ["Correlation"])
    key = "original_glcm_Correlation"
    window.results.bmode_glcm_correlation = _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


@supported_spatial_dims(2, 3)
@output_vars("bmode_glcm_energy")
def bmode_glcm_energy(
    scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
    window: Window, config: RfAnalysisConfig,
    image_data: UltrasoundRfImage, **kwargs
) -> None:
    """PyRadiomics GLCM JointEnergy, normalised by phantom."""
    ext = _make_extractor("glcm", ["JointEnergy"])
    key = "original_glcm_JointEnergy"
    window.results.bmode_glcm_energy = _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )