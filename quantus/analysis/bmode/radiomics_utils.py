"""
B-Mode Radiomics Analysis Functions

This module contains radiomics calculation functions for B-Mode analysis.
It leverages PyRadiomics to compute first-order and second-order (GLCM) texture features
from B-mode images derived from RF data.

All calculation functions return numeric values only. The PyQt/analysis framework
integration (setting window.results) is handled by wrapper functions in functions.py.
"""

import logging
import numpy as np
from scipy.signal import hilbert

# Configure logging
logger = logging.getLogger(__name__)

# Try to import radiomics features, but make it optional
try:
    import SimpleITK as sitk
    from radiomics import featureextractor
    RADIOMICS_AVAILABLE = True
    logger.debug("Successfully imported PyRadiomics and SimpleITK.")
except ImportError as e:
    logger.warning(f"PyRadiomics not available. B-Mode radiomics features will not work: {e}")
    RADIOMICS_AVAILABLE = False


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


if RADIOMICS_AVAILABLE:
    def _build_radiomics_image_and_mask(log_env: np.ndarray, image_data):
        """
        Wrap a log-envelope array as a pair of SimpleITK objects suitable for
        PyRadiomics. The mask is all-ones (label = 1), covering the entire
        window — real spatial masking is handled upstream by the QuantUS toolbox.
        
        A single voxel is set to label 2 at position [0,0] as a technical 
        requirement: PyRadiomics internally validates that more than one label 
        value exists in the mask before recognising label 1 as a valid region. 
        This does not spatially restrict the computation in any way.

        Spacing is correctly set based on the UltrasoundRfImage resolutions:
        - 2D: (lateral_res, axial_res) 
        - 3D: (lateral_res, axial_res, coronal_res)
        Note: SimpleITK treats the first dimension as 'x' (column index).
        """
        logger.debug(f"Building SITK image/mask. Input shape: {log_env.shape}")
        image = sitk.GetImageFromArray(log_env.astype(np.float32))
        
        # Set spatial spacing to ensure GLCM distances reflect physical distance (mm)
        if hasattr(image_data, 'spatial_dims') and image_data.spatial_dims == 3:
            spacing = (image_data.lateral_res, image_data.axial_res, image_data.coronal_res)
            logger.debug(f"3D Spacing (Lat, Ax, Cor): {spacing}")
        else:
            spacing = (image_data.lateral_res, image_data.axial_res)
            logger.debug(f"2D Spacing (Lat, Ax): {spacing}")
        
        image.SetSpacing(spacing)

        # PyRadiomics/SimpleITK indexing note:
        # np.array shape (Z, Y, X) -> SITK image [X, Y, Z]
        # For 2D: (Axial=Y, Lateral=X) -> Spacing (Lat_Res, Ax_Res)
        # For 3D: (Coronal=Z, Lateral=Y, Axial=X) -> Spacing (Ax_Res, Lat_Res, Cor_Res)
        # We ensure consistency with the input shape from framework.py
        
        mask_array = np.ones_like(log_env, dtype=np.uint8)
        mask_array.flat[0] = 2  # sentinel: satisfies PyRadiomics label validation
        mask = sitk.GetImageFromArray(mask_array)
        mask.CopyInformation(image)
        return image, mask


    def _make_extractor(feature_class: str, feature_names: list, bin_width: float = 1.0):
        """
        Return a RadiomicsFeatureExtractor configured to compute only the
        requested features on label = 1.  Resampling is disabled to prevent
        mask/image geometry mismatches on small windowed arrays.

        We use a fixed binWidth (default 1.0 dB) for log-compressed ultrasound 
        data to ensure textures are captured with sufficient resolution (~60-100 bins).
        """
        logger.debug(f"Creating extractor for {feature_class} with features {feature_names} (binWidth: {bin_width})")
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeaturesByName(**{feature_class: feature_names})
        extractor.settings["label"] = 1
        extractor.settings["resampledPixelSpacing"] = None
        extractor.settings["interpolator"] = sitk.sitkNearestNeighbor
        
        # Consistent binning for log-compressed (dB) data
        extractor.settings["binWidth"] = bin_width
        
        return extractor


    def _extract_feature(extractor, rf_window: np.ndarray, image_data, key: str):
        """
        Run PyRadiomics on a single RF window and return the named feature value.
        Returns None if rf_window is None (e.g. phantom not provided by QuantUS).
        """
        if rf_window is None:
            logger.debug("RF window is None, skipping extraction.")
            return None
        
        log_env = _get_log_envelope(rf_window)
        image, mask = _build_radiomics_image_and_mask(log_env, image_data)
        
        try:
            result = extractor.execute(image, mask)[key]
            logger.debug(f"Extracted feature {key}: {result}")
            return float(result)
        except Exception as e:
            logger.error(f"Failed to extract radiomics feature {key}: {e}")
            return None


    def _safe_ratio(scan_val: float, phantom_val, eps: float = 1e-10) -> float:
        """
        Divide scan feature by phantom feature, guarding against near-zero
        denominators.  If phantom_val is None (phantom window unavailable),
        returns the raw scan value instead of a normalised ratio.
        """
        if scan_val is None:
            return None
            
        if phantom_val is None:
            logger.debug(f"Scan value: {scan_val}. No phantom available.")
            return float(scan_val)
        
        ratio = float(scan_val / (phantom_val + eps))
        logger.debug(f"Scan: {scan_val}, Phantom: {phantom_val}, Normalized Ratio: {ratio}")
        return ratio

else:
    # Stub implementations when radiomics is not available
    def _build_radiomics_image_and_mask(log_env: np.ndarray, image_data):
        raise RuntimeError("PyRadiomics is not installed. Cannot compute radiomics features.")

    def _make_extractor(feature_class: str, feature_names: list, bin_width: float = 1.0):
        raise RuntimeError("PyRadiomics is not installed. Cannot compute radiomics features.")

    def _extract_feature(extractor, rf_window: np.ndarray, image_data, key: str):
        raise RuntimeError("PyRadiomics is not installed. Cannot compute radiomics features.")

    def _safe_ratio(scan_val: float, phantom_val, eps: float = 1e-10) -> float:
        raise RuntimeError("PyRadiomics is not installed. Cannot compute radiomics features.")


# ------------------------------------------------------------------
# Calculation functions - return values instead of setting window.results
# ------------------------------------------------------------------
# Used by wrapper functions in functions.py to get computed values
# ------------------------------------------------------------------

def calc_radiomics_mean(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, image_data) -> float:
    """Calculate PyRadiomics first-order Mean, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Mean"])
    key = "original_firstorder_Mean"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, image_data, key),
        _extract_feature(ext, phantom_rf_window, image_data, key),
    )


def calc_radiomics_std(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, image_data) -> float:
    """Calculate PyRadiomics first-order Standard Deviation, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Variance"])
    key = "original_firstorder_Variance"
    variance = _extract_feature(ext, scan_rf_window, image_data, key)
    phantom_variance = _extract_feature(ext, phantom_rf_window, image_data, key)
    
    # Convert variance → standard deviation
    std_scan = np.sqrt(variance) if variance is not None else None
    std_phantom = np.sqrt(phantom_variance) if phantom_variance is not None else None
    
    return _safe_ratio(std_scan, std_phantom)


def calc_radiomics_median(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, image_data) -> float:
    """Calculate PyRadiomics first-order Median, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Median"])
    key = "original_firstorder_Median"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, image_data, key),
        _extract_feature(ext, phantom_rf_window, image_data, key),
    )


def calc_radiomics_entropy(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, image_data) -> float:
    """Calculate PyRadiomics first-order Entropy, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Entropy"])
    key = "original_firstorder_Entropy"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, image_data, key),
        _extract_feature(ext, phantom_rf_window, image_data, key),
    )


def calc_radiomics_energy(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, image_data) -> float:
    """Calculate PyRadiomics first-order Energy, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Energy"])
    key = "original_firstorder_Energy"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, image_data, key),
        _extract_feature(ext, phantom_rf_window, image_data, key),
    )


def calc_radiomics_iqr(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, image_data) -> float:
    """Calculate PyRadiomics first-order InterquartileRange, normalised by phantom."""
    ext = _make_extractor("firstorder", ["InterquartileRange"])
    key = "original_firstorder_InterquartileRange"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, image_data, key),
        _extract_feature(ext, phantom_rf_window, image_data, key),
    )


def calc_glcm_contrast(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, image_data) -> float:
    """Calculate PyRadiomics GLCM Contrast, normalised by phantom."""
    ext = _make_extractor("glcm", ["Contrast"])
    key = "original_glcm_Contrast"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, image_data, key),
        _extract_feature(ext, phantom_rf_window, image_data, key),
    )


def calc_glcm_homogeneity(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, image_data) -> float:
    """Calculate PyRadiomics GLCM Homogeneity, normalised by phantom."""
    ext = _make_extractor("glcm", ["Idm"])
    key = "original_glcm_Idm"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, image_data, key),
        _extract_feature(ext, phantom_rf_window, image_data, key),
    )


def calc_glcm_correlation(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, image_data) -> float:
    """Calculate PyRadiomics GLCM Correlation, normalised by phantom."""
    ext = _make_extractor("glcm", ["Correlation"])
    key = "original_glcm_Correlation"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, image_data, key),
        _extract_feature(ext, phantom_rf_window, image_data, key),
    )


def calc_glcm_energy(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, image_data) -> float:
    """Calculate PyRadiomics GLCM JointEnergy, normalised by phantom."""
    ext = _make_extractor("glcm", ["JointEnergy"])
    key = "original_glcm_JointEnergy"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, image_data, key),
        _extract_feature(ext, phantom_rf_window, image_data, key),
    )


# ------------------------------------------------------------------
# Architecture Note:
# ------------------------------------------------------------------
# The decorated functions (with @supported_spatial_dims and @output_vars)
# are now implemented as wrapper functions in functions.py.
# These wrappers call the calc_* functions above and set window.results.
# This radiomics.py module provides pure calculation logic only.
