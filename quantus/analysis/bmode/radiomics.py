"""
B-Mode Radiomics Analysis Functions

This module contains radiomics calculation functions for B-Mode analysis.
It leverages PyRadiomics to compute first-order and second-order (GLCM) texture features
from B-mode images derived from RF data.

All calculation functions return numeric values only. The PyQt/analysis framework
integration (setting window.results) is handled by wrapper functions in functions.py.
"""

import numpy as np
import SimpleITK as sitk
from scipy.signal import hilbert
from radiomics import featureextractor


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
# Calculation functions - return values instead of setting window.results
# ------------------------------------------------------------------
# Used by wrapper functions in functions.py to get computed values
# ------------------------------------------------------------------

def calc_radiomics_mean(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray) -> float:
    """Calculate PyRadiomics first-order Mean, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Mean"])
    key = "original_firstorder_Mean"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


def calc_radiomics_std(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray) -> float:
    """Calculate PyRadiomics first-order Standard Deviation, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Variance"])
    key = "original_firstorder_Variance"
    variance = _extract_feature(ext, scan_rf_window, key)
    phantom_variance = _extract_feature(ext, phantom_rf_window, key)
    
    # Convert variance → standard deviation
    std_scan = np.sqrt(variance) if variance is not None else None
    std_phantom = np.sqrt(phantom_variance) if phantom_variance is not None else None
    
    return _safe_ratio(std_scan, std_phantom)


def calc_radiomics_median(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray) -> float:
    """Calculate PyRadiomics first-order Median, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Median"])
    key = "original_firstorder_Median"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


def calc_radiomics_entropy(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray) -> float:
    """Calculate PyRadiomics first-order Entropy, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Entropy"])
    key = "original_firstorder_Entropy"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


def calc_radiomics_energy(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray) -> float:
    """Calculate PyRadiomics first-order Energy, normalised by phantom."""
    ext = _make_extractor("firstorder", ["Energy"])
    key = "original_firstorder_Energy"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


def calc_radiomics_iqr(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray) -> float:
    """Calculate PyRadiomics first-order InterquartileRange, normalised by phantom."""
    ext = _make_extractor("firstorder", ["InterquartileRange"])
    key = "original_firstorder_InterquartileRange"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


def calc_glcm_contrast(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray) -> float:
    """Calculate PyRadiomics GLCM Contrast, normalised by phantom."""
    ext = _make_extractor("glcm", ["Contrast"])
    key = "original_glcm_Contrast"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


def calc_glcm_homogeneity(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray) -> float:
    """Calculate PyRadiomics GLCM Homogeneity, normalised by phantom."""
    ext = _make_extractor("glcm", ["Idm"])
    key = "original_glcm_Idm"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


def calc_glcm_correlation(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray) -> float:
    """Calculate PyRadiomics GLCM Correlation, normalised by phantom."""
    ext = _make_extractor("glcm", ["Correlation"])
    key = "original_glcm_Correlation"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


def calc_glcm_energy(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray) -> float:
    """Calculate PyRadiomics GLCM JointEnergy, normalised by phantom."""
    ext = _make_extractor("glcm", ["JointEnergy"])
    key = "original_glcm_JointEnergy"
    return _safe_ratio(
        _extract_feature(ext, scan_rf_window, key),
        _extract_feature(ext, phantom_rf_window, key),
    )


# ------------------------------------------------------------------
# Architecture Note:
# ------------------------------------------------------------------
# The decorated functions (with @supported_spatial_dims and @output_vars)
# are now implemented as wrapper functions in functions.py.
# These wrappers call the calc_* functions above and set window.results.
# This radiomics.py module provides pure calculation logic only.
