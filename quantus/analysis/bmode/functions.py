import numpy as np
from ..paramap.decorators import supported_spatial_dims, output_vars, dependencies
from ...data_objs.analysis_config import RfAnalysisConfig
from ...data_objs.analysis import Window
from ...data_objs.image import UltrasoundRfImage

# -------------------------------------------------
# Helper: convert 2D window to single-slice 3D volume for PyRadiomics compatibility (Z=1)
# -------------------------------------------------
def _make_fake_3d(arr: np.ndarray) -> np.ndarray:
    """
    Convert a 2D array (H, W) into a single-slice 3D array (1, H, W),
    required by PyRadiomics on some platforms (e.g. Windows).
    """

    return arr[np.newaxis, :, :]


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

#adding new function
# ------------------------------------------------------------------
# B-mode PyRadiomics First-Order Mean
# ------------------------------------------------------------------

@dependencies("scipy", "SimpleITK", "radiomics")
@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_mean")
def bmode_radiomics_mean(
    scan_rf_window: np.ndarray,
    phantom_rf_window: np.ndarray,
    window: Window,
    config: RfAnalysisConfig,
    image_data: UltrasoundRfImage,
    **kwargs
) -> None:

    from scipy.signal import hilbert
    import numpy as np
    import SimpleITK as sitk
    from radiomics import featureextractor

    # Envelope → log-compressed B-mode
    envelope = np.abs(hilbert(scan_rf_window, axis=0))
    log_env = 20.0 * np.log10(envelope + 1e-10)

    # Clean NaN / inf (MANDATORY)
    log_env = np.nan_to_num(log_env, nan=0.0, posinf=0.0, neginf=0.0)

    # Expand to single-slice 3D to satisfy PyRadiomics dimensional requirements
    log_env_3d = _make_fake_3d(log_env)
    mask_3d = np.ones_like(log_env_3d, dtype=np.uint8)

    # SimpleITK objects
    image = sitk.GetImageFromArray(log_env_3d.astype(np.float32))
    mask = sitk.GetImageFromArray(mask_3d)

    # Explicit geometry (prevents metadata mismatch)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ))

    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask.CopyInformation(image)

    # PyRadiomics extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings["label"] = 1
    extractor.settings["resampledPixelSpacing"] = None
    extractor.settings["normalize"] = False

    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")

    features = extractor.execute(image, mask)

    window.results.bmode_radiomics_mean = float(
        features["original_firstorder_Mean"]
    )


# ------------------------------------------------------------------
# B-mode PyRadiomics First-Order Standard Deviation
# ------------------------------------------------------------------

@dependencies("scipy", "SimpleITK", "radiomics")
@supported_spatial_dims(2, 3)
@output_vars("bmode_radiomics_std")
def bmode_radiomics_std(
    scan_rf_window: np.ndarray,
    phantom_rf_window: np.ndarray,
    window: Window,
    config: RfAnalysisConfig,
    image_data: UltrasoundRfImage,
    **kwargs
) -> None:

    from scipy.signal import hilbert
    import numpy as np
    import SimpleITK as sitk
    from radiomics import featureextractor

    # Envelope → log-compressed B-mode
    envelope = np.abs(hilbert(scan_rf_window, axis=0))
    log_env = 20.0 * np.log10(envelope + 1e-10)

    # Clean NaN / inf
    log_env = np.nan_to_num(log_env, nan=0.0, posinf=0.0, neginf=0.0)

    # Expand to single-slice 3D to satisfy PyRadiomics dimensional requirements
    log_env_3d = _make_fake_3d(log_env)
    mask_3d = np.ones_like(log_env_3d, dtype=np.uint8)

    image = sitk.GetImageFromArray(log_env_3d.astype(np.float32))
    mask = sitk.GetImageFromArray(mask_3d)

    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ))

    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask.CopyInformation(image)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings["label"] = 1
    extractor.settings["resampledPixelSpacing"] = None
    extractor.settings["normalize"] = False

    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")

    features = extractor.execute(image, mask)

    window.results.bmode_radiomics_std = float(
        features["original_firstorder_StandardDeviation"]
    )
