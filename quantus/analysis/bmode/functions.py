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
@output_vars("bmode_radiomics_mean")
@supported_spatial_dims(2, 3)
def bmode_radiomics_mean(
    scan_rf_window,
    phantom_rf_window,
    window,
    config,
    image_data,
    **kwargs
):
    from scipy.signal import hilbert
    import numpy as np

    # Envelope detection
    envelope = np.abs(hilbert(scan_rf_window, axis=0))

    # Log-compressed B-mode
    log_env = 20.0 * np.log10(envelope + 1e-10)

    # Clean numerical issues
    log_env = np.nan_to_num(log_env, nan=0.0, posinf=0.0, neginf=0.0)

    # First-order mean (radiomics equivalent)
    window.results.bmode_radiomics_mean = float(np.mean(log_env))


# ------------------------------------------------------------------
# B-mode PyRadiomics First-Order Standard Deviation 
# ------------------------------------------------------------------
@output_vars("bmode_radiomics_std")
@supported_spatial_dims(2, 3)
def bmode_radiomics_std(
    scan_rf_window,
    phantom_rf_window,
    window,
    config,
    image_data,
    **kwargs
):
    from scipy.signal import hilbert
    import numpy as np

    # Envelope detection
    envelope = np.abs(hilbert(scan_rf_window, axis=0))

    # Log-compressed B-mode
    log_env = 20.0 * np.log10(envelope + 1e-10)

    # Clean numerical issues
    log_env = np.nan_to_num(log_env, nan=0.0, posinf=0.0, neginf=0.0)

    # First-order standard deviation (radiomics equivalent)
    window.results.bmode_radiomics_std = float(np.std(log_env))


# ------------------------------------------------------------------
# B-mode Radiomics First-Order Median
# ------------------------------------------------------------------
@output_vars("bmode_radiomics_median")
@supported_spatial_dims(2, 3)
def bmode_radiomics_median(
    scan_rf_window,
    phantom_rf_window,
    window,
    config,
    image_data,
    **kwargs
):

    from scipy.signal import hilbert
    import numpy as np

    # Envelope detection
    envelope = np.abs(hilbert(scan_rf_window, axis=0))

    # Log-compressed B-mode
    log_env = 20.0 * np.log10(envelope + 1e-10)

    # Clean numerical issues
    log_env = np.nan_to_num(log_env, nan=0.0, posinf=0.0, neginf=0.0)

    # Median intensity
    window.results.bmode_radiomics_median = float(np.median(log_env))

# ------------------------------------------------------------------
# B-mode Radiomics First-Order Entropy
# ------------------------------------------------------------------

@output_vars("bmode_radiomics_entropy")
@supported_spatial_dims(2, 3)
def bmode_radiomics_entropy(
    scan_rf_window,
    phantom_rf_window,
    window,
    config,
    image_data,
    **kwargs
):

    from scipy.signal import hilbert
    import numpy as np
    # Envelope detection
    envelope = np.abs(hilbert(scan_rf_window, axis=0))

    # Log-compressed B-mode
    log_env = 20.0 * np.log10(envelope + 1e-10)

    # Clean numerical issues
    log_env = np.nan_to_num(log_env, nan=0.0, posinf=0.0, neginf=0.0)

    # Histogram for entropy
    hist, _ = np.histogram(log_env.flatten(), bins=64, density=True)

    # Remove zeros
    hist = hist[hist > 0]

    entropy = -np.sum(hist * np.log2(hist))

    window.results.bmode_radiomics_entropy = float(entropy)

# ------------------------------------------------------------------
# B-mode Radiomics First-Order energy
# ------------------------------------------------------------------

@output_vars("bmode_radiomics_energy")
@supported_spatial_dims(2, 3)
def bmode_radiomics_energy(
    scan_rf_window,
    phantom_rf_window,
    window,
    config,
    image_data,
    **kwargs
):

    from scipy.signal import hilbert
    import numpy as np
    envelope = np.abs(hilbert(scan_rf_window, axis=0))

    log_env = 20.0 * np.log10(envelope + 1e-10)
    log_env = np.nan_to_num(log_env, nan=0.0, posinf=0.0, neginf=0.0)

    energy = np.sum(log_env**2)

    window.results.bmode_radiomics_energy = float(energy)

# ------------------------------------------------------------------
# B-mode Radiomics First-Order Interquartile range
# ------------------------------------------------------------------
@output_vars("bmode_radiomics_iqr")
@supported_spatial_dims(2, 3)
def bmode_radiomics_iqr(
    scan_rf_window,
    phantom_rf_window,
    window,
    config,
    image_data,
    **kwargs
):

    from scipy.signal import hilbert
    import numpy as np

    envelope = np.abs(hilbert(scan_rf_window, axis=0))

    log_env = 20.0 * np.log10(envelope + 1e-10)

    log_env = np.nan_to_num(log_env, nan=0.0, posinf=0.0, neginf=0.0)

    q75 = np.percentile(log_env, 75)
    q25 = np.percentile(log_env, 25)

    iqr = q75 - q25

    window.results.bmode_radiomics_iqr = float(iqr)



# ------------------------------------------------------------------
# B-mode Radiomics 2nd-Order GLCM Contrast (PyRadiomics)
# ------------------------------------------------------------------
@output_vars("bmode_glcm_contrast")
@supported_spatial_dims(2, 3)
def bmode_glcm_contrast(
    scan_rf_window,
    phantom_rf_window,
    window,
    config,
    image_data,
    **kwargs
):
    """
    Compute GLCM Contrast feature from log-compressed B-mode envelope
    using PyRadiomics.

    Operates on QuantUS windowed RF signals.
    """
    import numpy as np
    import SimpleITK as sitk
    from scipy.signal import hilbert
    from radiomics import featureextractor


    # Step 1 — Envelope detection from RF
    envelope = np.abs(
        hilbert(scan_rf_window, axis=0)
    )


    # Step 2 — Log compression → B-mode
    log_env = 20*np.log10(
        envelope + 1e-10
    )


    # Step 3 — Clean numerical values
    log_env = np.nan_to_num(
        log_env,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    # Step 4 — Convert to PyRadiomics image
    image = sitk.GetImageFromArray(
        log_env.astype(np.float32)
    )


    # Step 5 — Create segmentation mask
    mask_array = np.ones_like(
        log_env,
        dtype=np.uint8
    )


    # Stabilizes PyRadiomics segmentation detection
    mask_array[0,0] = 2


    mask = sitk.GetImageFromArray(mask_array)


    # Geometry must match
    mask.CopyInformation(image)


    # Step 6 — PyRadiomics extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()


    extractor.disableAllFeatures()


    extractor.enableFeaturesByName(
        glcm=["Contrast"]
    )


    # Compute only label=1 region
    extractor.settings["label"] = 1


    # Step 7 — Compute features
    features = extractor.execute(
        image,
        mask
    )


    # Step 8 — Store in QuantUS
    window.results.bmode_glcm_contrast = float(
        features["original_glcm_Contrast"]
    )


# ------------------------------------------------------------------
# B-mode Radiomics 2nd-Order GLCM Homogeneity (PyRadiomics)
# ------------------------------------------------------------------
@output_vars("bmode_glcm_homogeneity")
@supported_spatial_dims(2, 3)
def bmode_glcm_homogeneity(
    scan_rf_window,
    phantom_rf_window,
    window,
    config,
    image_data,
    **kwargs
):

    import numpy as np
    import SimpleITK as sitk
    from scipy.signal import hilbert
    from radiomics import featureextractor


    # Envelope detection
    envelope = np.abs(
        hilbert(scan_rf_window, axis=0)
    )


    # Log compression → B-mode
    log_env = 20*np.log10(
        envelope + 1e-10
    )


    # Clean values
    log_env = np.nan_to_num(
        log_env,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    image = sitk.GetImageFromArray(
        log_env.astype(np.float32)
    )


    mask_array = np.ones_like(
        log_env,
        dtype=np.uint8
    )


    # Stabilization trick (same as contrast)
    mask_array[0,0] = 2


    mask = sitk.GetImageFromArray(mask_array)

    mask.CopyInformation(image)


    extractor = featureextractor.RadiomicsFeatureExtractor()

    extractor.disableAllFeatures()

    extractor.enableFeaturesByName(
        glcm=["JointAverage"]
    )

    extractor.settings["label"] = 1


    features = extractor.execute(
        image,
        mask
    )


    window.results.bmode_glcm_homogeneity = float(
        features["original_glcm_JointAverage"]
    )

# ------------------------------------------------------------------
# B-mode Radiomics 2nd-Order GLCM Correlation (PyRadiomics)
# ------------------------------------------------------------------
@output_vars("bmode_glcm_correlation")
@supported_spatial_dims(2, 3)
def bmode_glcm_correlation(
    scan_rf_window,
    phantom_rf_window,
    window,
    config,
    image_data,
    **kwargs
):

    import numpy as np
    import SimpleITK as sitk
    from scipy.signal import hilbert
    from radiomics import featureextractor


    # Step 1 — Envelope detection
    envelope = np.abs(
        hilbert(scan_rf_window, axis=0)
    )


    # Step 2 — Log compression → B-mode
    log_env = 20*np.log10(
        envelope + 1e-10
    )


    # Step 3 — Clean numerical values
    log_env = np.nan_to_num(
        log_env,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    # Step 4 — Convert to image
    image = sitk.GetImageFromArray(
        log_env.astype(np.float32)
    )


    # Step 5 — Mask
    mask_array = np.ones_like(
        log_env,
        dtype=np.uint8
    )

    # Same stabilization trick
    mask_array[0,0] = 2

    mask = sitk.GetImageFromArray(mask_array)

    mask.CopyInformation(image)


    # Step 6 — Radiomics extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()

    extractor.disableAllFeatures()

    extractor.enableFeaturesByName(
        glcm=["Correlation"]
    )

    extractor.settings["label"] = 1


    # Step 7 — Compute
    features = extractor.execute(
        image,
        mask
    )


    # Step 8 — Store result
    window.results.bmode_glcm_correlation = float(
        features["original_glcm_Correlation"]
    )

# ------------------------------------------------------------------
# B-mode Radiomics 2nd-Order GLCM Energy (PyRadiomics)
# ------------------------------------------------------------------
@output_vars("bmode_glcm_energy")
@supported_spatial_dims(2, 3)
def bmode_glcm_energy(
    scan_rf_window,
    phantom_rf_window,
    window,
    config,
    image_data,
    **kwargs
):

    import numpy as np
    import SimpleITK as sitk
    from scipy.signal import hilbert
    from radiomics import featureextractor


    # Envelope detection
    envelope = np.abs(
        hilbert(scan_rf_window, axis=0)
    )


    # Log-compressed B-mode
    log_env = 20*np.log10(
        envelope + 1e-10
    )


    log_env = np.nan_to_num(
        log_env,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    image = sitk.GetImageFromArray(
        log_env.astype(np.float32)
    )


    mask_array = np.ones_like(
        log_env,
        dtype=np.uint8
    )

    mask_array[0,0] = 2

    mask = sitk.GetImageFromArray(mask_array)

    mask.CopyInformation(image)


    extractor = featureextractor.RadiomicsFeatureExtractor()

    extractor.disableAllFeatures()

    extractor.enableFeaturesByName(
        glcm=["JointEnergy"]
    )

    extractor.settings["label"] = 1


    features = extractor.execute(
        image,
        mask
    )


    window.results.bmode_glcm_energy = float(
        features["original_glcm_JointEnergy"]
    )
