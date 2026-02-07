import numpy as np
from scipy.signal import hilbert
import SimpleITK as sitk
from radiomics import featureextractor

# ----------------------------
# 1. Create a fake RF patch
# ----------------------------
np.random.seed(0)
rf_patch = np.random.randn(128, 32)

print("RF patch shape:", rf_patch.shape)

# ----------------------------
# 2. Envelope detection
# ----------------------------
envelope = np.abs(hilbert(rf_patch, axis=0))

print("\nEnvelope stats:")
print("min:", envelope.min())
print("max:", envelope.max())
print("mean:", envelope.mean())
print("std:", envelope.std())

# ----------------------------
# 3. Log compression (B-mode)
# ----------------------------
log_envelope = 20 * np.log10(envelope + 1e-10)

print("\nLog-envelope (B-mode) stats:")
print("min:", log_envelope.min())
print("max:", log_envelope.max())
print("mean:", log_envelope.mean())
print("std:", log_envelope.std())

# ----------------------------
import SimpleITK as sitk
from radiomics import featureextractor

image = sitk.GetImageFromArray(log_envelope.astype(np.float32))
mask = sitk.GetImageFromArray(
    np.ones_like(log_envelope, dtype=np.uint8)
)

# 🔴 CRITICAL: make image & mask compatible
mask.CopyInformation(image)

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.disableAllFeatures()
extractor.enableFeatureClassByName("firstorder")

# 🔴 explicitly tell which label to use
extractor.settings["label"] = 1

features = extractor.execute(image, mask)

window.results.bmode_radiomics_mean = features["original_firstorder_Mean"]

print("\nRadiomics results:")
print("firstorder_Mean:", features["original_firstorder_Mean"])
print("firstorder_Std:", features["original_firstorder_StandardDeviation"])
