import numpy as np

from quantus.analysis.bmode.functions import (
    bmode_radiomics_mean_wrapper as bmode_radiomics_mean,
    bmode_radiomics_std_wrapper as bmode_radiomics_std,
    bmode_radiomics_median_wrapper as bmode_radiomics_median,
    bmode_radiomics_entropy_wrapper as bmode_radiomics_entropy,
    bmode_radiomics_energy_wrapper as bmode_radiomics_energy,
    bmode_glcm_contrast_wrapper as bmode_glcm_contrast,
    bmode_glcm_homogeneity_wrapper as bmode_glcm_homogeneity,
    bmode_glcm_correlation_wrapper as bmode_glcm_correlation,
    bmode_glcm_energy_wrapper as bmode_glcm_energy,
)


# Minimal dummy objects
class DummyWindow:
    class Results:
        pass
    results = Results()


class DummyConfig:
    pass


class DummyImage:
    pass


def test_bmode_radiomics_features():

    np.random.seed(0)

    # dummy RF patches
    scan_rf_window = np.random.randn(128, 32).astype(np.float32)

    phantom_rf_window = np.random.randn(128, 32).astype(np.float32)


    window = DummyWindow()
    config = DummyConfig()
    image_data = DummyImage()


    # -------- First Order --------

    bmode_radiomics_mean(
        scan_rf_window,
        phantom_rf_window,
        window,
        config,
        image_data
    )

    bmode_radiomics_std(
        scan_rf_window,
        phantom_rf_window,
        window,
        config,
        image_data
    )

    bmode_radiomics_median(
        scan_rf_window,
        phantom_rf_window,
        window,
        config,
        image_data
    )

    bmode_radiomics_entropy(
        scan_rf_window,
        phantom_rf_window,
        window,
        config,
        image_data
    )

    bmode_radiomics_energy(
        scan_rf_window,
        phantom_rf_window,
        window,
        config,
        image_data
    )


    # -------- GLCM --------

    bmode_glcm_contrast(
        scan_rf_window,
        phantom_rf_window,
        window,
        config,
        image_data
    )

    bmode_glcm_homogeneity(
        scan_rf_window,
        phantom_rf_window,
        window,
        config,
        image_data
    )

    bmode_glcm_correlation(
        scan_rf_window,
        phantom_rf_window,
        window,
        config,
        image_data
    )

    bmode_glcm_energy(
        scan_rf_window,
        phantom_rf_window,
        window,
        config,
        image_data
    )


    print("\nFIRST ORDER FEATURES")

    print("Mean:", window.results.bmode_radiomics_mean)
    print("Std:", window.results.bmode_radiomics_std)
    print("Median:", window.results.bmode_radiomics_median)
    print("Entropy:", window.results.bmode_radiomics_entropy)
    print("Energy:", window.results.bmode_radiomics_energy)


    print("\nGLCM FEATURES")

    print("Contrast:", window.results.bmode_glcm_contrast)
    print("Homogeneity:", window.results.bmode_glcm_homogeneity)
    print("Correlation:", window.results.bmode_glcm_correlation)
    print("Joint Energy:", window.results.bmode_glcm_energy)


if __name__ == "__main__":

    test_bmode_radiomics_features()