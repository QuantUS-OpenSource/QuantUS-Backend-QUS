import numpy as np
from typing import Tuple
from scipy.signal import zoom_fft

def compute_power_spec(
    rf_data: np.ndarray,
    start_frequency: float,
    end_frequency: float,
    sampling_frequency: float,
    n_fft: int,
    use_hanning: bool = True,
    average_lines: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the power spectrum of 3D spatial RF data using a Hanning window.
    Vectorized implementation using zoom FFT.

    Args:
        rf_data: 3D RF data (axial x lateral x elevational) or 2D (axial x lateral)
        start_frequency: lower bound of frequency range (Hz)
        end_frequency: upper bound of frequency range (Hz)
        sampling_frequency: RF sampling frequency (Hz)
        n_fft: number of frequency samples in analysis band

    Returns:
        freqs: frequency vector
        ps: averaged power spectrum
    """

    # Ensure 3D for unified processing
    if rf_data.ndim == 2:
        rf_data = rf_data[:, :, None]

    axial, lateral, elev = rf_data.shape

    if use_hanning:
        # Hanning window (axial dimension)
        window = np.hanning(axial)
        window = window * np.sqrt(len(window) / np.sum(window**2))

        # Apply window via broadcasting
        rf_windowed = rf_data * window[:, None, None]

        # Reshape to batch all lines
        rf_reshaped = rf_windowed.reshape(axial, lateral * elev)
    else:
        rf_reshaped = rf_data

    # Zoom FFT (equivalent to MATLAB zfft)
    Z = zoom_fft(
        rf_reshaped,
        [start_frequency, end_frequency],
        m=n_fft,
        fs=sampling_frequency,
        axis=0,
        endpoint=True
    )

    # Power spectrum
    PS = np.abs(Z) ** 2

    # Average across all spatial lines
    if average_lines:
        ps = np.squeeze(PS.mean(axis=1))
    else:
        ps = np.squeeze(PS)

    # Frequency vector
    freqs = np.linspace(start_frequency, end_frequency, n_fft, endpoint=True)

    return freqs, ps
