import numpy as np
from scipy.signal import hilbert
from scipy.special import hermite, factorial
from scipy.signal import convolve

from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.analysis import Window
from ....data_objs.image import UltrasoundRfImage

@supported_spatial_dims(2)
@output_vars("hscan_blue_channel", "hscan_red_channel", "hscan_green_channel", "central_freq_gh1", "central_freq_gh2", "wavelet_data")
@required_kwargs("gh1_order", "gh1_sigma", "gh1_wavelet_duration", "gh2_order", "gh2_sigma", "gh2_wavelet_duration")
@default_kwarg_vals(2, 0.11e-6, 3e-6, 8, 0.09e-6, 3e-6)
@location('full_segmentation')
def hscan(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
            window: Window, config: RfAnalysisConfig, 
            image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute the H-scan ratio for the ROI using Gaussian-Hermite wavelets.
    This implementation uses two different order Gaussian-Hermite wavelets to analyze
    the frequency content and scatterer properties of the ultrasound signal.

    Args:
        scan_rf_window (np.ndarray): RF data of the ROI.
        phantom_rf_window (np.ndarray): RF data of the phantom (not used in this analysis).
        window (Window): Window object to store results.
        config (RfAnalysisConfig): Configuration object for analysis.
        image_data (UltrasoundRfImage): Image data object containing metadata.
        **kwargs: Additional keyword arguments including:
            - gh1_order (int): Order of the first Gaussian-Hermite polynomial
            - gh1_sigma (float): Standard deviation of the first Gaussian window
            - gh1_wavelet_duration (float): Duration of the first wavelet in seconds
            - gh2_order (int): Order of the second Gaussian-Hermite polynomial
            - gh2_sigma (float): Standard deviation of the second Gaussian window
            - gh2_wavelet_duration (float): Duration of the second wavelet in seconds
                (same parameters as gh1)
    """
    assert scan_rf_window.ndim == 2, "scan_rf_window must be a 2D array (n lines x m samples)"
    
    def build_time_array(fs, wavelet_duration):
        """Build time array for the wavelet."""
        time_step = 1 / fs
        half_duration = wavelet_duration / 2
        return np.arange(-half_duration, half_duration, time_step)

    def build_gaussian_hermite_wavelet(order, fs, sigma, wavelet_duration):
        """Build a Gaussian-Hermite wavelet and return wavelet and its central frequency."""
        # Create time array
        time = build_time_array(fs, wavelet_duration)
        
        # Physicists' Hermite polynomial
        # https://en.wikipedia.org/wiki/Hermite_polynomials
        H_n = hermite(order)
        
        # Normalization factor
        energy = np.sqrt(np.pi / 2) * (factorial(2 * order) / (2**order * factorial(order)))
        normalization_factor = 1.0 / np.sqrt(energy)
        
        # Gaussian window
        gaussian_window = np.exp(-(time**2) / (1 * sigma**2))
        
        # Construct the wavelet
        wavelet = normalization_factor * H_n(time / sigma) * gaussian_window
        
        # Calculate central frequency
        fft_wavelet = np.fft.fft(wavelet)
        dt = 1 / fs
        freq_bins = np.fft.fftfreq(len(wavelet), d=dt)
        power_spectrum = np.abs(fft_wavelet) ** 2
        
        # Find central frequency (frequency with maximum amplitude in positive spectrum)
        positive_freqs = freq_bins > 0
        positive_freq_bins = freq_bins[positive_freqs]
        positive_power_spectrum = power_spectrum[positive_freqs]
        central_freq = positive_freq_bins[np.argmax(positive_power_spectrum)]
        
        return wavelet, central_freq, time, freq_bins, power_spectrum

    def convolve_signal_nd(signal_nd, wavelet_1d, signal_axis):
        """Convolve an n-dimensional signal with a 1D wavelet along the specified axis."""
        def convolve_1d(signal_1d):
            return convolve(signal_1d, wavelet_1d, mode='same', method='direct')
        
        # Apply convolution along the specified axis
        return np.apply_along_axis(
            func1d=convolve_1d,
            axis=signal_axis,
            arr=signal_nd
        )

    # Get required parameters
    gh1_order = kwargs['gh1_order']
    gh1_sigma = kwargs['gh1_sigma']
    gh1_wavelet_duration = kwargs['gh1_wavelet_duration']
    gh2_order = kwargs['gh2_order']
    gh2_sigma = kwargs['gh2_sigma']
    gh2_wavelet_duration = kwargs['gh2_wavelet_duration']

    # Build wavelets and get their central frequencies
    wavelet_gh1, central_freq_gh1, time_gh1, freq_bins_gh1, power_spectrum_gh1 = build_gaussian_hermite_wavelet(
        order=gh1_order,
        fs=config.sampling_frequency,
        sigma=gh1_sigma,
        wavelet_duration=gh1_wavelet_duration
    )
    
    wavelet_gh2, central_freq_gh2, time_gh2, freq_bins_gh2, power_spectrum_gh2 = build_gaussian_hermite_wavelet(
        order=gh2_order,
        fs=config.sampling_frequency,
        sigma=gh2_sigma,
        wavelet_duration=gh2_wavelet_duration
    )
        
    # Convolve signal with wavelets (signal_axis=1 for 2D RF data)
    convolved_gh1_rf = convolve_signal_nd(scan_rf_window, wavelet_gh1, signal_axis=0)
    convolved_gh2_rf = convolve_signal_nd(scan_rf_window, wavelet_gh2, signal_axis=0)
    
    convolved_gh1_ph = convolve_signal_nd(phantom_rf_window, wavelet_gh1, signal_axis=0)
    convolved_gh2_ph = convolve_signal_nd(phantom_rf_window, wavelet_gh2, signal_axis=0)
    
    # Compute envelopes using Hilbert transform
    gh1_envelope_rf = np.abs(hilbert(convolved_gh1_rf, axis=0))
    gh2_envelope_rf = np.abs(hilbert(convolved_gh2_rf, axis=0))
    gh1_envelope_ph = np.abs(hilbert(convolved_gh1_ph, axis=0))
    gh2_envelope_ph = np.abs(hilbert(convolved_gh2_ph, axis=0))
    envelope_rf = np.abs(hilbert(scan_rf_window, axis=0))
    envelope_ph = np.abs(hilbert(phantom_rf_window, axis=0))
    
    # Calculate H-scan blue channel
    hscan_blue_channel = (gh1_envelope_rf) / (gh1_envelope_ph)
    hscan_red_channel = (gh2_envelope_rf) / (gh2_envelope_ph)
    hscan_green_channel = (envelope_rf) / (envelope_ph)
   
    # Fill in attributes defined in "output_vars" decorator
    window.results.hscan_blue_channel = hscan_blue_channel
    window.results.hscan_red_channel = hscan_red_channel
    window.results.hscan_green_channel = hscan_green_channel
    window.results.central_freq_gh1 = central_freq_gh1 # MHz
    window.results.central_freq_gh2 = central_freq_gh2 # MHz
    
    # Store wavelet data for visualization
    window.results.wavelet_data = {
        'gh1': {
            'time': time_gh1,
            'wavelet': wavelet_gh1,
            'freq_bins': freq_bins_gh1,
            'power_spectrum': power_spectrum_gh1,
            'order': gh1_order,
            'sigma': gh1_sigma
        },
        'gh2': {
            'time': time_gh2,
            'wavelet': wavelet_gh2,
            'freq_bins': freq_bins_gh2,
            'power_spectrum': power_spectrum_gh2,
            'order': gh2_order,
            'sigma': gh2_sigma
        }
    }