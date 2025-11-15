import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Tuple
from scipy.signal import hilbert, stft, istft
from scipy.special import hermite, factorial
from scipy.signal import convolve
from scipy.optimize import curve_fit
from scipy.stats import linregress

from .transforms import compute_hanning_power_spec
from .decorators import *
from ...data_objs.analysis_config import RfAnalysisConfig
from ...data_objs.analysis import Window
from ...data_objs.image import UltrasoundRfImage

@supported_spatial_dims(2, 3)
@output_vars("f", "nps", "r_ps", "ps")
def compute_power_spectra(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute power spectra for a single window.
    """
    f, ps = compute_hanning_power_spec(
        scan_rf_window, config.transducer_freq_band[0],
        config.transducer_freq_band[1], config.sampling_frequency
    )
    ps = 20 * np.log10(ps)
    f, rPs = compute_hanning_power_spec(
        phantom_rf_window, config.transducer_freq_band[0],
        config.transducer_freq_band[1], config.sampling_frequency
    )
    rPs = 20 * np.log10(rPs)
    nps = np.asarray(ps) - np.asarray(rPs)
    # Fill in attributes defined in ResultsClass above
    window.results.nps = nps # dB
    window.results.f = f # Hz
    window.results.ps = ps # dB
    window.results.r_ps = rPs # dB

@supported_spatial_dims(2, 3)
@output_vars("mbf", "ss", "si")
@dependencies("compute_power_spectra")
def lizzi_feleppa(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute spectral analysis values for a single window.
    
    Args:
        scan_rf_window (np.ndarray): RF data of the window in the scan image.
        phantom_rf_window (np.ndarray): RF data of the window in the phantom image.
        window (Window): Window object to store results.
        config (RfAnalysisConfig): Configuration object for analysis.
    """
    # Accessible from above function call
    nps = window.results.nps
    f = window.results.f
    
    def compute_spectral_params(nps: np.ndarray, f: np.ndarray, 
                               low_f: int, high_f: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Perform spectral analysis on the normalized power spectrum.
        source: Lizzi et al. https://doi.org/10.1016/j.ultrasmedbio.2006.09.002
        
        Args:
            nps (np.ndarray): normalized power spectrum.
            f (np.ndarray): frequency array (Hz).
            low_f (int): lower bound of the frequency window for analysis (Hz).
            high_f (int): upper bound of the frequency window for analysis (Hz).
            
        Returns:
            Tuple: midband fit, frequency range, linear fit, and linear regression coefficients.
        """
        # 1. in one scan / run-through of data file's f array, find the data points on
        # the frequency axis closest to reference file's analysis window's LOWER bound and UPPER bounds
        smallest_diff_low_f = 999999999
        smallest_diff_high_f = 999999999

        for i in range(len(f)):
            current_diff_low_f = abs(low_f - f[i])
            current_diff_high_f = abs(high_f - f[i])

            if current_diff_low_f < smallest_diff_low_f:
                smallest_diff_low_f = current_diff_low_f
                smallest_diff_index_low_f = i

            if current_diff_high_f < smallest_diff_high_f:
                smallest_diff_high_f = current_diff_high_f
                smallest_diff_index_high_f = i

        # 2. compute linear regression within the analysis window
        f_band = f[
            smallest_diff_index_low_f:smallest_diff_index_high_f
        ]  # transpose row vector f in order for it to have same dimensions as column vector nps
        p = np.polyfit(
            f_band, nps[smallest_diff_index_low_f:smallest_diff_index_high_f], 1
        )
        nps_linfit = np.polyval(p, f_band)  # y_linfit is a column vecotr

        mbfit = p[0] * f_band[round(f_band.shape[0] / 2)] + p[1]

        return mbfit, f_band, nps_linfit, p
    
    mbf, _, _, p = compute_spectral_params(nps, f, config.analysis_freq_band[0], config.analysis_freq_band[1])
    
    # Fill in attributes defined in "output_vars" decorator
    window.results.mbf = mbf # dB
    window.results.ss = p[0]*1e6 # dB/MHz
    window.results.si = p[1] # dB

@supported_spatial_dims(2, 3)
@output_vars("att_coef")
@required_kwargs("ref_attenuation")
@default_kwarg_vals(0.7)
@dependencies("compute_power_spectra")
def attenuation_coef(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute the local attenuation coefficient of the ROI using the Spectral Difference
    Method for Local Attenuation Estimation. This method computes the attenuation coefficient
    for multiple frequencies and returns the slope of the attenuation as a function of frequency.

    Args:
        rf_data (np.ndarray): RF data of the ROI (n lines x m samples).
        ref_rf_data (np.ndarray): RF data of the phantom (n lines x m samples).
        overlap (float): Overlap percentage for analysis windows.
        window_depth (int): Depth of each window in samples.
    Updated and verified : Feb 2025 - IR
    """
    overlap = 50
    window_depth = min(100, scan_rf_window.shape[0] // 3)
    ref_attenuation = kwargs['ref_attenuation']  # Reference attenuation coefficient (dB/cm/MHz)
    
    sampling_frequency = config.sampling_frequency
    start_frequency = config.analysis_freq_band[0]
    end_frequency = config.analysis_freq_band[1]

    # Initialize arrays for storing intensities (log of power spectrum for each frequency)
    ps_sample = []  # ROI power spectra
    ps_ref = []     # Phantom power spectra

    start_idx = 0
    end_idx = window_depth
    window_center_indices = []
    counter = 0

    # Loop through the windows in the RF data
    while end_idx < scan_rf_window.shape[0]:
        sub_window_rf = scan_rf_window[start_idx:end_idx]
        f, ps = compute_hanning_power_spec(sub_window_rf, start_frequency, end_frequency, sampling_frequency)
        ps_sample.append(20 * np.log10(ps))  # Log scale intensity for the ROI

        ref_sub_window_rf = phantom_rf_window[start_idx:end_idx]
        ref_f, ref_ps = compute_hanning_power_spec(ref_sub_window_rf, start_frequency, end_frequency, sampling_frequency)
        ps_ref.append(20 * np.log10(ref_ps))  # Log scale intensity for the phantom

        window_center_indices.append((start_idx + end_idx) // 2)

        start_idx += int(window_depth * (1 - (overlap / 100)))
        end_idx = start_idx + window_depth
        counter += 1

    # Convert window depths to cm
    axial_res_cm = image_data.axial_res / 10
    window_depths_cm = np.array(window_center_indices) * axial_res_cm

    attenuation_coefficients = []  # One coefficient for each frequency

    f = f / 1e6
    ps_sample = np.array(ps_sample)
    ps_ref = np.array(ps_ref)

    mid_idx = f.shape[0] // 2
    start_idx = max(0, mid_idx - 25)
    end_idx = min(f.shape[0], mid_idx + 25)

    # Compute attenuation for each frequency
    for f_idx in range(start_idx, end_idx):
        normalized_intensities = np.subtract(ps_sample[:, f_idx], ps_ref[:, f_idx])
        p = np.polyfit(window_depths_cm, normalized_intensities, 1)
        local_attenuation = ref_attenuation * f[f_idx] - (1 / 4) * p[0]  # dB/cm
        attenuation_coefficients.append(local_attenuation / f[f_idx])  # dB/cm/MHz

    attenuation_coef = np.mean(attenuation_coefficients)
    
    # Fill in attributes defined in "output_vars" decorator
    window.results.att_coef = attenuation_coef # dB/cm/MHz

@supported_spatial_dims(2, 3)
@output_vars("bsc")
@required_kwargs("ref_bsc")
@default_kwarg_vals(0.0006)
@dependencies("compute_power_spectra", "attenuation_coef")
def bsc(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
            window: Window, config: RfAnalysisConfig, 
            image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute the backscatter coefficient of the ROI using the reference phantom method.
    Assumes instrumentation and beam terms have the same effect on the signal from both 
    image and phantom. 

    Source: Yao et al. (1990): https://doi.org/10.1177/016173469001200105. PMID: 2184569

    Args:
        freq_arr (np.ndarray): Frequency array of power spectra (Hz).
        scan_ps (np.ndarray): Power spectrum of the analyzed scan at the current region.
        ref_ps (np.ndarray): Power spectrum of the reference phantom at the current region.
        att_coef (float): Attenuation coefficient of the current region (dB/cm/MHz).
        frequency (int): Frequency on which to compute backscatter coefficient (Hz).
        roi_depth (int): Depth of the start of the ROI in samples.
        
    Returns:
        float: Backscatter coefficient of the ROI for the central frequency (1/cm-sr).
        Updated and verified : Feb 2025 - IR
    """
    freq_arr = window.results.f
    scan_ps = window.results.ps
    ref_ps = window.results.r_ps
    att_coef = window.results.att_coef
    roi_depth = scan_rf_window.shape[0]
    
    # Required kwargs
    ref_attenuation = kwargs['ref_attenuation']
    ref_backscatter_coef = kwargs['ref_bsc']
    
    # Optional kwarg
    frequency = kwargs.get('bsc_freq', config.center_frequency)  # Frequency for backscatter coefficient calculation
    
    index = np.argmin(np.abs(freq_arr - frequency))
    ps_sample = scan_ps[index]
    ps_ref = ref_ps[index]
    s_ratio = ps_sample / ps_ref

    np_conversion_factor = np.log(10) / 20 
    converted_att_coef = att_coef * np_conversion_factor  # dB/cm/MHz -> Np/cm/MHz
    converted_ref_att_coef = ref_attenuation * np_conversion_factor  # dB/cm/MHz -> Np/cm        

    window_depth_cm = roi_depth * image_data.axial_res / 10  # cm
    converted_att_coef *= frequency / 1e6  # Np/cm
    converted_ref_att_coef *= frequency / 1e6  # Np/cm        

    att_comp = np.exp(4 * window_depth_cm * (converted_att_coef - converted_ref_att_coef)) 
    bsc = s_ratio * ref_backscatter_coef * att_comp

    # Fill in attributes defined in "output_vars" decorator
    window.results.bsc = bsc # 1/cm-sr

@supported_spatial_dims(2)
@output_vars("nak_w", "nak_u")
def nakagami_params(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
            window: Window, config: RfAnalysisConfig, 
            image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute Nakagami parameters for the ROI.

    Source: Tsui, P. H., Wan, Y. L., Huang, C. C. & Wang, M. C. 
    Effect of adaptive threshold filtering on ultrasonic Nakagami 
    parameter to detect variation in scatterer concentration. Ultrason. 
    Imaging 32, 229–242 (2010). https://doi.org/10.1177%2F016173461003200403

    Args:
        rf_data (np.ndarray): RF data of the ROI (n lines x m samples).
        
    Returns:
        Tuple: Nakagami parameters (w, u) for the ROI.
    """
    r = np.abs(hilbert(scan_rf_window, axis=0))
    w = np.nanmean(r ** 2, axis=1)
    u = (w ** 2) / np.var(r ** 2, axis=1)

    # Averaging to get single parameter values
    w = np.nanmean(w)
    u = np.nanmean(u)
    
    # Fill in attributes defined in "output_vars" decorator
    window.results.nak_w = w
    window.results.nak_u = u

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
    
@supported_spatial_dims(2, 3)
@output_vars("central_freq_shift", "central_freq_scan", "central_freq_ref")
@dependencies("compute_power_spectra")
def central_freq_shift(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """
    Compute the central frequency shift between scan and reference using Gaussian fit or peak method.

    Args:
        scan_rf_window (np.ndarray): RF data of the scan window.
        phantom_rf_window (np.ndarray): RF data of the phantom window.
        window (Window): Window object to store results.
        config (RfAnalysisConfig): Configuration object for analysis.
        image_data (UltrasoundRfImage): Image data object containing metadata.
        apply_gaussian_fit (bool, optional): If True (default), fit a Gaussian to the power spectrum to find central frequency. If False, use the peak frequency directly.
    """
    freq_arr = window.results.f
    scan_ps = window.results.ps # dB
    ref_ps = window.results.r_ps # dB

    # get non db power spectra
    scan_ps_non_db = 10 ** (scan_ps / 20)
    ref_ps_non_db = 10 ** (ref_ps / 20)
    
    # Find the peak (maximum) for initial guess
    scan_peak_idx = np.argmax(scan_ps_non_db)
    ref_peak_idx = np.argmax(ref_ps_non_db)
    scan_peak_freq = freq_arr[scan_peak_idx]
    ref_peak_freq = freq_arr[ref_peak_idx]

    # fit gaussian function to power spectra and find central frequency
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    
    # Initial guesses: amplitude, center, width
    scan_p0 = [np.max(scan_ps_non_db), scan_peak_freq, 1e6]
    ref_p0 = [np.max(ref_ps_non_db), ref_peak_freq, 1e6]

    # Check for kwarg to apply gaussian fit
    apply_gaussian_fit = kwargs.get('CF_gaussian_fit')
    
    if apply_gaussian_fit:
        # Fit Gaussian to scan and reference, with robust error handling
        try:
            popt_scan, _ = curve_fit(gaussian, freq_arr, scan_ps_non_db, p0=scan_p0)
            central_freq_scan = popt_scan[1]
        except Exception as e:
            central_freq_scan = scan_peak_freq

        try:
            popt_ref, _ = curve_fit(gaussian, freq_arr, ref_ps_non_db, p0=ref_p0)
            central_freq_ref = popt_ref[1]
        except Exception as e:
            central_freq_ref = ref_peak_freq
    else:
        # Use peak frequency directly
        central_freq_scan = scan_peak_freq
        central_freq_ref = ref_peak_freq

    # compute central frequency shift
    central_freq_shift = central_freq_scan - central_freq_ref

    # store results
    window.results.central_freq_scan = central_freq_scan
    window.results.central_freq_ref = central_freq_ref
    window.results.central_freq_shift = central_freq_shift

@supported_spatial_dims(2)
@output_vars("bsc_stft_energy_dict", "bsc_stft_frequencies", "mean_central_freq_scan")
@required_kwargs("stft_ref_bsc", "stft_ref_attenuation", "stft_window", "stft_nperseg", "stft_noverlap", "stft_CF_gaussian_fit", "stft_attenuation_correction")
@default_kwarg_vals(0.0006, 0.7, "hann", 32, 16, False, True)
@location('full_segmentation')
def bsc_stft(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
            window: Window, config: RfAnalysisConfig, 
            image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute the backscatter coefficient using STFT-based power spectra calculation.
    Computes signal energy at each frequency for each position in the RF data using STFT.

    Args:
        scan_rf_window (np.ndarray): RF data of the ROI (m samples x n lines).
        phantom_rf_window (np.ndarray): RF data of the phantom (m samples x n lines).
        window (Window): Window object to store results.
        config (RfAnalysisConfig): Configuration object for analysis.
        image_data (UltrasoundRfImage): Image data object containing metadata.
        **kwargs: Additional keyword arguments including:
            - stft_ref_bsc (float): Reference backscatter coefficient (1/cm-sr)
            - stft_ref_attenuation (float): Reference attenuation coefficient (dB/cm/MHz)
            - stft_window (str): Window function for STFT
            - stft_nperseg (int): Length of each segment for STFT
            - stft_noverlap (int): Number of points to overlap between segments
            - stft_CF_gaussian_fit (bool): Whether to use Gaussian fit for central frequency calculation
            - stft_attenuation_correction (bool, optional): If True, apply attenuation correction. Defaults to True
    """
    assert scan_rf_window.ndim == 2, "scan_rf_window must be a 2D array (m samples x n lines)"
    assert phantom_rf_window.ndim == 2, "phantom_rf_window must be a 2D array (m samples x n lines)"
    
    print(f"scan_rf_window.shape: {scan_rf_window.shape}")
    print(f"phantom_rf_window.shape: {phantom_rf_window.shape}")

    # Extract STFT parameters
    stft_window = kwargs['stft_window']
    stft_nperseg = kwargs['stft_nperseg']
    stft_noverlap = kwargs['stft_noverlap']

    # Extract required parameters
    ref_backscatter_coef = kwargs['stft_ref_bsc']
    ref_attenuation = kwargs['stft_ref_attenuation']
    apply_gaussian_fit = kwargs['stft_CF_gaussian_fit']

    # Get optional parameter for applying attenuation correction
    attenuation_correction = kwargs.get('stft_attenuation_correction', True)

    # Calculate depth the same way as in bsc function
    roi_depth = scan_rf_window.shape[0]  # samples dimension
    depth_cm = roi_depth * image_data.axial_res / 100  # cm
    # print(f"depth_cm: {depth_cm}")

    # Get sampling frequency and frequency band
    sampling_frequency = config.sampling_frequency
    
    def find_central_frequency(signal_1d: np.ndarray) -> float:
        """Find central frequency for a 1D signal using FFT and peak or Gaussian fit approach."""
        
        # Compute FFT for the signal
        fft_result = np.fft.fft(signal_1d)
        
        # Calculate frequency array
        dt = 1 / sampling_frequency
        freq_array = np.fft.fftfreq(len(signal_1d), dt)
        
        # Get positive frequencies only
        positive_freq_mask = freq_array > 0
        positive_freqs = freq_array[positive_freq_mask]
        positive_fft = fft_result[positive_freq_mask]
        
        # Calculate power spectrum
        power_spectrum = np.abs(positive_fft)**2
        
        # Find the frequency with maximum power
        max_power_idx = np.argmax(power_spectrum)
        max_energy_freq = positive_freqs[max_power_idx] / 1e6  # Convert to MHz
        
        if apply_gaussian_fit:
            try:
                # Fit Gaussian to power spectrum
                def gaussian(x, a, x0, sigma):
                    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
                
                # Use frequency points as x-axis for Gaussian fit
                freq_points = positive_freqs / 1e6  # Convert to MHz
                p0 = [np.max(power_spectrum), max_energy_freq, 1.0]  # amplitude, center, width
                
                popt, _ = curve_fit(gaussian, freq_points, power_spectrum, p0=p0)
                central_freq_hz = popt[1] * 1e6  # Convert back to Hz
            except Exception as e:
                # If Gaussian fit fails, use peak frequency
                central_freq_hz = max_energy_freq * 1e6
        else:
            # Use peak frequency directly
            central_freq_hz = max_energy_freq * 1e6
        
        return central_freq_hz
    
    def calculate_stft_energy_spectrum(rf_data: np.ndarray) -> Tuple[dict, np.ndarray]:
        """
        Calculate the energy spectrum using STFT for each frequency.
        
        Args:
            rf_data (np.ndarray): Input RF data array (2D) with shape (samples, lines)
        
        Returns:
            Tuple[dict, np.ndarray]: A tuple containing:
                - Dictionary mapping frequencies to their corresponding
                  2D energy arrays with shape (lines, time_points)
                - Array of frequency values in MHz
        """
        n_samples, n_lines = rf_data.shape
        
        # First compute STFT for one line to get frequency array and time points
        f, t, Zxx_ref = stft(rf_data[:, 0], 
                            fs=sampling_frequency, 
                            window=stft_window,
                            nperseg=stft_nperseg, 
                            noverlap=stft_noverlap)
        
        n_freqs = len(f)
        n_times = Zxx_ref.shape[1]
        
        # Initialize dictionary to store frequency-energy mappings
        energy_dict = {}
        
        # For each frequency, create an energy array with STFT dimensions
        for freq_idx in range(n_freqs):
            current_freq = f[freq_idx] / 1e6  # Convert to MHz
            
            # Initialize energy array for this frequency with STFT dimensions
            energy_array = np.zeros((n_lines, n_times), dtype=float)
            
            # Process each line
            for line_idx in range(n_lines):
                # Compute STFT for this line
                _, _, Zxx = stft(rf_data[:, line_idx],
                               fs=sampling_frequency,
                               window=stft_window,
                               nperseg=stft_nperseg,
                               noverlap=stft_noverlap)
                
                # Extract power at current frequency for all time points
                energy_array[line_idx, :] = np.abs(Zxx[freq_idx, :])**2
            
            # Add energy array to dictionary with frequency as key
            energy_dict[current_freq] = energy_array
        
        # Convert frequencies to MHz
        frequencies_mhz = f / 1e6
        
        return energy_dict, frequencies_mhz
    
    def normalize_energy_with_reference(energy_dict_scan: dict, energy_dict_ref: dict) -> dict:
        """
        Normalize scan energy dictionary using reference energy dictionary.
        
        Args:
            energy_dict_scan (dict): Dictionary mapping frequencies to scan energy arrays
            energy_dict_ref (dict): Dictionary mapping frequencies to reference energy arrays
            
        Returns:
            dict: Normalized energy dictionary
        """
        normalized_dict = {}
        
        for freq in energy_dict_scan.keys():
            if freq in energy_dict_ref:
                # Normalize scan energy by reference energy
                normalized_dict[freq] = energy_dict_scan[freq] / energy_dict_ref[freq]
            else:
                # If frequency not found in reference, use scan data as is
                normalized_dict[freq] = energy_dict_scan[freq]
                
        return normalized_dict
    
    def calculate_mean_attenuation_coefficient(scan_rf_window: np.ndarray, energy_dict_scan: dict, frequencies: np.ndarray, mean_central_freq_scan: float) -> Tuple[float, np.ndarray]:
        """
        Calculate mean attenuation coefficient using pre-calculated STFT energy spectrum.
        Uses the provided mean central frequency for all lines.
        Only uses scan data, not reference data.
        
        Args:
            scan_rf_window (np.ndarray): RF data of the scan window
            energy_dict_scan (dict): Pre-calculated STFT energy spectrum for scan data
            frequencies (np.ndarray): Frequency array corresponding to energy_dict_scan
            mean_central_freq_scan (float): Mean central frequency in Hz to use for all lines
            
        Returns:
            Tuple[float, np.ndarray]: Mean attenuation coefficient in dB/cm/MHz and resampled depth array
        """

        # Create depth array for the entire RF data based on axial resolution
        # depth_cm is the total depth, we need to create an array of depth values
        n_samples = scan_rf_window.shape[0]  # samples dimension

        roi_depth_array_cm = np.linspace(0, depth_cm, n_samples)
                        
        def plot_regression_line(depth_array, log_amplitude, slope, intercept, line_idx, att_coef):
            """Plot regression line for a single line of the signal."""
            plt.figure(figsize=(10, 6))
            
            # Plot original data points
            plt.scatter(depth_array, log_amplitude, alpha=0.6, label='Data points', color='blue')
            
            # Plot regression line
            depth_range = np.linspace(depth_array[0], depth_array[-1], 100)
            regression_line = slope * depth_range + intercept
            plt.plot(depth_range, regression_line, 'r-', linewidth=2, label=f'Regression line (slope={slope:.3f})')
            
            plt.xlabel('Depth (cm)')
            plt.ylabel('Log Amplitude (dB)')
            plt.title(f'Attenuation Analysis - Line {line_idx}\nAttenuation Coefficient: {att_coef:.3f} dB/cm/MHz')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # Use the provided mean central frequency
        central_freq_scan = mean_central_freq_scan
        central_freq_mhz = mean_central_freq_scan / 1e6
               
        # Calculate attenuation coefficient for each line using the mean central frequency
        attenuation_coefficients = []
        
        for line_idx in range(scan_rf_window.shape[1]):  # lines dimension
            # Find the frequency index closest to mean central frequency
            freq_idx = np.argmin(np.abs(frequencies * 1e6 - central_freq_scan))  # Convert frequencies to Hz
            
            # Get the energy at central frequency for this line from pre-calculated spectrum
            central_freq_key = frequencies[freq_idx]
            if central_freq_key in energy_dict_scan:
                energy_at_central_freq = energy_dict_scan[central_freq_key][line_idx, :]  # Shape: (time_points,)
                
                # Convert energy to amplitude (sqrt of energy)
                amplitude_at_central_freq = np.sqrt(energy_at_central_freq)
                
                # Convert to dB scale
                log_amplitude = 20 * np.log10(amplitude_at_central_freq + 1e-10)  # Add small value to avoid log(0)
                
                # Ensure depth array matches the size of log_amplitude
                # Use ROI-specific depth array and resample to match STFT time points
                log_amplitude_n_points = len(log_amplitude)
                if log_amplitude_n_points != len(roi_depth_array_cm):
                    # Resample ROI depth array to match the number of time points
                    depth_array_resampled = np.linspace(roi_depth_array_cm[0], roi_depth_array_cm[-1], log_amplitude_n_points)
                else:
                    # If sizes match, use ROI depth array directly
                    depth_array_resampled = roi_depth_array_cm
                                 
                # Fit linear regression to get slope
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(depth_array_resampled, log_amplitude)
                    
                    # Attenuation coefficient is related to the negative slope
                    # slope: dB/cm (from linear regression of log_amplitude vs depth)
                    # central_freq_mhz: MHz (central frequency)
                    # -slope / central_freq_mhz: dB/cm/MHz (attenuation coefficient)
                    att_coef_line = -slope / central_freq_mhz  # dB/cm/MHz
                    attenuation_coefficients.append(att_coef_line)
                    
                    # Plot regression line for this line
                    #plot_regression_line(depth_array_resampled, log_amplitude, slope, intercept, line_idx, att_coef_line)
                except:
                    # If regression fails, skip this line
                    continue
        
        # Calculate mean attenuation coefficient
        if attenuation_coefficients:
            mean_att_coef = np.mean(attenuation_coefficients)
            print(f"[bsc_stft] Mean attenuation coefficient: {mean_att_coef:.6f} dB/cm/MHz")
            
            # Warn if final mean attenuation coefficient is negative
            if mean_att_coef < 0:
                warnings.warn(f"Final mean attenuation coefficient is negative: {mean_att_coef:.6f} dB/cm/MHz. Consider adjusting STFT parameters (window size, overlap, or window type) to improve signal analysis.")
            
            #print(f"depth_array_resampled: {depth_array_resampled}")
        else:
            # throw error
            raise ValueError("No attenuation coefficients could be calculated")
        
        # Return mean attenuation coefficient and resampled depth array
        # We need to return the resampled depth array that matches the STFT time points
        # Get the number of time points from the first frequency in energy_dict_scan
        
        return mean_att_coef, depth_array_resampled
    
    # Calculate energy spectra for scan and phantom data
    energy_dict_scan, frequencies = calculate_stft_energy_spectrum(scan_rf_window)
    energy_dict_phantom, _ = calculate_stft_energy_spectrum(phantom_rf_window)
    
    # Normalize scan energy using phantom reference
    energy_dict_normalized = normalize_energy_with_reference(energy_dict_scan, energy_dict_phantom)
    
    # Calculate mean central frequency from all lines before attenuation correction
    n_lines = scan_rf_window.shape[1]  # lines dimension
    central_freqs_scan = np.zeros(n_lines)
    
    for line_idx in range(n_lines):
        # Get the single line RF data
        line_rf_data = scan_rf_window[:, line_idx]
        
        # Find central frequency for this line
        central_freqs_scan[line_idx] = find_central_frequency(line_rf_data)
    
    # Calculate mean central frequency from all lines
    mean_central_freq_scan = np.mean(central_freqs_scan)
    mean_central_freq_mhz = mean_central_freq_scan / 1e6
    
    print(f"[bsc_stft] Mean central frequency from all lines: {mean_central_freq_mhz:.2f} MHz")
    print(f"[bsc_stft] Central frequency range: {np.min(central_freqs_scan/1e6):.2f} - {np.max(central_freqs_scan/1e6):.2f} MHz")
    
    # Apply attenuation correction only if requested
    if attenuation_correction:
        # Calculate mean attenuation coefficient using STFT for each line
        # depth_array_cm returned here is already ROI-specific and resampled to match STFT time points
        mean_att_coef, depth_array_cm = calculate_mean_attenuation_coefficient(scan_rf_window, energy_dict_scan, frequencies, mean_central_freq_scan)
       
        print("[bsc_stft] Applying attenuation correction...")
        # Apply attenuation correction using calculated mean attenuation coefficient
        # Convert attenuation coefficients to Nepers
        np_conversion_factor = np.log(10) / 20 
        converted_att_coef = mean_att_coef * np_conversion_factor  # dB/cm/MHz -> Np/cm/MHz
        converted_ref_attenuation = ref_attenuation * np_conversion_factor  # dB/cm/MHz -> Np/cm/MHz
        
        # Apply attenuation correction to each frequency
        for freq in energy_dict_normalized.keys():
            freq_mhz = freq
            converted_att_coef_freq = converted_att_coef * freq_mhz  # Np/cm
            converted_ref_att_coef_freq = converted_ref_attenuation * freq_mhz  # Np/cm
                            
            # Calculate attenuation compensation factor
            # depth_array_cm is ROI-specific and already resampled to match STFT time points from calculate_mean_attenuation_coefficient
            att_comp = np.exp(4 * depth_array_cm[-1] * (converted_att_coef_freq - converted_ref_att_coef_freq))
            
            #print(f"att_comp: {att_comp}")
            
            # Apply attenuation correction - broadcast att_comp to match energy array dimensions
            # att_comp shape: (time_points,) -> reshape to (1, time_points) for broadcasting
            att_comp_reshaped = att_comp.reshape(1, -1)
            energy_dict_normalized[freq] = energy_dict_normalized[freq] * att_comp_reshaped
    else:
        print("[bsc_stft] Skipping attenuation correction as requested.")
    
    # Calculate final BSC values using reference backscatter coefficient
    for freq in energy_dict_normalized.keys():
        energy_dict_normalized[freq] = energy_dict_normalized[freq] * ref_backscatter_coef
    
    # Fill in attributes defined in "output_vars" decorator
    # Transpose all arrays in the energy dictionary
    energy_dict_transposed = {}
    for freq in energy_dict_normalized.keys():
        energy_dict_transposed[freq] = energy_dict_normalized[freq].T
    
    window.results.bsc_stft_energy_dict = energy_dict_transposed
    window.results.bsc_stft_frequencies = frequencies
    window.results.mean_central_freq_scan = mean_central_freq_scan  # Store mean central frequency in Hz

