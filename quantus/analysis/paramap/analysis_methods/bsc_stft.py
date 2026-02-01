import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Tuple
from scipy.signal import stft
from scipy.optimize import curve_fit
from scipy.stats import linregress

from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.analysis import Window
from ....data_objs.image import UltrasoundRfImage

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