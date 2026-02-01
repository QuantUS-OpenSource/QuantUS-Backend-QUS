from pathlib import Path

import matplotlib.pyplot as plt

from ..decorators import *
from ....data_objs.analysis import ParamapAnalysisBase

@dependencies('hscan')
def plot_hscan_wavelets(analysis_obj: ParamapAnalysisBase, dest_folder: str, **kwargs) -> None:
    """Plot the wavelets and their frequency spectra used in H-scan analysis.
    
    Args:
        analysis_obj (ParamapAnalysisBase): Analysis object containing the wavelet data
        dest_folder (str): Directory to save the plots
    """
    print(f"[DEBUG] plot_hscan_wavelets called with dest_folder: {dest_folder}")
    
    if not Path(dest_folder).is_dir():
        print(f"[ERROR] plot_hscan_wavelets: Output folder doesn't exist: {dest_folder}")
        return
        
    if not hasattr(analysis_obj.single_window.results, 'wavelet_data'):
        print(f"[ERROR] Wavelet data not found in results")
        print(f"[DEBUG] Available results attributes: {dir(analysis_obj.single_window.results)}")
        return
        
    print(f"[DEBUG] Wavelet data found successfully")
    
    wavelet_data = analysis_obj.single_window.results.wavelet_data
    
    for gh_key, data in wavelet_data.items():
        fig = plt.figure(figsize=(12, 4))
        
        # Plot the Hermite wavelet
        plt.subplot(1, 2, 1)
        time_us = data['time'] * 1e6  # Convert to microseconds
        plt.plot(time_us, data['wavelet'], label=f'GH{data["order"]}')
        
        # Plot vertical lines at ±3σ
        plt.axvline(x=(3 * data['sigma']) * 1e6, color='blue', linestyle='--', label='+3σ')
        plt.axvline(x=(-3 * data['sigma']) * 1e6, color='blue', linestyle='--', label='-3σ')
        
        plt.title('Hermite Wavelet')
        plt.xlabel('Time [µs]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        # Plot frequency spectrum
        plt.subplot(1, 2, 2)
        # Only plot positive frequencies
        positive_mask = data['freq_bins'] >= 0
        plt.plot(data['freq_bins'][positive_mask] / 1e6, data['power_spectrum'][positive_mask], 
                label=f'FFT of GH{data["order"]}')
        
        plt.title('Positive Frequency Spectrum')
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        fig.savefig(Path(dest_folder) / f'wavelet_{gh_key}_spectrum.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)