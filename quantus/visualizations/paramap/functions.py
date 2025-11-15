from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from ...image_loading.utc_loaders.transforms import scanConvert
import cv2

from .decorators import *
from ...data_objs.analysis import ParamapAnalysisBase

@dependencies('compute_power_spectra')
def plot_ps_window_data(analysis_obj: ParamapAnalysisBase, dest_folder: str, **kwargs) -> None:
    """Plots the power spectrum data for each window in the ROI.

    The power spectrum data is plotted along with the average power spectrum and a line of best fit
    used for the midband fit, spectral slope, and spectral intercept calculations. Also plots the 
    frequency band used for analysis.
    """
    
    assert Path(dest_folder).is_dir(), "plot_ps_data visualization: Power spectrum plot output folder doesn't exist"
    assert hasattr(analysis_obj.windows[0].results, 'ss'), "Spectral slope not found in results"
    assert hasattr(analysis_obj.windows[0].results, 'si'), "Spectral intercept not found in results"
    assert hasattr(analysis_obj.windows[0].results, 'nps'), "Normalized power spectrum not found in results"
    assert hasattr(analysis_obj.windows[0].results, 'f'), "Frequency not found in results"
    
    ss_arr = [window.results.ss for window in analysis_obj.windows]
    si_arr = [window.results.si for window in analysis_obj.windows]
    nps_arr = [window.results.nps for window in analysis_obj.windows]

    fig, ax = plt.subplots()

    ss_mean = np.mean(np.array(ss_arr)/1e6)
    si_mean = np.mean(si_arr)
    nps_arr = [window.results.nps for window in analysis_obj.windows]
    av_nps = np.mean(nps_arr, axis=0)
    f = analysis_obj.windows[0].results.f
    x = np.linspace(min(f), max(f), 100)
    y = ss_mean*x + si_mean

    for nps in nps_arr[:-1]:
        ax.plot(f/1e6, nps, c="b", alpha=0.2)
    ax.plot(f/1e6, nps_arr[-1], c="b", alpha=0.2, label="Window NPS")
    ax.plot(f/1e6, av_nps, color="r", label="Av NPS")
    ax.plot(x/1e6, y, c="orange", label="Av LOBF")
    ax.plot(2*[analysis_obj.config.analysis_freq_band[0]/1e6], [np.amin(nps_arr), np.amax(nps_arr)], c="purple")
    ax.plot(2*[analysis_obj.config.analysis_freq_band[1]/1e6], [np.amin(nps_arr), np.amax(nps_arr)], c="purple", label="Analysis Band")
    ax.set_title("Normalized Power Spectra")
    ax.legend()
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_ylim([np.amin(nps_arr), np.amax(nps_arr)])
    ax.set_xlim([min(f)/1e6, max(f)/1e6])
    
    fig.savefig(Path(dest_folder) / 'nps_plot.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

@dependencies('hscan')
def plot_hscan_result(analysis_obj: ParamapAnalysisBase, dest_folder: str, **kwargs) -> None:
    """Plots the H-scan results overlaid on the B-mode image.

    The H-scan results consist of two channels (red and blue) that represent different
    frequency components and scatterer properties in the ultrasound signal. These are overlaid
    on the B-mode image to provide anatomical context. Red channel represents low frequency
    components while blue channel represents high frequency components.
    """
    print(f"[DEBUG] plot_hscan_result called with dest_folder: {dest_folder}")
    
    if not Path(dest_folder).is_dir():
        print(f"[ERROR] H-scan plot output folder doesn't exist: {dest_folder}")
        return
    
    if not hasattr(analysis_obj.single_window.results, 'hscan_blue_channel'):
        print(f"[ERROR] H-scan blue channel not found in results")
        print(f"[DEBUG] Available results attributes: {dir(analysis_obj.single_window.results)}")
        return
        
    if not hasattr(analysis_obj.single_window.results, 'hscan_red_channel'):
        print(f"[ERROR] H-scan red channel not found in results")
        print(f"[DEBUG] Available results attributes: {dir(analysis_obj.single_window.results)}")
        return
        
    print(f"[DEBUG] H-scan channels found successfully")
    
    # Get the B-mode image and convert to RGB
    if analysis_obj.image_data.sc_bmode is not None:
        bmode = cv2.cvtColor(np.array(analysis_obj.image_data.sc_bmode, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        bmode = cv2.cvtColor(np.array(analysis_obj.image_data.bmode, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Get H-scan results
    blue_channel = analysis_obj.single_window.results.hscan_blue_channel
    red_channel = analysis_obj.single_window.results.hscan_red_channel
    
    # Create full-size channel images with ROI data
    blue_full = np.zeros_like(analysis_obj.image_data.bmode, dtype=np.float32)
    red_full = np.zeros_like(analysis_obj.image_data.bmode, dtype=np.float32)
    
    # Get single window boundaries
    single_window = analysis_obj.single_window
    
    # Map the H-scan data to the full image size using the window boundaries
    ax_slice = slice(single_window.ax_min, single_window.ax_max + 1)
    lat_slice = slice(single_window.lat_min, single_window.lat_max + 1)
    blue_full[ax_slice, lat_slice] = blue_channel
    red_full[ax_slice, lat_slice] = red_channel
    
    # Get the appropriate segmentation mask
    mask = analysis_obj.seg_data.seg_mask
    if mask.ndim == 3:  # Handle 3D case if needed
        mask = mask[:, :, analysis_obj.seg_data.frame]
           
    # Calculate normalization ranges using percentiles to handle outliers
    # Only consider values within the ROI for normalization
    valid_blue = blue_full[mask > 0]
    valid_red = red_full[mask > 0]
    blue_min, blue_max = np.percentile(valid_blue[valid_blue != 0], [5, 95])
    red_min, red_max = np.percentile(valid_red[valid_red != 0], [5, 95])
    
    # Scan convert all the data if needed
    if analysis_obj.image_data.sc_bmode is not None:
        image_data = analysis_obj.image_data
        if image_data.sc_bmode.ndim == 2:
            # Scan convert the channel data
            final_blue = scanConvert(blue_full, image_data.width, image_data.tilt,
                                   image_data.start_depth, image_data.end_depth,
                                   desiredHeight=image_data.sc_bmode.shape[0])[0].scArr
            
            final_red = scanConvert(red_full, image_data.width, image_data.tilt,
                                  image_data.start_depth, image_data.end_depth,
                                  desiredHeight=image_data.sc_bmode.shape[0])[0].scArr
            
            # Scan convert the mask
            final_mask = scanConvert(mask.astype(np.float32), image_data.width, image_data.tilt,
                                   image_data.start_depth, image_data.end_depth,
                                   desiredHeight=image_data.sc_bmode.shape[0])[0].scArr
            
            # Threshold the scan-converted mask to ensure binary values
            final_mask = (final_mask > 0.5).astype(np.float32)
        else:
            raise NotImplementedError("3D scan conversion not yet implemented for H-scan visualization")
    else:
        final_blue = blue_full
        final_red = red_full
        final_mask = mask.astype(np.float32)
    
    # Normalize each channel
    final_blue_norm = np.clip((final_blue - blue_min) / (blue_max - blue_min), 0, 1)
    final_red_norm = np.clip((final_red - red_min) / (red_max - red_min), 0, 1)
    
    # Create RGBA overlay
    overlay = np.zeros((*bmode.shape[:2], 4), dtype=np.float32)
    
    # Set RGB values where mask is active
    overlay[..., 0] = final_red_norm * final_mask  # Red channel
    overlay[..., 2] = final_blue_norm * final_mask  # Blue channel
    overlay[..., 3] = final_mask * 0.5  # Alpha channel (50% transparency where mask is active)
    
    # Calculate proper aspect ratio based on resolution
    if analysis_obj.image_data.sc_bmode is not None:
        width = bmode.shape[1] * analysis_obj.image_data.sc_lateral_res
        height = bmode.shape[0] * analysis_obj.image_data.sc_axial_res
    else:
        width = bmode.shape[1] * analysis_obj.image_data.lateral_res
        height = bmode.shape[0] * analysis_obj.image_data.axial_res
    aspect = width/height
    
    # Create figure for main overlay
    fig, ax = plt.subplots()
    
    # Plot B-mode image and overlay
    im = ax.imshow(bmode)
    im = ax.imshow(overlay)
    extent = im.get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
    ax.axis('off')
    
    # Save the main overlay figure without any padding
    fig.savefig(Path(dest_folder) / 'hscan_overlay.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Create and save red channel legend
    fig_red, ax_red = plt.subplots(figsize=(2, 10))
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax_red.imshow(gradient, aspect='auto', cmap=plt.cm.Reds)
    ax_red.set_yticks(np.linspace(0, 255, 6))
    ax_red.set_yticklabels([f"{np.round(red_min + i*((red_max-red_min)/5), 2)}" for i in np.linspace(0, 5, 6)])
    ax_red.spines['top'].set_visible(False)
    ax_red.spines['right'].set_visible(False)
    ax_red.spines['left'].set_visible(False)
    ax_red.spines['bottom'].set_visible(False)
    ax_red.set_xticks([])
    ax_red.set_title('Low Frequency')
    ax_red.invert_yaxis()
    fig_red.savefig(Path(dest_folder) / 'hscan_red_legend.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig_red)
    
    # Create and save blue channel legend
    fig_blue, ax_blue = plt.subplots(figsize=(2, 10))
    ax_blue.imshow(gradient, aspect='auto', cmap=plt.cm.Blues)
    ax_blue.set_yticks(np.linspace(0, 255, 6))
    ax_blue.set_yticklabels([f"{np.round(blue_min + i*((blue_max-blue_min)/5), 2)}" for i in np.linspace(0, 5, 6)])
    ax_blue.spines['top'].set_visible(False)
    ax_blue.spines['right'].set_visible(False)
    ax_blue.spines['left'].set_visible(False)
    ax_blue.spines['bottom'].set_visible(False)
    ax_blue.set_xticks([])
    ax_blue.set_title('High Frequency')
    ax_blue.invert_yaxis()
    fig_blue.savefig(Path(dest_folder) / 'hscan_blue_legend.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig_blue)

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

@dependencies('bsc_stft')
def plot_bsc_stft(analysis_obj: ParamapAnalysisBase, dest_folder: str, **kwargs) -> None:
    """Plots the BSC_STFT energy data at the central frequency for each line individually.
    
    This function calculates the central frequency for each line in the ROI and extracts
    the corresponding BSC_STFT energy data. It creates both a 2D energy map where each line
    uses its own central frequency, and a line plot showing energy variation with depth.
    
    Args:
        analysis_obj (ParamapAnalysisBase): Analysis object containing BSC_STFT results
        dest_folder (str): Directory to save the plots
        **kwargs: Additional keyword arguments including:
            - line_idx (int, optional): Specific line index to plot in detail. If not provided,
              will plot the middle line
    """
    if kwargs.get('hide_all_visualizations', False):
        return
    
    assert Path(dest_folder).is_dir(), "plot_bsc_stft visualization: Output folder doesn't exist"
    
    # Check if BSC_STFT results are available
    if not hasattr(analysis_obj.single_window.results, 'bsc_stft_energy_dict'):
        print("[plot_bsc_stft] BSC_STFT energy dictionary not found in results. Skipping BSC_STFT plot.")
        return
    
    if not hasattr(analysis_obj.single_window.results, 'bsc_stft_frequencies'):
        print("[plot_bsc_stft] BSC_STFT frequencies not found in results. Skipping BSC_STFT plot.")
        return
    
    if not hasattr(analysis_obj.single_window.results, 'mean_central_freq_scan'):
        print("[plot_bsc_stft] Mean central frequency not found in results. Skipping BSC_STFT plot.")
        return
    
    # Get BSC_STFT data
    energy_dict = analysis_obj.single_window.results.bsc_stft_energy_dict
    frequencies = analysis_obj.single_window.results.bsc_stft_frequencies
    mean_central_freq_scan = analysis_obj.single_window.results.mean_central_freq_scan  # Hz
    mean_central_freq_mhz = mean_central_freq_scan / 1e6  # Convert to MHz
    
    # Get the first window to access the RF data and config
    first_window = analysis_obj.windows[0] if analysis_obj.windows else analysis_obj.single_window
    
    # Extract RF data from the window using the same method as in the analysis framework
    if analysis_obj.image_data.bmode.ndim == 2:
        scan_rf_window = analysis_obj.image_data.rf_data[
            first_window.ax_min: first_window.ax_max + 1, 
            first_window.lat_min: first_window.lat_max + 1
        ]
    elif analysis_obj.image_data.bmode.ndim == 3:
        scan_rf_window = analysis_obj.image_data.rf_data[
            first_window.cor_min: first_window.cor_max + 1, 
            first_window.lat_min: first_window.lat_max + 1,
            first_window.ax_min: first_window.ax_max + 1
        ]
    else:
        raise ValueError("Invalid RF data dimensions. Expected 2D or 3D data.")
    
    # Use the stored mean central frequency from BSC_STFT analysis
    num_lines = scan_rf_window.shape[0]
    line_bsc_energies = []
    
    print(f"[plot_bsc_stft] Using stored mean central frequency: {mean_central_freq_mhz:.2f} MHz")
    
    # Find the closest available frequency in BSC_STFT results to the mean central frequency
    freq_idx = np.argmin(np.abs(frequencies - mean_central_freq_mhz))
    actual_freq = frequencies[freq_idx]
    
    print(f"[plot_bsc_stft] Using frequency {actual_freq:.2f} MHz (closest to mean central frequency)")
    
    # Get BSC_STFT energy at the mean central frequency for each line
    for line_idx in range(num_lines):
        # Get BSC_STFT energy at this frequency for this line
        if actual_freq in energy_dict:
            # Get the energy data for this line at this frequency
            # energy_dict[actual_freq] has shape (lines, time_points)
            # We want the energy for this specific line
            if line_idx < energy_dict[actual_freq].shape[0]:
                line_energy = energy_dict[actual_freq][line_idx, :]  # Get all time points for this line
                line_bsc_energies.append(line_energy)
            else:
                # If line index is out of bounds, use zeros
                line_bsc_energies.append(np.zeros(energy_dict[actual_freq].shape[1]))
        else:
            # If frequency not found, use zeros
            line_bsc_energies.append(np.zeros(energy_dict[list(energy_dict.keys())[0]].shape[1]))
    
    # Convert to numpy array
    line_bsc_energies = np.array(line_bsc_energies)  # Shape: (lines, time_points)
    
    # Create full-size BSC_STFT energy image with ROI data
    bsc_stft_full = np.zeros_like(analysis_obj.image_data.bmode, dtype=np.float32)
    
    # Get single window boundaries
    single_window = analysis_obj.single_window
    
    # Get window dimensions
    window_height = single_window.ax_max - single_window.ax_min + 1
    window_width = single_window.lat_max - single_window.lat_min + 1
    
    # Resample the energy data to match window dimensions
    if line_bsc_energies.shape != (window_height, window_width):
        # Use interpolation to resample the energy data
        
        # Create coordinate grids for original and target dimensions
        y_orig, x_orig = np.mgrid[0:line_bsc_energies.shape[0], 0:line_bsc_energies.shape[1]]
        y_target, x_target = np.mgrid[0:window_height, 0:window_width]
        
        # Normalize coordinates to [0, 1] range
        y_orig_norm = y_orig / (line_bsc_energies.shape[0] - 1) if line_bsc_energies.shape[0] > 1 else 0
        x_orig_norm = x_orig / (line_bsc_energies.shape[1] - 1) if line_bsc_energies.shape[1] > 1 else 0
        y_target_norm = y_target / (window_height - 1) if window_height > 1 else 0
        x_target_norm = x_target / (window_width - 1) if window_width > 1 else 0
        
        # Flatten arrays for griddata
        points = np.column_stack((y_orig_norm.flatten(), x_orig_norm.flatten()))
        values = line_bsc_energies.flatten()
        xi = np.column_stack((y_target_norm.flatten(), x_target_norm.flatten()))
        
        # Interpolate
        resampled_energy = griddata(points, values, xi, method='linear', fill_value=0)
        resampled_energy = resampled_energy.reshape(window_height, window_width)
    else:
        resampled_energy = line_bsc_energies
    
    # Map to full image
    ax_slice = slice(single_window.ax_min, single_window.ax_max + 1)
    lat_slice = slice(single_window.lat_min, single_window.lat_max + 1)
    bsc_stft_full[ax_slice, lat_slice] = resampled_energy
    
    # Get the appropriate segmentation mask
    mask = analysis_obj.seg_data.seg_mask
    if mask.ndim == 3:  # Handle 3D case if needed
        mask = mask[:, :, analysis_obj.seg_data.frame]
    
    # Calculate normalization range using percentiles to handle outliers
    # Only consider values within the ROI for normalization
    valid_bsc_stft = bsc_stft_full[mask > 0]
    bsc_stft_min, bsc_stft_max = np.percentile(valid_bsc_stft[valid_bsc_stft != 0], [5, 95])
    
    # Get the B-mode image (keep as grayscale)
    if analysis_obj.image_data.sc_bmode is not None:
        bmode = np.array(analysis_obj.image_data.sc_bmode, dtype=np.uint8)
    else:
        bmode = np.array(analysis_obj.image_data.bmode, dtype=np.uint8)
    
    # Scan convert the BSC_STFT data if needed
    if analysis_obj.image_data.sc_bmode is not None:
        image_data = analysis_obj.image_data
        if image_data.sc_bmode.ndim == 2:
            # Scan convert the BSC_STFT data
            final_bsc_stft = scanConvert(bsc_stft_full, image_data.width, image_data.tilt,
                                    image_data.start_depth, image_data.end_depth,
                                    desiredHeight=image_data.sc_bmode.shape[0])[0].scArr
            
            # Scan convert the mask
            final_mask = scanConvert(mask.astype(np.float32), image_data.width, image_data.tilt,
                                   image_data.start_depth, image_data.end_depth,
                                   desiredHeight=image_data.sc_bmode.shape[0])[0].scArr
            
            # Threshold the scan-converted mask to ensure binary values
            final_mask = (final_mask > 0.5).astype(np.float32)
        else:
            raise NotImplementedError("3D scan conversion not yet implemented for BSC_STFT visualization")
    else:
        final_bsc_stft = bsc_stft_full
        final_mask = mask.astype(np.float32)
    
    # Normalize BSC_STFT data
    final_bsc_stft_norm = np.clip((final_bsc_stft - bsc_stft_min) / (bsc_stft_max - bsc_stft_min), 0, 1)
    
    # Calculate proper aspect ratio based on resolution
    if analysis_obj.image_data.sc_bmode is not None:
        width = bmode.shape[1] * analysis_obj.image_data.sc_lateral_res
        height = bmode.shape[0] * analysis_obj.image_data.sc_axial_res
    else:
        width = bmode.shape[1] * analysis_obj.image_data.lateral_res
        height = bmode.shape[0] * analysis_obj.image_data.axial_res
    aspect = width/height
    
    # Create figure for main overlay
    fig, ax = plt.subplots()
    
    # Plot B-mode image as background (full image)
    ax.imshow(bmode, cmap='gray')
    
    # Create masked array for BSC_STFT data (only show where mask is active)
    # This ensures BSC_STFT data only appears within the ROI
    masked_bsc_stft = np.ma.masked_where(final_mask == 0, final_bsc_stft_norm)
    
    # Plot BSC_STFT data as colorful contour heatmap overlay (only within ROI)
    im = ax.imshow(masked_bsc_stft, cmap='viridis', alpha=0.8, vmin=0, vmax=1)
    
    # Add contour lines for better visualization (only within ROI)
    contour = ax.contour(masked_bsc_stft, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    
    extent = im.get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
    ax.axis('off')
    
    # Save the main overlay figure without any padding
    fig.savefig(Path(dest_folder) / 'bsc_stft_overlay_line_by_line.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Create and save BSC_STFT legend
    fig_legend, ax_legend = plt.subplots(figsize=(2, 10))
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax_legend.imshow(gradient, aspect='auto', cmap='viridis')
    ax_legend.set_yticks(np.linspace(0, 255, 6))
    ax_legend.set_yticklabels([f"{np.round(bsc_stft_min + i*((bsc_stft_max-bsc_stft_min)/5), 4)}" for i in np.linspace(0, 5, 6)])
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.set_xticks([])
    ax_legend.set_title('BSC_STFT Energy\n(Line-by-line CF)')
    ax_legend.invert_yaxis()
    fig_legend.savefig(Path(dest_folder) / 'bsc_stft_legend_line_by_line.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig_legend)
