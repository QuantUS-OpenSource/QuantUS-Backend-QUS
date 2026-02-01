from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from ....image_loading.utc_loaders.transforms import scanConvert

from ..decorators import *
from ....data_objs.analysis import ParamapAnalysisBase

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
