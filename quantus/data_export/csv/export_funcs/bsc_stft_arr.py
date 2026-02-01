from pathlib import Path
from typing import Dict

import numpy as np
from scipy.interpolate import griddata

from ..decorators import required_kwargs
from ....data_objs.visualizations import ParamapDrawingBase

@required_kwargs('output_folder')
def bsc_stft_arr(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
    """
    Export the full 2D parametric maps for BSC_STFT energy data at all available frequencies as CSV files, 
    preserving the ROI shape and location in the full image.
    Each file will be named 'bsc_stft_energy_{frequency}MHz_paramap.csv'.
    
    Args:
        visualizations_obj (ParamapDrawingBase): The visualizations object containing the data.
        data_dict (Dict[str, str]): (Unused) Dictionary to store the exported data.
        output_folder (str): Path to the folder where CSVs will be saved.
    """
    output_folder = Path(kwargs['output_folder'])
    output_folder.mkdir(parents=True, exist_ok=True)

    # Check if BSC_STFT results are available
    if not hasattr(visualizations_obj.analysis_obj.single_window.results, 'bsc_stft_energy_dict'):
        raise AttributeError("BSC_STFT energy dictionary not found in results. Ensure 'bsc_stft' analysis was run.")
    
    # Get BSC_STFT data
    energy_dict = visualizations_obj.analysis_obj.single_window.results.bsc_stft_energy_dict
    
    # Get single window boundaries
    single_window = visualizations_obj.analysis_obj.single_window
    
    # Use scan-converted segmentation mask if available, else original
    seg_data = visualizations_obj.analysis_obj.seg_data
    if hasattr(seg_data, 'sc_seg_mask') and seg_data.sc_seg_mask is not None:
        roi_mask = seg_data.sc_seg_mask > 0
    else:
        roi_mask = seg_data.seg_mask > 0

    # Export energy data for each frequency
    for freq in energy_dict.keys():
        # Get energy array for this frequency (shape: lines x time_points)
        energy_array = energy_dict[freq]
        
        # Create full-size energy image with ROI data
        energy_full = np.zeros_like(visualizations_obj.analysis_obj.image_data.bmode, dtype=np.float32)
        
        # Get window dimensions
        window_height = single_window.ax_max - single_window.ax_min + 1
        window_width = single_window.lat_max - single_window.lat_min + 1
        
        # Resample the energy data to match window dimensions if needed
        if energy_array.shape != (window_height, window_width):
             # Use interpolation to resample the energy data
             
             # Create coordinate grids for original and target dimensions
            y_orig, x_orig = np.mgrid[0:energy_array.shape[0], 0:energy_array.shape[1]]
            y_target, x_target = np.mgrid[0:window_height, 0:window_width]
            
            # Normalize coordinates to [0, 1] range
            y_orig_norm = y_orig / (energy_array.shape[0] - 1) if energy_array.shape[0] > 1 else 0
            x_orig_norm = x_orig / (energy_array.shape[1] - 1) if energy_array.shape[1] > 1 else 0
            y_target_norm = y_target / (window_height - 1) if window_height > 1 else 0
            x_target_norm = x_target / (window_width - 1) if window_width > 1 else 0
            
            # Flatten arrays for griddata
            points = np.column_stack((y_orig_norm.flatten(), x_orig_norm.flatten()))
            values = energy_array.flatten()
            xi = np.column_stack((y_target_norm.flatten(), x_target_norm.flatten()))
            
            # Interpolate
            resampled_energy = griddata(points, values, xi, method='linear', fill_value=0)
            resampled_energy = resampled_energy.reshape(window_height, window_width)
        else:
            resampled_energy = energy_array
        
        # Map to full image
        ax_slice = slice(single_window.ax_min, single_window.ax_max + 1)
        lat_slice = slice(single_window.lat_min, single_window.lat_max + 1)
        energy_full[ax_slice, lat_slice] = resampled_energy
        
        # Create output array with zero outside ROI
        output_array = np.full_like(energy_full, 0.0, dtype=float)
        output_array[roi_mask] = energy_full[roi_mask]
        
        # Save to CSV file
        freq_str = f"{freq:.2f}"
        out_path = output_folder / f"bsc_stft_energy_{freq_str}MHz_paramap.csv"
        np.savetxt(out_path, output_array, delimiter=",", fmt="%g")
    
    data_dict["info"] = [f"Exported BSC_STFT energy data for {len(energy_dict)} frequencies. See separate CSVs in output_folder."]
