from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from scipy.interpolate import griddata

from .decorators import required_kwargs
from ...data_objs.visualizations import ParamapDrawingBase

@required_kwargs()
def descr_vals(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
    """Compute descriptive values for each parameter in the analysis object and save to a CSV file.
    This includes mean, median, standard deviation, minimum, and maximum values.

    Args:
        visualizations_obj (ParamapDrawingBase): The visualizations object containing the data.
        params (Dict[str, str]): Dictionary of parameters to compute mean values for.
    """
    params = visualizations_obj.analysis_obj.windows[0].results.__dict__.keys()
    for param in params:
        if isinstance(getattr(visualizations_obj.analysis_obj.windows[0].results, param), (str, list, np.ndarray)):
            continue
        param_arr = [getattr(window.results, param) for window in visualizations_obj.analysis_obj.windows]
        data_dict[f"mean_{param}"] = [np.mean(param_arr)]
        data_dict[f"std_{param}"] = [np.std(param_arr)]
        data_dict[f"min_{param}"] = [np.min(param_arr)]
        data_dict[f"max_{param}"] = [np.max(param_arr)]
        data_dict[f"median_{param}"] = [np.median(param_arr)]

@required_kwargs()
def hscan_stats(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
    """Export H-scan statistics to CSV format. This includes descriptive statistics for both
    blue (high frequency) and red (low frequency) channels within the ROI.

    Args:
        visualizations_obj (ParamapDrawingBase): The visualizations object containing the data.
        data_dict (Dict[str, str]): Dictionary to store the exported data.
    """
    # Check if H-scan data exists
    if not (hasattr(visualizations_obj.analysis_obj.single_window.results, 'hscan_blue_channel') and
            hasattr(visualizations_obj.analysis_obj.single_window.results, 'hscan_red_channel')):
        return

    # Get H-scan data
    blue_channel = visualizations_obj.analysis_obj.single_window.results.hscan_blue_channel
    red_channel = visualizations_obj.analysis_obj.single_window.results.hscan_red_channel

    # Create window index map
    window_idx_map = np.zeros_like(visualizations_obj.analysis_obj.image_data.bmode, dtype=int)
    
    # Fill window index map using all windows
    for i, window in enumerate(visualizations_obj.analysis_obj.windows):
        if window.cor_min == -1:
            window_idx_map[window.ax_min:window.ax_max+1, window.lat_min:window.lat_max+1] = i+1
        else:
            window_idx_map[window.cor_min:window.cor_max+1, window.lat_min:window.lat_max+1, window.ax_min:window.ax_max+1] = i+1

    # Create full-size channel images with ROI data
    blue_full = np.zeros_like(visualizations_obj.analysis_obj.image_data.bmode, dtype=np.float32)
    red_full = np.zeros_like(visualizations_obj.analysis_obj.image_data.bmode, dtype=np.float32)

    # Get single window boundaries
    single_window = visualizations_obj.analysis_obj.single_window

    # Apply the window mask to the channel data using window_idx_map
    window_points = np.transpose(np.where(window_idx_map > 0))
    for point in window_points:
        # Map from global coordinates to single window coordinates
        rel_ax = point[0] - single_window.ax_min
        rel_lat = point[1] - single_window.lat_min
        if 0 <= rel_ax < blue_channel.shape[0] and 0 <= rel_lat < blue_channel.shape[1]:
            blue_full[tuple(point)] = blue_channel[rel_ax, rel_lat]
            red_full[tuple(point)] = red_channel[rel_ax, rel_lat]

    # Calculate global statistics across all windows
    valid_mask = window_idx_map > 0
    if np.any(valid_mask):
        # Blue channel statistics
        blue_values = blue_full[valid_mask]
        data_dict['hscan_blue_mean'] = [np.mean(blue_values)]
        data_dict['hscan_blue_std'] = [np.std(blue_values)]
        data_dict['hscan_blue_min'] = [np.min(blue_values)]
        data_dict['hscan_blue_max'] = [np.max(blue_values)]
        data_dict['hscan_blue_median'] = [np.median(blue_values)]

        # Red channel statistics
        red_values = red_full[valid_mask]
        data_dict['hscan_red_mean'] = [np.mean(red_values)]
        data_dict['hscan_red_std'] = [np.std(red_values)]
        data_dict['hscan_red_min'] = [np.min(red_values)]
        data_dict['hscan_red_max'] = [np.max(red_values)]
        data_dict['hscan_red_median'] = [np.median(red_values)]

@required_kwargs('pyradiomics_yaml_paths')
def radiomics_stats(visualizations_objs: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
    """Export radiomics statistics to CSV format. This includes descriptive statistics for each
    radiomics feature within the ROI.

    Args:
        visualizations_obj (ParamapDrawingBase): The visualizations object containing the data.
        data_dict (Dict[str, str]): Dictionary to store the exported data.
        pyradiomics_yaml_paths (List[str]): Paths to the PyRadiomics configuration YAML files. One for each analysis run.
    """
    import logging
    logging.getLogger('radiomics').setLevel(logging.ERROR)

    assert(type(kwargs['pyradiomics_yaml_paths']) == list), "pyradiomics_yaml_paths must be a list of paths to PyRadiomics YAML configuration files."

    if hasattr(visualizations_objs.analysis_obj.image_data, 'sc_bmode'):
        pix_dims = [visualizations_objs.analysis_obj.image_data.sc_axial_res, visualizations_objs.analysis_obj.image_data.sc_lateral_res]
        if hasattr(visualizations_objs.analysis_obj.image_data, 'sc_coronal_res') and visualizations_objs.analysis_obj.image_data.sc_coronal_res is not None:
            pix_dims.append(visualizations_objs.analysis_obj.image_data.sc_coronal_res)
    else:
        pix_dims = [visualizations_objs.analysis_obj.image_data.axial_res, visualizations_objs.analysis_obj.image_data.lateral_res]
        if hasattr(visualizations_objs.analysis_obj.image_data, 'coronal_res') and visualizations_objs.analysis_obj.image_data.coronal_res is not None:
            pix_dims.append(visualizations_objs.analysis_obj.image_data.coronal_res)

    for radiomics_config_path in kwargs['pyradiomics_yaml_paths']:
        extractor = featureextractor.RadiomicsFeatureExtractor(radiomics_config_path)

        for paramap, paramap_name in zip(visualizations_objs.numerical_paramaps, visualizations_objs.paramap_names):
            mask_arr = ~np.isnan(paramap)
            mask = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
            mask.SetSpacing(tuple(reversed(pix_dims)))

            paramap_z = np.nan_to_num(paramap, nan=0.0)
            image = sitk.GetImageFromArray(paramap_z)
            image.SetSpacing(tuple(reversed(pix_dims)))

            features = extractor.execute(image, mask)

            for name in features.keys():
                val_type = type(features[name])
                if val_type in [list, dict, tuple, str]:
                    continue
                data_key = f'{paramap_name}_{Path(radiomics_config_path).stem}_{name}'
                data_dict[data_key] = [features[name]]


@required_kwargs('output_folder')
def paramap_arr(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
    """
    Export the full 2D parametric map for each parameter as a CSV file, preserving the ROI shape and location in the full image.
    Each file will be named '{param}_paramap.csv'.
    Args:
        visualizations_obj (ParamapDrawingBase): The visualizations object containing the data.
        data_dict (Dict[str, str]): (Unused) Dictionary to store the exported data.
        output_folder (str): Path to the folder where CSVs will be saved.
    """
    output_folder = Path(kwargs['output_folder'])
    output_folder.mkdir(parents=True, exist_ok=True)

    # Use scan-converted idx_map if available and sc_bmode is present
    idx_map = (visualizations_obj.sc_window_idx_map
               if hasattr(visualizations_obj, "sc_window_idx_map") and visualizations_obj.sc_window_idx_map is not None
               else visualizations_obj.window_idx_map)
    for i, (paramap, paramap_name) in enumerate(zip(visualizations_obj.numerical_paramaps, visualizations_obj.paramap_names)):
        roi_mask = idx_map > 0
        if paramap.shape == idx_map.shape or np.prod(paramap.shape) == np.prod(idx_map.shape):
            # Already full-size (reshape if needed)
            full_paramap = paramap.reshape(idx_map.shape)
        else:
            # Fill ROI locations in a full-size array
            full_paramap = np.full_like(idx_map, np.nan, dtype=float)
            if paramap.size != np.count_nonzero(roi_mask):
                raise ValueError(f"paramap.size ({paramap.size}) does not match ROI mask count ({np.count_nonzero(roi_mask)}) for {paramap_name}")
            full_paramap[roi_mask] = paramap.ravel()
        out_path = output_folder / f"{paramap_name}_paramap.csv"
        np.savetxt(out_path, full_paramap, delimiter=",", fmt="%g")

@required_kwargs('output_folder')
def hscan_arr(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
    """
    Export the full 2D parametric maps for H-scan blue and red channels as CSV files, preserving the ROI shape and location in the full image.
    Each file will be named 'hscan_blue_channel_paramap.csv' and 'hscan_red_channel_paramap.csv'.
    Args:
        visualizations_obj (ParamapDrawingBase): The visualizations object containing the data.
        data_dict (Dict[str, str]): (Unused) Dictionary to store the exported data.
        output_folder (str): Path to the folder where CSVs will be saved.
    """
    output_folder = Path(kwargs['output_folder'])
    output_folder.mkdir(parents=True, exist_ok=True)

    # Require precomputed H-scan paramaps
    if not (hasattr(visualizations_obj, 'hscan_paramaps') and hasattr(visualizations_obj, 'hscan_paramap_names')):
        raise AttributeError("visualizations_obj must have 'hscan_paramaps' and 'hscan_paramap_names' attributes. Ensure 'plot_hscan_result' visualization was run.")

    # Use scan-converted segmentation mask if available, else original
    seg_data = visualizations_obj.analysis_obj.seg_data
    if hasattr(seg_data, 'sc_seg_mask') and seg_data.sc_seg_mask is not None:
        roi_mask = seg_data.sc_seg_mask > 0
    else:
        roi_mask = seg_data.seg_mask > 0

    for paramap, paramap_name in zip(visualizations_obj.hscan_paramaps, visualizations_obj.hscan_paramap_names):
        if paramap.shape != roi_mask.shape:
            raise ValueError(f"paramap.shape {paramap.shape} does not match ROI mask shape {roi_mask.shape} for {paramap_name}")
        full_paramap = np.full_like(paramap, np.nan, dtype=float)
        full_paramap[roi_mask] = paramap[roi_mask]
        out_path = output_folder / f"{paramap_name}_paramap.csv"
        np.savetxt(out_path, full_paramap, delimiter=",", fmt="%g")
    data_dict["info"] = ["See separate CSVs in output_folder"]

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
