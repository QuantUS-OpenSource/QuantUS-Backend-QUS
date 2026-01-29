from typing import Dict

import numpy as np

from ..decorators import required_kwargs
from ....data_objs.visualizations import ParamapDrawingBase

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