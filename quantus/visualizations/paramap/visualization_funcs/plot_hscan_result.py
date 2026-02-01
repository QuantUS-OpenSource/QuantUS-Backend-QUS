from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ....image_loading.utc_loaders.transforms import scanConvert
import cv2

from ..decorators import *
from ....data_objs.analysis import ParamapAnalysisBase

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