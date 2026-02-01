from pathlib import Path
from typing import Dict

import numpy as np

from ..decorators import required_kwargs
from ....data_objs.visualizations import ParamapDrawingBase

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