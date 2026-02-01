from pathlib import Path
from typing import Dict

import numpy as np

from ..decorators import required_kwargs
from ....data_objs.visualizations import ParamapDrawingBase

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