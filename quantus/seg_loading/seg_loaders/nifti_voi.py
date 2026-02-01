from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_closing

from ..transforms import map_1d_to_3d
from ..decorators import extensions
from ...data_objs.seg import BmodeSeg
from ...data_objs.image import UltrasoundRfImage

@extensions(".nii", ".nii.gz")
def nifti_voi(image_data: UltrasoundRfImage, seg_path: str, **kwargs) -> BmodeSeg:
    """
    Function for loading ROI data from a pickle file saved from the QuantUS UI.
     
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the ROI file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the ROI file name.
    """
    out = BmodeSeg()
    
    seg = nib.load(seg_path)
    out.sc_seg_mask = np.asarray(seg.dataobj, dtype=np.uint8)
    out.frame = 0
    out.seg_name = Path(seg_path).stem

    coord_map = image_data.coord_map_3d
    masked_coords_3d = np.transpose(np.where(out.sc_seg_mask))
    original_coords_3d = [map_1d_to_3d(coord_map[tuple(coord)], image_data.rf_data.shape[0], image_data.rf_data.shape[1], image_data.rf_data.shape[2])
                           for coord in masked_coords_3d]
    
    out.seg_mask = np.zeros_like(image_data.bmode)
    out.seg_mask[tuple(np.transpose(original_coords_3d))] = 1
    out.seg_mask = binary_closing(out.seg_mask, structure=np.ones((3, 3, 3))).astype(np.uint8)

    return out