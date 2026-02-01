from pathlib import Path
from typing import Dict

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

from ..decorators import required_kwargs
from ....data_objs.visualizations import ParamapDrawingBase

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