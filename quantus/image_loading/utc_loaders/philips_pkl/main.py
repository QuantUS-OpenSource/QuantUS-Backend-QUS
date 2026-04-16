import pickle

from ....data_objs.image import UltrasoundRfImage

class EntryClass(UltrasoundRfImage):
    """
    Class for Terason RF image data.
    This class is used to parse RF data from Terason ultrasound machines.
    """
    extensions = [".pkl", ".pickle"]
    spatial_dims = 3
    gui_kwargs = []; cli_kwargs = []
    default_gui_kwarg_vals = []; default_cli_kwarg_vals = []

    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        super().__init__(scan_path, phantom_path)
        
        with open(scan_path, "rb") as f:
            img_data, img_info = pickle.load(f)
        with open(phantom_path, "rb") as f:
            ref_data, ref_info = pickle.load(f)
        
        self.rf_data = img_data.rf
        self.phantom_rf_data = ref_data.rf
        self.bmode = img_data.bMode
        self.sc_bmode = img_data.scBmode
        self.axial_res = img_info.xResRF
        self.lateral_res = img_info.yResRF
        self.coronal_res = img_info.zResRF
        self.sc_axial_res = img_info.axialRes
        self.sc_lateral_res = img_info.lateralRes
        self.sc_coronal_res = img_info.coronalRes
        self.coord_map_3d = img_data.coordMap3d
        
        self.sc_params_3d = img_info.scParams
