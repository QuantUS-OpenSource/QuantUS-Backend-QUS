import numpy as np
from typing import List

from ...data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig, ParamapAnalysisBase
from ...data_objs.analysis import Window
from ..options import get_analysis_types

class_name = "ParamapAnalysis"

class ParamapAnalysis(ParamapAnalysisBase):
    """
    Class to complete RF analysis via the sliding window technique
    and generate a corresponding parametric map.
    """
    def __init__(self, image_data: UltrasoundRfImage, config: RfAnalysisConfig, seg: BmodeSeg, 
                 function_names: List[str], **kwargs):
        # Type checking
        assert isinstance(image_data, UltrasoundRfImage), 'image_data must be an UltrasoundRfImage child class'
        assert isinstance(config, RfAnalysisConfig), 'config must be an RfAnalysisConfig'
        assert isinstance(seg, BmodeSeg), 'seg_data must be a BmodeSeg'
        # assert image_data.rf_data.shape == image_data.phantom_rf_data.shape or image_data.rf_data.shape[1:] == image_data.phantom_rf_data.shape, \
        #     'RF data and phantom RF data must have the same shape'
        super().__init__()
        
        self.analysis_kwargs = kwargs
        self.function_names = function_names
        self.ax_win_size_pixels = None
        self.lat_win_size_pixels = None
        _, self.functions = get_analysis_types()
        self.seg_data = seg
        self.image_data = image_data
        self.config = config
        if hasattr(self.seg_data, "splines") and len(self.seg_data.splines) == 2:
            self.spline_x = seg.splines[0]
            self.spline_y = seg.splines[1]
        self.determine_func_order()
            
    def determine_func_order(self):
        """Determine the order of functions to be applied to the data.
        
        This function is called in the constructor and sets the order of functions
        to be applied to the data based on the provided function names.
        """
        self.ordered_funcs = []; self.ordered_func_names = []; self.results_names = []
        self.unordered_window_func_names = set(); self.unordered_full_seg_func_names = set()
        
        def assign_locs(func_name, deps, locs):
            """Assign locations for the function based on its dependencies and locations."""
            if 'window' in locs:
                self.unordered_window_func_names.add(func_name)
                [self.unordered_full_seg_func_names.add(dep) for dep in deps]
            if 'full_segmentation' in locs:
                self.unordered_full_seg_func_names.add(func_name)
                [self.unordered_window_func_names.add(dep) for dep in deps]
        
        def process_deps(func_name):
            if func_name in self.ordered_func_names:
                return
            if func_name in self.functions["paramap"]:
                # Handle function dependencies and outputs
                function = self.functions["paramap"][func_name]
                deps = function.deps if hasattr(function, 'deps') else []
                results_names = function.outputs if hasattr(function, 'outputs') else []
                for dep in deps:
                    process_deps(dep)
                
                # Handle function locations
                locs = function.location if hasattr(function, 'location') else ['window', 'full_segmentation']
                assign_locs(func_name, deps, locs)
            else:
                raise ValueError(f"Function '{func_name}' not found!")
            
            self.ordered_funcs.append(function)
            self.ordered_func_names.append(func_name)
            self.results_names.extend(results_names)

        for function_name in self.function_names:
            process_deps(function_name)
            
    def generate_seg_windows_3d(self):
        """Generate 3D voxel windows for UTC analysis based on user-defined spline."""
        # Some axial/lateral/coronal dims
        axial_pix_size = round(self.config.ax_win_size / self.image_data.axial_res)  # mm/(mm/pix)
        lateral_pix_size = round(self.config.lat_win_size / self.image_data.lateral_res)  # mm(mm/pix)
        coronal_pix_size = round(self.config.cor_win_size / self.image_data.coronal_res)  # mm/(mm/pix)
        
        # Overlap fraction determines the incremental distance between windows
        axial_increment = axial_pix_size * (1 - self.config.axial_overlap)
        lateral_increment = lateral_pix_size * (1 - self.config.lateral_overlap)
        coronal_increment = coronal_pix_size * (1 - self.config.coronal_overlap)
        
        # Determine windows - Find Volume to Iterate Over
        axial_start = np.min(np.where(np.any(self.seg_data.seg_mask, axis=(0, 1)))[0])
        axial_end = np.max(np.where(np.any(self.seg_data.seg_mask, axis=(0, 1)))[0])
        lateral_start = np.min(np.where(np.any(self.seg_data.seg_mask, axis=(0, 2)))[0])
        lateral_end = np.max(np.where(np.any(self.seg_data.seg_mask, axis=(0, 2)))[0])
        coronal_start = np.min(np.where(np.any(self.seg_data.seg_mask, axis=(1, 2)))[0])
        coronal_end = np.max(np.where(np.any(self.seg_data.seg_mask, axis=(1, 2)))[0])
        
        self.windows = []
        
        for axial_pos in np.arange(axial_start, axial_end, axial_increment):
            for lateral_pos in np.arange(lateral_start, lateral_end, lateral_increment):
                for coronal_pos in np.arange(coronal_start, coronal_end, coronal_increment):
                    # Convert axial, lateral, and coronal positions to indices
                    axial_ind = np.round(axial_pos).astype(int)
                    lateral_ind = np.round(lateral_pos).astype(int)
                    coronal_ind = np.round(coronal_pos).astype(int)
                    
                    # Determine if window is inside analysis volume
                    mask_vals = self.seg_data.seg_mask[
                        coronal_ind : (coronal_ind + coronal_pix_size),
                        lateral_ind : (lateral_ind + lateral_pix_size),
                        axial_ind : (axial_ind + axial_pix_size),
                    ]
                    
                    # Define Percentage Threshold
                    total_number_of_elements_in_region = mask_vals.size
                    number_of_ones_in_region = len(np.where(mask_vals == True)[0])
                    percentage_ones = number_of_ones_in_region / total_number_of_elements_in_region
                    
                    if percentage_ones > self.config.window_thresh:
                        # Add ROI to output structure, quantize back to valid distances
                        new_window = Window()
                        new_window.ax_min = int(axial_pos)
                        new_window.ax_max = int(axial_pos + axial_pix_size)
                        new_window.lat_min = int(lateral_pos)
                        new_window.lat_max = int(lateral_pos + lateral_pix_size)
                        new_window.cor_min = int(coronal_pos)
                        new_window.cor_max = int(coronal_pos + coronal_pix_size)
                        self.windows.append(new_window)
                        
    def get_seg_window_lens(self):
        wavelength = self.config.speed_of_sound / self.config.center_frequency * 1000 # mm
        if self.config.ax_win_size_units == 1: # mm
            self.ax_win_len_pixels = round(self.config.ax_win_size / self.image_data.axial_res)
        elif self.config.ax_win_size_units == 2: # wavelength
            self.ax_win_len_pixels = round((self.config.ax_win_size_units * wavelength) / self.image_data.axial_res)
        else:
            raise ValueError("Axial window size units not recognized")
        
        if self.config.lat_win_size_units == 1: # mm
            self.lat_win_len_pixels = round(self.config.lat_win_size / self.image_data.lateral_res)
        elif self.config.lat_win_size_units == 2: # wavelength
            self.lat_win_len_pixels = round((self.config.lat_win_size_units * wavelength) / self.image_data.lateral_res)
        elif self.config.lat_win_size_units == 3: # pix
            self.lat_win_len_pixels = round(self.config.lat_win_size)
        else:
            raise ValueError("Lateral window size units not recognized")
        
    def generate_seg_windows(self):
        """Generate windows for parametric map analysis based on user-defined segmentation.
        """
        if len(self.seg_data.seg_mask.shape) == 3: # 3D analysis
            return self.generate_seg_windows_3d()
        
        self.get_seg_window_lens()
        
        # Some axial/lateral dims
        ax_pix_size = self.ax_win_len_pixels
        lat_pix_size = self.lat_win_len_pixels
        
        axial = list(range(self.image_data.rf_data.shape[0]))
        lateral = list(range(self.image_data.rf_data.shape[1]))

        # Overlap fraction determines the incremental distance between windows
        axial_increment = round(ax_pix_size * (1 - self.config.axial_overlap))
        lateral_increment = round(lat_pix_size * (1 - self.config.lateral_overlap))
        
        if lateral_increment < 1:
            print('Warning: there are too few number of A-lines in the sub-ROI, possibly because the lateral step size is too large. Increasing the lateral sub-ROI size is suggested.')
            lateral_increment = 1

        # Determine windows - Find Region to Iterate Over
        mask = self.seg_data.seg_mask
        mask_indices = np.where(mask)
        axial_start = mask_indices[0][0]
        axial_end = min(mask_indices[0][-1], mask.shape[0]-ax_pix_size-1)
        lateral_start = np.min(mask_indices[1])
        lateral_end = min(np.max(mask_indices[1]), mask.shape[1]-lat_pix_size-1)

        self.windows = []
        mask = self.seg_data.seg_mask

        for axial_ix in range(axial_start, axial_end, axial_increment):
            for lateral_ix in range(lateral_start, lateral_end, lateral_increment):
                # Determine if ROI is Inside Analysis Region
                mask_vals = mask[
                    axial_ix : (axial_ix + ax_pix_size),
                    lateral_ix : (lateral_ix + lat_pix_size),
                ]

                # Define Percentage Threshold
                total_elements_in_region = mask_vals.size
                ones_in_region = len(np.where(mask_vals == 1)[0])
                percentage_ones = ones_in_region / total_elements_in_region

                if percentage_ones > self.config.window_thresh:
                    # Add window to output structure, quantize back to valid distances
                    new_window = Window()
                    new_window.lat_min = int(lateral[lateral_ix])
                    new_window.lat_max = int(lateral[lateral_ix + lat_pix_size - 1])
                    new_window.ax_min = int(axial[axial_ix])
                    new_window.ax_max = int(axial[axial_ix + ax_pix_size - 1])
                    self.windows.append(new_window)

    def compute_window_vals(self, window, full_segmentation=False):
        """Compute parametric map values for a single window.
        
        Args:
            window (Window): Window object to store results.
        """
        if self.image_data.bmode.ndim == 2:
            img_window = self.image_data.rf_data[
                window.ax_min: window.ax_max + 1, window.lat_min: window.lat_max + 1
            ]
            n_ref_lines = self.image_data.phantom_rf_data.shape[1]
            lat_start_ix = round(0.25*n_ref_lines)
            lat_end_ix = round(0.75*n_ref_lines)
            phantom_window = self.image_data.phantom_rf_data[
                window.ax_min: window.ax_max + 1, lat_start_ix: lat_end_ix+1
            ]
        elif self.image_data.bmode.ndim == 3:
            img_window = self.image_data.rf_data[
                window.cor_min: window.cor_max + 1, window.lat_min: window.lat_max + 1,
                window.ax_min: window.ax_max + 1
            ]
            phantom_window = self.image_data.phantom_rf_data[
                window.cor_min: window.cor_max + 1, window.lat_min: window.lat_max + 1,
                window.ax_min: window.ax_max + 1
            ]
        else:
            raise ValueError("Invalid RF data dimensions. Expected 2D or 3D data.")

        if not full_segmentation:
            for i, function in enumerate(self.ordered_funcs):
                if self.ordered_func_names[i] in self.unordered_window_func_names:
                    function(img_window, phantom_window, window, self.config, self.image_data, **self.analysis_kwargs)
        else:
            for i, function in enumerate(self.ordered_funcs):
                if self.ordered_func_names[i] in self.unordered_full_seg_func_names:
                    function(img_window, phantom_window, window, self.config, self.image_data, **self.analysis_kwargs)

    def compute_single_window(self):
        """Define a single window that contains all parametric map windows for analysis, capturing the entire segmentation.
        Perform analysis on this full segmentation window.
        """
        min_ax = min([window.ax_min for window in self.windows])
        max_ax = max([window.ax_max for window in self.windows])
        min_lat = min([window.lat_min for window in self.windows])
        max_lat = max([window.lat_max for window in self.windows])
        if self.image_data.bmode.ndim == 3:
            min_cor = min([window.cor_min for window in self.windows])
            max_cor = max([window.cor_max for window in self.windows])
        
        self.single_window = Window()
        self.single_window.lat_min = min_lat
        self.single_window.lat_max = max_lat
        self.single_window.ax_min = min_ax
        self.single_window.ax_max = max_ax
        if self.image_data.bmode.ndim == 3:
            self.single_window.cor_min = min_cor
            self.single_window.cor_max = max_cor
        
        self.compute_window_vals(self.single_window, full_segmentation=True)
