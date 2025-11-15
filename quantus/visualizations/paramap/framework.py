import pickle
from pathlib import Path
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ...image_loading.utc_loaders.transforms import scanConvert
from ...data_objs.visualizations import ParamapDrawingBase
from ...data_objs.analysis import ParamapAnalysisBase
from ...data_objs.image import UltrasoundRfImage
from ...data_objs.seg import BmodeSeg
from .functions import *

class_name = "ParamapAnalysis"

class ParamapVisualizations(ParamapDrawingBase):
    """
    Class to complete visualizations of parametric map-based UTC analysis.
    """

    def __init__(self, analysis_obj: ParamapAnalysisBase, visualization_funcs: list, **kwargs):
        # Type checking
        assert isinstance(analysis_obj, ParamapAnalysisBase), "analysis_obj must be a ParamapAnalysisBase child class"
        super().__init__(analysis_obj)

        # Default to project-level folder 'Visualization_Results' unless overridden.
        # If a relative path is provided, resolve it against the repository root
        # so outputs land consistently in the project folder, regardless of CWD.
        from pathlib import Path as _P
        repo_root = _P(__file__).resolve().parents[2]  # .../quantus
        repo_root = repo_root.parent                    # repository root
        default_dir = repo_root / "Visualization_Results"
        provided = kwargs.get('paramap_folder_path', str(default_dir))
        provided_path = _P(provided)
        if not provided_path.is_absolute():
            provided_path = (repo_root / provided_path).resolve()
        self.paramap_folder_path = str(provided_path)
        
        self.analysis_obj = analysis_obj
        self.visualization_funcs = visualization_funcs
        self.kwargs = kwargs
        self.numerical_paramaps = []
        self.windowed_paramap_mask = []
        self.paramap_names = []
        self.legend_paramaps = []
        self.plots = []
        
    def save_2d_paramap(self, bmode: np.ndarray, paramap: np.ndarray, legend: plt.Figure, dest_path: Path) -> None:
        """Saves the parametric map and legend to the specified path.
        
        Args:
            bmode (np.ndarray): The B-mode image to save.
            paramap (np.ndarray): The parametric map to save.
            legend (plt.Figure): The legend figure to save.
            dest_path (str): The destination path for saving the parametric map.
        """
        assert str(dest_path).endswith('.png'), "Parametric map output path must end with .png"
        
        # Overlay the paramap on the B-mode image
        fig, ax = plt.subplots()
        if self.analysis_obj.image_data.sc_bmode is not None:
            width = paramap.shape[1]*self.analysis_obj.image_data.sc_lateral_res
            height = paramap.shape[0]*self.analysis_obj.image_data.sc_axial_res
        else:
            width = paramap.shape[1]*self.analysis_obj.image_data.lateral_res
            height = paramap.shape[0]*self.analysis_obj.image_data.axial_res
        aspect = width/height
        im = ax.imshow(bmode)
        im = ax.imshow(paramap)
        extent = im.get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
        ax.axis('off')
        
        fig.savefig(dest_path.parent / (dest_path.stem + '_paramap.png'), bbox_inches='tight', pad_inches=0)
        legend.savefig(dest_path.parent / (dest_path.stem + '_legend.png'), bbox_inches='tight', pad_inches=0)
        # Explicitly close figures to prevent accumulation of open figures
        plt.close(fig)
        plt.close(legend)
        
    def save_general_paramap(self, paramap: np.ndarray, legend: plt.Figure, dest_path: Path) -> None:
        """Saves the parametric map to the specified path.
        
        Args:
            paramap (np.ndarray): The parametric map to save.
            dest_path (str): The destination path for saving the parametric map.
        """
        assert str(dest_path).endswith('.pkl'), "Parametric map output path must end with .pkl"
        
        legend.savefig(dest_path.parent / (dest_path.stem + '_legend.png'), bbox_inches='tight', pad_inches=0)
        # Close legend figure to avoid open figure warnings
        plt.close(legend)
        
        np.save(dest_path.parent / (dest_path.stem + '_paramap.npy'), paramap)
        
    def export_visualizations(self):
        """Used to specify which visualizations to export and where.
        """
        if len(self.visualization_funcs):
            paramap_folder_path = Path(self.paramap_folder_path)
            paramap_folder_path.mkdir(parents=True, exist_ok=True)

        if "paramaps" in self.visualization_funcs:
            if len(self.analysis_obj.image_data.bmode.shape) == 2:
                if self.analysis_obj.image_data.sc_bmode is not None:
                    bmode = cv2.cvtColor(np.array(self.analysis_obj.image_data.sc_bmode, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
                else:
                    bmode = cv2.cvtColor(np.array(self.analysis_obj.image_data.bmode, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                bmode = None
                # Always save bmode and segmentation data
                if self.analysis_obj.image_data.sc_bmode is not None:
                    np.save(paramap_folder_path / 'bmode.npy', self.analysis_obj.image_data.sc_bmode)
                    np.save(paramap_folder_path / 'segmentation.npy', self.analysis_obj.seg_data.sc_seg_mask)
                    pixdims = np.array([self.analysis_obj.image_data.sc_coronal_res, self.analysis_obj.image_data.sc_lateral_res, 
                                        self.analysis_obj.image_data.sc_axial_res])
                else:
                    np.save(paramap_folder_path / 'bmode.npy', self.analysis_obj.image_data.bmode)
                    np.save(paramap_folder_path / 'segmentation.npy', self.analysis_obj.seg_data.seg_mask)
                    pixdims = np.array([self.analysis_obj.image_data.coronal_res, self.analysis_obj.image_data.lateral_res, 
                                        self.analysis_obj.image_data.axial_res])
                np.save(paramap_folder_path / 'pixdims.npy', pixdims)

            # Save parametric maps
            params = self.analysis_obj.windows[0].results.__dict__.keys()
            cmap_ix = 0
            windowed_paramap_mask_local = None
            for param in params:
                if isinstance(getattr(self.analysis_obj.windows[0].results, param), (str, list, np.ndarray)):
                    continue
                colored_paramap, legend, numerical_paramap, windowed_paramap_mask_local = self.draw_paramap(param, self.cmaps[cmap_ix % len(self.cmaps)])
                self.numerical_paramaps.append(numerical_paramap)
                self.paramap_names.append(param)
                cmap_ix += 1
                
                # Always save parametric maps
                if bmode is not None:
                    self.save_2d_paramap(bmode, colored_paramap, legend, paramap_folder_path / f'{param}.png')
                else:
                    # Save numerical data and also a PNG of the colored paramap for preview
                    self.save_general_paramap(numerical_paramap, legend, paramap_folder_path / f'{param}.pkl')
                    try:
                        fig, ax = plt.subplots()
                        im = ax.imshow(colored_paramap)
                        ax.axis('off')
                        fig.savefig(paramap_folder_path / f'{param}.png', bbox_inches='tight', pad_inches=0)
                        plt.close(fig)
                    except Exception as e:
                        pass # 3d preview failed, but numerical data is saved
            # Set windowed mask if produced by draw_paramap; else default to empty list
            self.windowed_paramap_mask = windowed_paramap_mask_local if windowed_paramap_mask_local is not None else []

        # H-scan paramap storage 
        if "plot_hscan_result" in self.visualization_funcs:
            # Only store if H-scan results are present
            if hasattr(self.analysis_obj.single_window.results, 'hscan_blue_channel') and \
               hasattr(self.analysis_obj.single_window.results, 'hscan_red_channel'):
                blue_channel = self.analysis_obj.single_window.results.hscan_blue_channel
                red_channel = self.analysis_obj.single_window.results.hscan_red_channel
                blue_full = np.zeros_like(self.analysis_obj.image_data.bmode, dtype=np.float32)
                red_full = np.zeros_like(self.analysis_obj.image_data.bmode, dtype=np.float32)
                single_window = self.analysis_obj.single_window
                ax_slice = slice(single_window.ax_min, single_window.ax_max + 1)
                lat_slice = slice(single_window.lat_min, single_window.lat_max + 1)
                blue_full[ax_slice, lat_slice] = blue_channel
                red_full[ax_slice, lat_slice] = red_channel
                # Scan convert if needed
                if self.analysis_obj.image_data.sc_bmode is not None:
                    image_data = self.analysis_obj.image_data
                    if image_data.sc_bmode.ndim == 2:
                        blue_sc = scanConvert(blue_full, image_data.width, image_data.tilt,
                                             image_data.start_depth, image_data.end_depth,
                                             desiredHeight=image_data.sc_bmode.shape[0])[0].scArr
                        red_sc = scanConvert(red_full, image_data.width, image_data.tilt,
                                            image_data.start_depth, image_data.end_depth,
                                            desiredHeight=image_data.sc_bmode.shape[0])[0].scArr
                        self.hscan_paramaps = [blue_sc, red_sc]
                    else:
                        raise NotImplementedError("3D scan conversion not yet implemented for H-scan visualization")
                else:
                    self.hscan_paramaps = [blue_full, red_full]
                self.hscan_paramap_names = ["hscan_blue_channel", "hscan_red_channel"]

        # Complete all custom visualizations
        for func_name in self.visualization_funcs:
            if func_name == "paramaps":
                continue
            try:
                function = globals()[func_name]
                function(self.analysis_obj, self.paramap_folder_path, **self.kwargs)
            except Exception as e:
                print(f"ERROR: Visualization function '{func_name}' failed: {e}")
                import traceback
                traceback.print_exc()
