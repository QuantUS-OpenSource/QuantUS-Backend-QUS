from pathlib import Path
from typing import List
from ..paramap.framework import ParamapVisualizations
from .functions import *

class_name = "BmodeVisualizations"

class BmodeVisualizations(ParamapVisualizations):
    """
    Class to complete B-mode visualization (heatmaps) for bmode/radiomics results.
    Inherits from ParamapVisualizations to reuse the standard heatmap generation logic.
    """
    def __init__(self, analysis_obj, visualization_funcs, **kwargs):
        super().__init__(analysis_obj, visualization_funcs, **kwargs)
        # Use colormap objects/names instead of strings to avoid 'IndexError' in draw_paramap
        self.cmaps = [self.cmaps[0], self.cmaps[1], self.cmaps[2], self.cmaps[3]]

