import numpy as np
from typing import List

from ...data_objs import UltrasoundRfImage, BmodeSeg, RfAnalysisConfig
from ..paramap.framework import ParamapAnalysis
from .functions import *

class_name = "BmodeAnalysis"

class BmodeAnalysis(ParamapAnalysis):
    """
    Class to complete B-mode analysis via the sliding window technique.
    """
    def determine_func_order(self):
        """Determine the order of functions to be applied to the data.
        Overridden to use globals from the bmode.functions module.
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
            if func_name in globals():
                # Handle function dependencies and outputs
                function = globals()[func_name]
                deps = function.deps if hasattr(function, 'deps') else []
                results_names = function.outputs if hasattr(function, 'outputs') else []
                for dep in deps:
                    process_deps(dep)
                
                # Handle function locations
                locs = function.location if hasattr(function, 'location') else ['window', 'full_segmentation']
                assign_locs(func_name, deps, locs)
            else:
                raise ValueError(f"Function '{func_name}' not found in Bmode analysis!")
            
            self.ordered_funcs.append(function)
            self.ordered_func_names.append(func_name)
            self.results_names.extend(results_names)

        for function_name in self.function_names:
            process_deps(function_name)
