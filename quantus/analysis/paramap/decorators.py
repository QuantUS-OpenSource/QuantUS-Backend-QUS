from ...plugin_utils import attr_decorator

dependencies = attr_decorator('deps')
supported_spatial_dims = attr_decorator('supported_spatial_dims')
required_kwargs = attr_decorator('required_kwargs')
output_vars = attr_decorator('outputs')
location = attr_decorator('location')
default_kwarg_vals = attr_decorator('default_kwarg_vals')

# Which slice of the phantom RF data a window/full_segmentation function receives as
# phantom_rf_window. The framework does the slicing (see ParamapAnalysis.get_phantom_window);
# plugins only declare the policy they need:
#   "matched"       (default) -- same axial and lateral bounds as the scan window, so
#                    phantom_rf_window.shape == scan_rf_window.shape. Required by any method
#                    that combines scan and phantom data element-wise (e.g. hscan).
#   "central_lines" -- 2D only: the window's axial bounds, but the central 50% of the
#                    phantom's lines regardless of the window's lateral position. For methods
#                    that average the reference spectrum across lines (e.g. MATLAB-matched AC/BSC).
PHANTOM_WINDOW_POLICIES = ("matched", "central_lines")
phantom_window = attr_decorator('phantom_window')
