"""Interactive 3D viewer for parametric maps exported by the "paramaps" visualization
function (see export_visualizations() in quantus/visualizations/paramap/framework.py).

3D parametric maps can't be previewed as a flat image the way 2D ones can, so this opens
an interactive napari session instead. napari is only imported inside view_3d_paramap()
since it's a heavy, optional dependency not needed for the rest of the pipeline.
"""

from pathlib import Path

import numpy as np


def view_3d_paramap(paramap_dir: str, param_name: str):
    """Open a napari viewer with a 3D B-mode volume overlaid with a computed parametric map.

    Args:
        paramap_dir: Destination folder passed to the paramap visualization stage
            (contains bmode.npy, pixdims.npy, and <param_name>_paramap.npy).
        param_name: Name of the analysis parameter to overlay (e.g. "mbf", "snr") —
            matches an attribute name on Window.results.

    Returns:
        The napari.Viewer instance (kept alive by the caller for interactive use).
    """
    import napari

    paramap_dir = Path(paramap_dir)
    bmode = np.load(paramap_dir / "bmode.npy")
    paramap = np.load(paramap_dir / f"{param_name}_paramap.npy")
    pixdims = np.load(paramap_dir / "pixdims.npy")

    affine = np.eye(4)
    affine[0, 0] = pixdims[0]
    affine[1, 1] = pixdims[1]
    affine[2, 2] = pixdims[2]

    viewer = napari.Viewer()
    viewer.add_image(bmode, name='Image', colormap='gray', blending='additive', affine=affine)
    viewer.add_image(paramap.T, name='Parametric Map', colormap='jet', blending='additive',
                      opacity=0.7, affine=affine)
    return viewer
