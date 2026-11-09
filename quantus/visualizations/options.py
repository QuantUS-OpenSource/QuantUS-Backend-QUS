from pathlib import Path
from typing import List, Tuple

from argparse import ArgumentParser

from ..plugin_utils import discover_plugin_classes, discover_marked_functions, get_external_plugin_root

def visualization_args(parser: ArgumentParser):
    parser.add_argument('visualization_type', type=str, default='paramap_drawing',
                        help='Visualization type to use. Available visualization types: ' + ', '.join(get_visualization_types().keys()))
    parser.add_argument('--visualization_kwargs', type=str, default='{}',
                        help='Visualization kwargs in JSON format needed for visualization class.')
    parser.add_argument('--visualization_output_path', type=str, default='visualizations.pkl',
                        help='Path to output visualization class instance')
    parser.add_argument('--save_visualization_class', type=bool, default=False,
                        help='Save visualization class instance to VISUALIZATION_OUTPUT_PATH')

def get_visualization_types() -> Tuple[dict, dict]:
    """Get visualization types for the CLI.
    
    Returns:
        dict: Dictionary of visualization types.
        dict: Dictionary of visualization functions for each type.
    """
    current_dir = Path(__file__).parent
    types = discover_plugin_classes(
        current_dir, __package__, "framework", lambda name: f"{name.capitalize()}Visualizations",
        external_stage="visualizations",
    )

    external_root = get_external_plugin_root()
    functions = {}
    for type_name in types:
        methods_path = current_dir / type_name / "visualization_funcs"
        external_methods_path = external_root / "visualizations" / type_name / "visualization_funcs" if external_root else None
        functions[type_name] = discover_marked_functions(
            __package__, f".{type_name}.visualization_funcs", methods_path, "deps",
            external_funcs_dir=external_methods_path,
        )

    functions['paramap']['paramaps'] = None # Built-in function

    return types, functions

def get_compatible_funcs(visualization_type: str, analysis_methods: List[str]) -> list:
    """Get compatible visualization functions for a given analysis type and visualization type.
    
    Args:
        visualization_type: The visualization type to check compatibility for.
        analysis_methods: List of analysis methods performed.

    Returns:
        List of compatible visualization function names.
    """
    _, visualization_functions = get_visualization_types()
    compatible_funcs = []
    viz_funcs = visualization_functions.get(visualization_type, {})
    for func_name, func in viz_funcs.items():
        if hasattr(func, 'deps'):
            if set(func.deps) - set(analysis_methods):
                continue
            compatible_funcs.append(func_name)
        else:
            # If no dependencies are specified, consider it compatible
            compatible_funcs.append(func_name)
    return compatible_funcs
