import importlib
from pathlib import Path
from typing import List, Tuple

from argparse import ArgumentParser

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
    types = {}
    current_dir = Path(__file__).parent
    for folder in current_dir.iterdir():
        # Check if the item is a directory and not a hidden directory
        if folder.is_dir() and not folder.name.startswith("_"):
            try:
                # Attempt to import the module
                module = importlib.import_module(
                    f".{folder.name}.framework", package=__package__
                )
                entry_class = getattr(module, f"{folder.name.capitalize()}Visualizations", None)
                if entry_class:
                    types[folder.name] = entry_class
            except ModuleNotFoundError as e:
                # Handle the case where the module cannot be found
                print(f"Module quantus.visualizations.{folder.name}.framework could not be found: {e}")
                pass
            
    functions = {}
    for type_name, type_class in types.items():
        try:
            module = importlib.import_module(f'.{type_name}.functions', package=__package__)
            for name, obj in vars(module).items():
                try:
                    if callable(obj) and obj.__module__ == __package__ + f'.{type_name}.functions':
                        if not isinstance(obj, type):
                            functions[type_name] = functions.get(type_name, {})
                            functions[type_name][name] = obj
                except (TypeError, KeyError):
                    pass
        except ModuleNotFoundError as e:
            # Handle the case where the functions module cannot be found
            print(f"Module quantus.visualizations.{type_name}.functions could not be found: {e}")
            functions[type_name] = {}
            
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
