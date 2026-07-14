from pathlib import Path
from typing import Tuple

from argparse import ArgumentParser

from ..plugin_utils import discover_plugin_classes, discover_marked_functions, get_external_plugin_root

def data_export_args(parser: ArgumentParser):
    parser.add_argument('data_export_type', type=str, default='',
                        help='Data export type to use. Available data export types: ' + ', '.join(get_data_export_types()[0].keys()))
    parser.add_argument('data_export_path', type=str,
                        help='Path to save exported numerical data to. Must end in .csv or .pkl')
    parser.add_argument('--data_export_kwargs', type=str, default='{}',
                        help='Data export kwargs in JSON format needed for data export class.')
    

def get_data_export_types() -> Tuple[dict, dict]:
    """Get visualization types for the CLI.
    
    Returns:
        dict: Dictionary of visualization types.
        dict: Dictionary of visualization functions for each type.
    """
    current_dir = Path(__file__).parent
    types = discover_plugin_classes(
        current_dir, __package__, "framework", lambda name: f"{name.upper()}Export",
        external_stage="data_export",
    )

    external_root = get_external_plugin_root()
    functions = {}
    for type_name in types:
        methods_path = current_dir / type_name / "export_funcs"
        external_methods_path = external_root / "data_export" / type_name / "export_funcs" if external_root else None
        functions[type_name] = discover_marked_functions(
            __package__, f".{type_name}.export_funcs", methods_path, "required_kwargs",
            external_funcs_dir=external_methods_path,
        )

    return types, functions
