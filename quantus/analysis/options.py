from pathlib import Path
from typing import Tuple

from argparse import ArgumentParser

from ..plugin_utils import discover_plugin_classes, discover_marked_functions, get_external_plugin_root, get_plugin_params, merge_external_plugins

def analysis_args(parser: ArgumentParser):
    parser.add_argument('analysis_type', type=str, default='spectral_paramap',
                        help='Analysis type to complete. Available analysis types: ' + ', '.join(get_analysis_types()[0].keys()))
    parser.add_argument('--analysis_kwargs', type=str, default='{}',
                        help='Analysis kwargs in JSON format needed for analysis class.')
    
def get_required_kwargs(analysis_type: str, analysis_funcs: list) -> dict:
    """Get required kwargs for a given list of analysis functions.

    Args:
        analysis_type (str): the type of analysis to perform.
        analysis_funcs (list): list of analysis functions to apply.

    Returns:
        list: List of required kwargs for the specified analysis functions.
    """
    
    all_analysis_funcs = get_analysis_types()[1]

    # Consider dependencies of analysis functions as well
    for name in analysis_funcs:
        deps = all_analysis_funcs[analysis_type][name].deps if hasattr(all_analysis_funcs[analysis_type][name], 'deps') else []
        for dep in deps:
            if dep not in analysis_funcs:
                analysis_funcs.append(dep)

    # Find all required kwargs with a default value
    required_kwargs = {}
    for name in analysis_funcs:
        for param in get_plugin_params(all_analysis_funcs[analysis_type][name], 'required_kwargs', 'default_kwarg_vals'):
            if param.has_default:
                required_kwargs[param.name] = param.default

    organized_kwargs = {}; set_kwargs = []
    for name in analysis_funcs:
        organized_kwargs[name] = {}
        params = get_plugin_params(all_analysis_funcs[analysis_type][name], 'required_kwargs')
        for param in params:
            if param.name in required_kwargs and param.name not in set_kwargs:
                organized_kwargs[name][param.name] = required_kwargs[param.name]
                set_kwargs.append(param.name)

    return organized_kwargs
    
def get_analysis_types() -> Tuple[dict, dict]:
    """Get analysis types for the CLI.
    
    Returns:
        dict: Dictionary of analysis types.
        dict: Dictionary of analysis functions for each type.
    """
    current_dir = Path(__file__).parent
    types = discover_plugin_classes(
        current_dir, __package__, "framework", lambda name: f"{name.capitalize()}Analysis",
        external_stage="analysis",
    )

    external_root = get_external_plugin_root()
    functions = {}
    for type_name in types:
        methods_path = current_dir / type_name / "analysis_methods"
        external_methods_path = external_root / "analysis" / type_name / "analysis_methods" if external_root else None
        functions[type_name] = discover_marked_functions(
            __package__, f".{type_name}.analysis_methods", methods_path, "outputs",
            external_funcs_dir=external_methods_path,
        )

        # Aggregate functions (@location("aggregate")) live in their own sibling folder,
        # analysis_aggregators/, instead of nested inside analysis_methods/ -- keeps "a folder
        # under analysis_methods/ is one plugin" unambiguous (e.g. homodyned_k_params/) rather
        # than sometimes meaning a whole category of plugins. Discovered the same way as
        # analysis_methods/, then merged into the same dict.
        aggregators_path = current_dir / type_name / "analysis_aggregators"
        external_aggregators_path = external_root / "analysis" / type_name / "analysis_aggregators" if external_root else None
        aggregate_functions = discover_marked_functions(
            __package__, f".{type_name}.analysis_aggregators", aggregators_path, "outputs",
            external_funcs_dir=external_aggregators_path,
        )
        merge_external_plugins(functions[type_name], aggregate_functions, aggregators_path)

    return types, functions
