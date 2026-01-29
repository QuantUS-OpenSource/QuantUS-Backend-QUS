import importlib
from pathlib import Path
from typing import Tuple

from argparse import ArgumentParser

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
    
    # Find all required kwargs
    required_kwargs = {}
    for name in analysis_funcs:
        # Consider dependencies of analysis functions as well
        deps = all_analysis_funcs[analysis_type][name].deps if hasattr(all_analysis_funcs[analysis_type][name], 'deps') else []
        for dep in deps:
            if dep not in analysis_funcs:
                analysis_funcs.append(dep)

    for name in analysis_funcs:
        kwargs = all_analysis_funcs[analysis_type][name].required_kwargs if hasattr(all_analysis_funcs[analysis_type][name], 'required_kwargs') else []
        kwarg_vals = all_analysis_funcs[analysis_type][name].default_kwarg_vals if hasattr(all_analysis_funcs[analysis_type][name], 'default_kwarg_vals') else ()
        required_kwargs.update(dict(zip(kwargs, kwarg_vals)))

    organized_kwargs = {}; set_kwargs = []
    for name in analysis_funcs:
        organized_kwargs[name] = {}
        kwargs = all_analysis_funcs[analysis_type][name].required_kwargs if hasattr(all_analysis_funcs[analysis_type][name], 'required_kwargs') else []
        for kw in kwargs:
            if kw in required_kwargs and kw not in set_kwargs:
                organized_kwargs[name][kw] = required_kwargs[kw]
                set_kwargs.append(kw)

    return organized_kwargs
    
def get_analysis_types() -> Tuple[dict, dict]:
    """Get analysis types for the CLI.
    
    Returns:
        dict: Dictionary of analysis types.
        dict: Dictionary of analysis functions for each type.
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
                entry_class = getattr(module, f"{folder.name.capitalize()}Analysis", None)
                if entry_class:
                    types[folder.name] = entry_class
            except ModuleNotFoundError:
                # Handle the case where the module cannot be found
                pass
            
    functions = {}
    for type_name, type_class in types.items():
        methods_path = Path(__file__).parent / type_name / "analysis_methods"
        for file in methods_path.iterdir():
            try:
                module = importlib.import_module(f'.{type_name}.analysis_methods.{file.stem}', package=__package__)
                for name, obj in vars(module).items():
                    try:
                        if callable(obj) and hasattr(obj, 'outputs'):
                            functions[type_name] = functions.get(type_name, {})
                            functions[type_name][name] = obj
                    except (TypeError, KeyError):
                        pass
            except ModuleNotFoundError:
                # Handle the case where the functions module cannot be found
                functions[type_name] = {}
    
    print("DISCOVERED TYPES:", types.keys())
    print("DISCOVERED FUNCTIONS:", {k: list(v.keys()) for k, v in functions.items()})
            
    return types, functions
