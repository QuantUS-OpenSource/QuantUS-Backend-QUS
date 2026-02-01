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
    import sys
    types = {}
    current_dir = Path(__file__).parent
    
    # 1. Check Internal-TUL for analysis types
    project_root = Path(__file__).parents[4]
    internal_analysis_path = project_root / "Internal-TUL" / "QuantUS-QUS" / "analysis"
    
    if internal_analysis_path.exists():
        # Internal modules in QUS depend on quantus.full_workflow from engines/qus/quantus
        qus_engine_root = project_root / "engines" / "qus"
        if qus_engine_root.exists() and str(qus_engine_root) not in sys.path:
            sys.path.append(str(qus_engine_root))

    dirs_to_scan = [(current_dir, __package__)]
    if internal_analysis_path.exists():
        if str(internal_analysis_path) not in sys.path:
            sys.path.append(str(internal_analysis_path))
        dirs_to_scan.append((internal_analysis_path, None))

    for scan_dir, pkg in dirs_to_scan:
        for folder in scan_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith("_"):
                try:
                    if pkg:
                        module = importlib.import_module(f".{folder.name}.framework", package=pkg)
                    else:
                        module = importlib.import_module(f"{folder.name}.framework")
                        
                    entry_class = getattr(module, f"{folder.name.capitalize()}Analysis", None)
                    if entry_class:
                        types[folder.name] = entry_class
                except ModuleNotFoundError:
                    pass
            
    functions = {}
    for type_name, type_class in types.items():
        # Try to find functions in Internal-TUL first, then public
        func_modules = []
        if internal_analysis_path.exists():
            internal_func_path = internal_analysis_path / type_name / "functions.py"
            if internal_func_path.exists():
                func_modules.append((f"{type_name}.functions", None))
        
        func_modules.append((f'.{type_name}.functions', __package__))

        functions[type_name] = {}
        for mod_name, pkg in func_modules:
            try:
                module = importlib.import_module(mod_name, package=pkg)
                for name, obj in vars(module).items():
                    try:
                        if callable(obj) and hasattr(obj, 'outputs'):
                            if name not in functions[type_name]:
                                functions[type_name][name] = obj
                    except (TypeError, KeyError):
                        pass
            except ModuleNotFoundError:
                pass
            
    return types, functions
