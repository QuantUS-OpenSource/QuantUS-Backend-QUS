from typing import List

def extensions(*exts: List[str]) -> dict:
    """
    A decorator to specify the acceptable file extensions for a function.

    Args:
        exts (list): List of acceptable file extensions.

    Returns:
        function: The decorated function with the specified extensions.
    """
    def decorator(func):
        if type(func) is not dict:
            out_dict = {}
            out_dict['func'] = func
            out_dict['exts'] = exts
            return out_dict
        func['exts'] = exts
        return func
    return decorator

def required_kwargs(*kwarg_names: List[str]) -> dict:
    """
    A decorator to specify the required keyword arguments for a function.

    Args:
        kwarg_names (list): List of required keyword argument names.

    Returns:
        function: The decorated function with the specified keyword arguments.
    """
    def decorator(func):
        func.required_kwargs = kwarg_names
        return func
    return decorator

def default_kwarg_vals(*vals: List[str]):
    """
    A decorator to specify default values for keyword arguments in a function.

    Args:
        vals (list): List of default values for the keyword arguments.

    Returns:
        function: The decorated function with default keyword argument values.
    """
    def decorator(func):
        func.default_kwarg_vals = vals
        return func
    
    return decorator
