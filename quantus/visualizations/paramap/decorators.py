from typing import List

def dependencies(*deps: List[str]) -> dict:
    """
    A decorator to specify what methods must be called before this function.

    Args:
        deps (list): List of dependencies required by the function.

    Returns:
        function: The decorated function with the specified dependencies.
    """
    def decorator(func):
        func.deps = deps
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
