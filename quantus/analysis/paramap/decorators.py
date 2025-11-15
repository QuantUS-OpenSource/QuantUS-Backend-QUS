from typing import List

def dependencies(*deps: List[str]) -> dict:
    """
    A decorator to specify the dependencies of a function.

    Args:
        deps (list): List of dependencies required by the function.

    Returns:
        function: The decorated function with the specified dependencies.
    """
    def decorator(func):
        func.deps = deps
        return func
    return decorator

def supported_spatial_dims(*dims: List[int]) -> dict:
    """
    A decorator to specify the supported spatial dimensions for a function.

    Args:
        dims (list): List of supported spatial dimensions (e.g., 2, 3).

    Returns:
        function: The decorated function with the specified spatial dimensions.
    """
    def decorator(func):
        func.supported_spatial_dims = dims
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

def output_vars(*names: List[str]) -> dict:
    """
    A decorator to specify the variable names written to in ResultsClass.

    Args:
        names (list): List of variable names to be written to in ResultsClass.

    Returns:
        function: The decorated function with the specified variable names.
    """
    def decorator(func):
        func.outputs = names
        return func
    return decorator

def location(*locs: List[str]):
    """
    A decorator to specify the locations a function can be called from in the
    parametric map analysis pipeline. Possible locations include:
    'window', 'full_segmentation'

    Args:
        locs (list): List of locations where the function can be called.
            Defaults to both 'window' and 'full_segmentation'.

    Returns:
        function: The decorated function with the location flag.
    """
    def decorator(func):
        func.location = locs
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
