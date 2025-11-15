def extensions(*exts: list) -> dict:
    """
    A decorator to specify the acceptable file extensions for a function.

    Args:
        exts (list): List of acceptable file extensions.

    Returns:
        function: The decorated function with the specified extensions.
    """
    def decorator(func):
        func.supported_extensions = exts
        return func
    
    return decorator

def gui_kwargs(*kwargs: list) -> dict:
    """
    A decorator to specify the required keyword arguments for a function to
    be accessible from the GUI.

    Args:
        kwargs (list): List of required keyword arguments.

    Returns:
        function: The decorated function with the specified keyword arguments.
    """
    def decorator(func):
        func.gui_kwargs = kwargs
        return func
    
    return decorator

def default_gui_kwarg_vals(*default_vals: list) -> dict:
    """
    A decorator to specify the default values for the required keyword arguments
    for a function to be accessible from the GUI.

    Args:
        default_vals (list): List of default values for the required keyword arguments.
    """
    def decorator(func):
        func.default_gui_kwarg_vals = default_vals
        return func
    
    return decorator
