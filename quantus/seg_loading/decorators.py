from typing import List

from ..plugin_utils import attr_decorator


def extensions(*exts: List[str]) -> dict:
    """
    A decorator to specify the acceptable file extensions for a function.

    Unlike the other stages' plugin functions, seg loaders are discovered as
    module-level dicts (see quantus/seg_loading/options.py), so this decorator
    wraps the function in a dict instead of setting a plain attribute.

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


required_kwargs = attr_decorator('required_kwargs')
default_kwarg_vals = attr_decorator('default_kwarg_vals')
