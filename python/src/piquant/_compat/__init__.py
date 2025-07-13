import importlib.util

if importlib.util.find_spec('numpy') is not None:
    try:
        from .api_numpy import *
    except ImportError:
        pass
if importlib.util.find_spec('torch') is not None:
    try:
        from .api_torch import *
    except ImportError:
        pass
