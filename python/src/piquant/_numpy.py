import importlib
from typing import TYPE_CHECKING, Tuple, Dict
from piquant._quant import *

if TYPE_CHECKING:
    import numpy
else:
    torch = None
    if importlib.util.find_spec('numpy') is not None:
        import numpy

__numpy_dtype_map: Dict['numpy.target_quant_dtype', QuantDtype] = {
    numpy.uint8: QuantDtype.UINT8,
    numpy.int8: QuantDtype.INT8,
    numpy.uint16: QuantDtype.UINT16,
    numpy.int16: QuantDtype.INT16,
    numpy.uint32: QuantDtype.UINT32,
    numpy.int32: QuantDtype.INT32,
    numpy.uint64: QuantDtype.UINT64,
    numpy.int64: QuantDtype.INT64,
    numpy.float32: QuantDtype.F32,
    numpy.float64: QuantDtype.F64,
}
