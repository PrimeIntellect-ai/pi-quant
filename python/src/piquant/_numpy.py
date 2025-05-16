import importlib
from typing import TYPE_CHECKING, Dict
from piquant._quant import *
import numpy

def _get_data_ptr(arr: 'numpy.ndarray') -> int:
    if numpy is None:
        raise ImportError('numpy is not installed')
    return arr.__array_interface__['data'][0]


def _is_cont(arr: 'numpy.ndarray') -> bool:
    if numpy is None:
        raise ImportError('numpy is not installed')
    return arr.flags['C_CONTIGUOUS']


def _numpy_to_piquant_dtype(dtype: 'numpy.dtype') -> QuantDtype:
    if numpy is None:
        raise ImportError('numpy is not installed')
    if dtype == numpy.uint8:  # For some reason a dict is not working here
        return QuantDtype.UINT8
    elif dtype == numpy.int8:
        return QuantDtype.INT8
    elif dtype == numpy.uint16:
        return QuantDtype.UINT16
    elif dtype == numpy.int16:
        return QuantDtype.INT16
    elif dtype == numpy.uint32:
        return QuantDtype.UINT32
    elif dtype == numpy.int32:
        return QuantDtype.INT32
    elif dtype == numpy.uint64:
        return QuantDtype.UINT64
    elif dtype == numpy.int64:
        return QuantDtype.INT64
    elif dtype == numpy.float32:
        return QuantDtype.F32
    elif dtype == numpy.float64:
        return QuantDtype.F64
    else:
        raise ValueError(f'Unsupported numpy dtype: {dtype}')


def compute_quant_config_numpy(
    arr: 'numpy.ndarray', *, target_quant_dtype: QuantDtype, ctx: Union[Context, None] = None
) -> Tuple[float, int]:
    """
    Compute the scale and zero point of a tensor.
        :param arr: Input array, must be of type float32.
        :param target_quant_dtype: Data type which the tensor will be quantized to
        :param ctx: Context to use for computation, if None, the default context will be used.
    """
    if numpy is None:
        raise ImportError('numpy is not installed')
    if ctx is None:
        ctx = Context.default()
    if not _is_cont(arr):
        arr = numpy.ascontiguousarray(arr)
    assert arr.dtype == numpy.float32, f'Expected arr of type float32, got {arr.dtype}'
    return ctx.compute_quant_config_raw_ptr(_get_data_ptr(arr), target_quant_dtype, arr.size)


def quantize_numpy(
    in_array: 'numpy.ndarray',
    out_array: Union['numpy.ndarray', None] = None,
    *,
    config: QuantConfig = QuantConfig(),
    ctx: Union[Context, None] = None,
) -> 'numpy.ndarray':
    """
    Quantize a numpy array using the given configuration.
    :param in_array: Input array, must be of type float32.
    :param out_array: Quantized output array, if None, a new array will be created.
    :param config: Quantization configuration, including scale, zero point, and round mode.
    :param ctx: Context to use for quantization, if None, the default context will be used.
    :return: Quantized array.
    """
    if numpy is None:
        raise ImportError('numpy is not installed')

    if ctx is None:
        ctx = Context.default()

    if in_array.dtype != numpy.float32:
        in_array = in_array.astype(numpy.float32)

    if out_array is None:
        out_array = numpy.empty_like(in_array, dtype=numpy.uint8)

    if not _is_cont(in_array):
        in_array = numpy.ascontiguousarray(in_array)
    if not _is_cont(out_array):
        out_array = numpy.ascontiguousarray(out_array)
    if in_array.size != out_array.size:
        raise ValueError(f'Input and output arrays must have the same size, got {in_array.size} and {out_array.size}')

    ctx.quantize_raw_ptr(
        _get_data_ptr(in_array),
        _numpy_to_piquant_dtype(in_array.dtype),
        _get_data_ptr(out_array),
        _numpy_to_piquant_dtype(out_array.dtype),
        numel=in_array.size,
        scale=config.scale,
        zero_point=config.zero_point,
        round_mode=config.mode,
    )
    return out_array


def dequantize_numpy(
    in_array: 'numpy.ndarray',
    out_array: Union['numpy.ndarray', None] = None,
    *,
    config: DequantConfig = DequantConfig(),
    ctx: Union[Context, None] = None,
) -> 'numpy.ndarray':
    """
    Dequantize a numpy array using the given configuration.
    :param in_array: Input array. Must be in a quantized format (e.g., uint8).
    :param out_array: Dequantized output array in a dequantized format (e.g. float32). If None, a new array will be created.
    :param config: Dequantization configuration, including scale, zero point, and reduce operation.
    :param ctx: Context to use for dequantization, if None, the default context will be used.
    :return: Dequantized array.
    """

    if numpy is None:
        raise ImportError('numpy is not installed')

    if ctx is None:
        ctx = Context.default()

    if out_array is None:
        out_array = numpy.empty_like(in_array, dtype=numpy.float32)

    if not _is_cont(in_array):
        in_array = numpy.ascontiguousarray(in_array)

    if not _is_cont(out_array):
        out_array = numpy.ascontiguousarray(out_array)

    if in_array.size != out_array.size:
        raise ValueError(
            f'Input and output arrays must have the same number of elements: {in_array.size} != {out_array.size}'
        )

    ctx.dequantize_raw_ptr(
        _get_data_ptr(in_array),
        _numpy_to_piquant_dtype(in_array.dtype),
        _get_data_ptr(out_array),
        _numpy_to_piquant_dtype(out_array.dtype),
        numel=in_array.size,
        scale=config.scale,
        zero_point=config.zero_point,
        reduce_op=config.reduce_op,
    )
    return out_array
