from .._core import *

import numpy as np


def _get_data_ptr(arr: np.ndarray) -> int:
    return arr.__array_interface__['data'][0]


def _is_cont(arr: np.ndarray) -> bool:
    return arr.flags['C_CONTIGUOUS']


def _numpy_to_piquant_dtype(dtype: np.dtype) -> QuantDtype:
    if dtype == np.uint8:  # For some reason a dict is not working here
        return QuantDtype.UINT8
    elif dtype == np.int8:
        return QuantDtype.INT8
    elif dtype == np.uint16:
        return QuantDtype.UINT16
    elif dtype == np.int16:
        return QuantDtype.INT16
    elif dtype == np.uint32:
        return QuantDtype.UINT32
    elif dtype == np.int32:
        return QuantDtype.INT32
    elif dtype == np.uint64:
        return QuantDtype.UINT64
    elif dtype == np.int64:
        return QuantDtype.INT64
    elif dtype == np.float32:
        return QuantDtype.F32
    elif dtype == np.float64:
        return QuantDtype.F64
    else:
        raise ValueError(f'Unsupported np dtype: {dtype}')


def compute_quant_config_numpy(
    arr: np.ndarray, *, target_quant_dtype: QuantDtype, ctx: Union[Context, None] = None
) -> Tuple[float, int]:
    """
    Compute the scale and zero point of a tensor.
        :param arr: Input array, must be of type float32.
        :param target_quant_dtype: Data type which the tensor will be quantized to
        :param ctx: Context to use for computation, if None, the default context will be used.
    """
    if ctx is None:
        ctx = Context.default()
    if not _is_cont(arr):
        arr = np.ascontiguousarray(arr)
    assert arr.dtype == np.float32, f'Expected arr of type float32, got {arr.dtype}'
    return ctx.compute_quant_config_raw_ptr(_get_data_ptr(arr), target_quant_dtype, arr.size)


def quantize_numpy(
    in_array: np.ndarray,
    out_array: Union[np.ndarray, None] = None,
    *,
    config: QuantConfig = QuantConfig(),
    ctx: Union[Context, None] = None,
) -> np.ndarray:
    """
    Quantize a np array using the given configuration.
    :param in_array: Input array, must be of type float32.
    :param out_array: Quantized output array, if None, a new array will be created.
    :param config: Quantization configuration, including scale, zero point, and round mode.
    :param ctx: Context to use for quantization, if None, the default context will be used.
    :return: Quantized array.
    """

    if ctx is None:
        ctx = Context.default()

    if in_array.dtype != np.float32:
        in_array = in_array.astype(np.float32)

    if out_array is None:
        out_array = np.empty_like(in_array, dtype=np.uint8)

    if not _is_cont(in_array):
        in_array = np.ascontiguousarray(in_array)
    if not _is_cont(out_array):
        out_array = np.ascontiguousarray(out_array)
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
    in_array: np.ndarray,
    out_array: Union[np.ndarray, None] = None,
    *,
    config: DequantConfig = DequantConfig(),
    ctx: Union[Context, None] = None,
) -> np.ndarray:
    """
    Dequantize a np array using the given configuration.
    :param in_array: Input array. Must be in a quantized format (e.g., uint8).
    :param out_array: Dequantized output array in a dequantized format (e.g. float32). If None, a new array will be created.
    :param config: Dequantization configuration, including scale, zero point, and reduce operation.
    :param ctx: Context to use for dequantization, if None, the default context will be used.
    :return: Dequantized array.
    """

    if ctx is None:
        ctx = Context.default()

    if out_array is None:
        out_array = np.empty_like(in_array, dtype=np.float32)

    if not _is_cont(in_array):
        in_array = np.ascontiguousarray(in_array)

    if not _is_cont(out_array):
        out_array = np.ascontiguousarray(out_array)

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
