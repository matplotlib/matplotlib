from __future__ import annotations

import warnings
from types import ModuleType
from typing import Any

import numpy as np

from xarray.namedarray._typing import (
    _arrayapi,
    _DType,
    _ScalarType,
    _ShapeType,
    _SupportsImag,
    _SupportsReal,
)
from xarray.namedarray.core import NamedArray

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        r"The numpy.array_api submodule is still experimental",
        category=UserWarning,
    )
    import numpy.array_api as nxp  # noqa: F401


def _get_data_namespace(x: NamedArray[Any, Any]) -> ModuleType:
    if isinstance(x._data, _arrayapi):
        return x._data.__array_namespace__()

    return np


# %% Creation Functions


def astype(
    x: NamedArray[_ShapeType, Any], dtype: _DType, /, *, copy: bool = True
) -> NamedArray[_ShapeType, _DType]:
    """
    Copies an array to a specified data type irrespective of Type Promotion Rules rules.

    Parameters
    ----------
    x : NamedArray
        Array to cast.
    dtype : _DType
        Desired data type.
    copy : bool, optional
        Specifies whether to copy an array when the specified dtype matches the data
        type of the input array x.
        If True, a newly allocated array must always be returned.
        If False and the specified dtype matches the data type of the input array,
        the input array must be returned; otherwise, a newly allocated array must be
        returned. Default: True.

    Returns
    -------
    out : NamedArray
        An array having the specified data type. The returned array must have the
        same shape as x.

    Examples
    --------
    >>> narr = NamedArray(("x",), nxp.asarray([1.5, 2.5]))
    >>> narr
    <xarray.NamedArray (x: 2)>
    Array([1.5, 2.5], dtype=float64)
    >>> astype(narr, np.dtype(np.int32))
    <xarray.NamedArray (x: 2)>
    Array([1, 2], dtype=int32)
    """
    if isinstance(x._data, _arrayapi):
        xp = x._data.__array_namespace__()
        return x._new(data=xp.astype(x._data, dtype, copy=copy))

    # np.astype doesn't exist yet:
    return x._new(data=x._data.astype(dtype, copy=copy))  # type: ignore[attr-defined]


# %% Elementwise Functions


def imag(
    x: NamedArray[_ShapeType, np.dtype[_SupportsImag[_ScalarType]]], /  # type: ignore[type-var]
) -> NamedArray[_ShapeType, np.dtype[_ScalarType]]:
    """
    Returns the imaginary component of a complex number for each element x_i of the
    input array x.

    Parameters
    ----------
    x : NamedArray
        Input array. Should have a complex floating-point data type.

    Returns
    -------
    out : NamedArray
        An array containing the element-wise results. The returned array must have a
        floating-point data type with the same floating-point precision as x
        (e.g., if x is complex64, the returned array must have the floating-point
        data type float32).

    Examples
    --------
    >>> narr = NamedArray(("x",), np.asarray([1.0 + 2j, 2 + 4j]))  # TODO: Use nxp
    >>> imag(narr)
    <xarray.NamedArray (x: 2)>
    array([2., 4.])
    """
    xp = _get_data_namespace(x)
    out = x._new(data=xp.imag(x._data))
    return out


def real(
    x: NamedArray[_ShapeType, np.dtype[_SupportsReal[_ScalarType]]], /  # type: ignore[type-var]
) -> NamedArray[_ShapeType, np.dtype[_ScalarType]]:
    """
    Returns the real component of a complex number for each element x_i of the
    input array x.

    Parameters
    ----------
    x : NamedArray
        Input array. Should have a complex floating-point data type.

    Returns
    -------
    out : NamedArray
        An array containing the element-wise results. The returned array must have a
        floating-point data type with the same floating-point precision as x
        (e.g., if x is complex64, the returned array must have the floating-point
        data type float32).

    Examples
    --------
    >>> narr = NamedArray(("x",), np.asarray([1.0 + 2j, 2 + 4j]))  # TODO: Use nxp
    >>> real(narr)
    <xarray.NamedArray (x: 2)>
    array([1., 2.])
    """
    xp = _get_data_namespace(x)
    out = x._new(data=xp.real(x._data))
    return out
