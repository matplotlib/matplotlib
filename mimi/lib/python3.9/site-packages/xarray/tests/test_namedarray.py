from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, cast, overload

import numpy as np
import pytest

from xarray.core.indexing import ExplicitlyIndexed
from xarray.namedarray._typing import _arrayfunction_or_api, _DType_co, _ShapeType_co
from xarray.namedarray.core import NamedArray, from_array
from xarray.namedarray.utils import _default

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from xarray.namedarray._typing import (
        _AttrsLike,
        _DimsLike,
        _DType,
        _Shape,
        duckarray,
    )
    from xarray.namedarray.utils import Default


class CustomArrayBase(Generic[_ShapeType_co, _DType_co]):
    def __init__(self, array: duckarray[Any, _DType_co]) -> None:
        self.array: duckarray[Any, _DType_co] = array

    @property
    def dtype(self) -> _DType_co:
        return self.array.dtype

    @property
    def shape(self) -> _Shape:
        return self.array.shape


class CustomArray(
    CustomArrayBase[_ShapeType_co, _DType_co], Generic[_ShapeType_co, _DType_co]
):
    def __array__(self) -> np.ndarray[Any, np.dtype[np.generic]]:
        return np.array(self.array)


class CustomArrayIndexable(
    CustomArrayBase[_ShapeType_co, _DType_co],
    ExplicitlyIndexed,
    Generic[_ShapeType_co, _DType_co],
):
    def __array_namespace__(self) -> ModuleType:
        return np


@pytest.fixture
def random_inputs() -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))


def test_namedarray_init() -> None:
    dtype = np.dtype(np.int8)
    expected = np.array([1, 2], dtype=dtype)
    actual: NamedArray[Any, np.dtype[np.int8]]
    actual = NamedArray(("x",), expected)
    assert np.array_equal(np.asarray(actual.data), expected)

    with pytest.raises(AttributeError):
        expected2 = [1, 2]
        actual2: NamedArray[Any, Any]
        actual2 = NamedArray(("x",), expected2)  # type: ignore[arg-type]
        assert np.array_equal(np.asarray(actual2.data), expected2)


@pytest.mark.parametrize(
    "dims, data, expected, raise_error",
    [
        (("x",), [1, 2, 3], np.array([1, 2, 3]), False),
        ((1,), np.array([4, 5, 6]), np.array([4, 5, 6]), False),
        ((), 2, np.array(2), False),
        # Fail:
        (("x",), NamedArray("time", np.array([1, 2, 3])), np.array([1, 2, 3]), True),
    ],
)
def test_from_array(
    dims: _DimsLike,
    data: ArrayLike,
    expected: np.ndarray[Any, Any],
    raise_error: bool,
) -> None:
    actual: NamedArray[Any, Any]
    if raise_error:
        with pytest.raises(TypeError, match="already a Named array"):
            actual = from_array(dims, data)

            # Named arrays are not allowed:
            from_array(actual)  # type: ignore[call-overload]
    else:
        actual = from_array(dims, data)

        assert np.array_equal(np.asarray(actual.data), expected)


def test_from_array_with_masked_array() -> None:
    masked_array: np.ndarray[Any, np.dtype[np.generic]]
    masked_array = np.ma.array([1, 2, 3], mask=[False, True, False])  # type: ignore[no-untyped-call]
    with pytest.raises(NotImplementedError):
        from_array(("x",), masked_array)


def test_from_array_with_0d_object() -> None:
    data = np.empty((), dtype=object)
    data[()] = (10, 12, 12)
    narr = from_array((), data)
    np.array_equal(np.asarray(narr.data), data)


# TODO: Make xr.core.indexing.ExplicitlyIndexed pass as a subclass of_arrayfunction_or_api
# and remove this test.
def test_from_array_with_explicitly_indexed(
    random_inputs: np.ndarray[Any, Any]
) -> None:
    array: CustomArray[Any, Any]
    array = CustomArray(random_inputs)
    output: NamedArray[Any, Any]
    output = from_array(("x", "y", "z"), array)
    assert isinstance(output.data, np.ndarray)

    array2: CustomArrayIndexable[Any, Any]
    array2 = CustomArrayIndexable(random_inputs)
    output2: NamedArray[Any, Any]
    output2 = from_array(("x", "y", "z"), array2)
    assert isinstance(output2.data, CustomArrayIndexable)


def test_properties() -> None:
    data = 0.5 * np.arange(10).reshape(2, 5)
    named_array: NamedArray[Any, Any]
    named_array = NamedArray(["x", "y"], data, {"key": "value"})
    assert named_array.dims == ("x", "y")
    assert np.array_equal(np.asarray(named_array.data), data)
    assert named_array.attrs == {"key": "value"}
    assert named_array.ndim == 2
    assert named_array.sizes == {"x": 2, "y": 5}
    assert named_array.size == 10
    assert named_array.nbytes == 80
    assert len(named_array) == 2


def test_attrs() -> None:
    named_array: NamedArray[Any, Any]
    named_array = NamedArray(["x", "y"], np.arange(10).reshape(2, 5))
    assert named_array.attrs == {}
    named_array.attrs["key"] = "value"
    assert named_array.attrs == {"key": "value"}
    named_array.attrs = {"key": "value2"}
    assert named_array.attrs == {"key": "value2"}


def test_data(random_inputs: np.ndarray[Any, Any]) -> None:
    named_array: NamedArray[Any, Any]
    named_array = NamedArray(["x", "y", "z"], random_inputs)
    assert np.array_equal(np.asarray(named_array.data), random_inputs)
    with pytest.raises(ValueError):
        named_array.data = np.random.random((3, 4)).astype(np.float64)


def test_real_and_imag() -> None:
    expected_real: np.ndarray[Any, np.dtype[np.float64]]
    expected_real = np.arange(3, dtype=np.float64)

    expected_imag: np.ndarray[Any, np.dtype[np.float64]]
    expected_imag = -np.arange(3, dtype=np.float64)

    arr: np.ndarray[Any, np.dtype[np.complex128]]
    arr = expected_real + 1j * expected_imag

    named_array: NamedArray[Any, np.dtype[np.complex128]]
    named_array = NamedArray(["x"], arr)

    actual_real: duckarray[Any, np.dtype[np.float64]] = named_array.real.data
    assert np.array_equal(np.asarray(actual_real), expected_real)
    assert actual_real.dtype == expected_real.dtype

    actual_imag: duckarray[Any, np.dtype[np.float64]] = named_array.imag.data
    assert np.array_equal(np.asarray(actual_imag), expected_imag)
    assert actual_imag.dtype == expected_imag.dtype


# Additional tests as per your original class-based code
@pytest.mark.parametrize(
    "data, dtype",
    [
        ("foo", np.dtype("U3")),
        (b"foo", np.dtype("S3")),
    ],
)
def test_0d_string(data: Any, dtype: DTypeLike) -> None:
    named_array: NamedArray[Any, Any]
    named_array = from_array([], data)
    assert named_array.data == data
    assert named_array.dims == ()
    assert named_array.sizes == {}
    assert named_array.attrs == {}
    assert named_array.ndim == 0
    assert named_array.size == 1
    assert named_array.dtype == dtype


def test_0d_object() -> None:
    named_array: NamedArray[Any, Any]
    named_array = from_array([], (10, 12, 12))
    expected_data = np.empty((), dtype=object)
    expected_data[()] = (10, 12, 12)
    assert np.array_equal(np.asarray(named_array.data), expected_data)

    assert named_array.dims == ()
    assert named_array.sizes == {}
    assert named_array.attrs == {}
    assert named_array.ndim == 0
    assert named_array.size == 1
    assert named_array.dtype == np.dtype("O")


def test_0d_datetime() -> None:
    named_array: NamedArray[Any, Any]
    named_array = from_array([], np.datetime64("2000-01-01"))
    assert named_array.dtype == np.dtype("datetime64[D]")


@pytest.mark.parametrize(
    "timedelta, expected_dtype",
    [
        (np.timedelta64(1, "D"), np.dtype("timedelta64[D]")),
        (np.timedelta64(1, "s"), np.dtype("timedelta64[s]")),
        (np.timedelta64(1, "m"), np.dtype("timedelta64[m]")),
        (np.timedelta64(1, "h"), np.dtype("timedelta64[h]")),
        (np.timedelta64(1, "us"), np.dtype("timedelta64[us]")),
        (np.timedelta64(1, "ns"), np.dtype("timedelta64[ns]")),
        (np.timedelta64(1, "ps"), np.dtype("timedelta64[ps]")),
        (np.timedelta64(1, "fs"), np.dtype("timedelta64[fs]")),
        (np.timedelta64(1, "as"), np.dtype("timedelta64[as]")),
    ],
)
def test_0d_timedelta(
    timedelta: np.timedelta64, expected_dtype: np.dtype[np.timedelta64]
) -> None:
    named_array: NamedArray[Any, Any]
    named_array = from_array([], timedelta)
    assert named_array.dtype == expected_dtype
    assert named_array.data == timedelta


@pytest.mark.parametrize(
    "dims, data_shape, new_dims, raises",
    [
        (["x", "y", "z"], (2, 3, 4), ["a", "b", "c"], False),
        (["x", "y", "z"], (2, 3, 4), ["a", "b"], True),
        (["x", "y", "z"], (2, 4, 5), ["a", "b", "c", "d"], True),
        ([], [], (), False),
        ([], [], ("x",), True),
    ],
)
def test_dims_setter(dims: Any, data_shape: Any, new_dims: Any, raises: bool) -> None:
    named_array: NamedArray[Any, Any]
    named_array = NamedArray(dims, np.asarray(np.random.random(data_shape)))
    assert named_array.dims == tuple(dims)
    if raises:
        with pytest.raises(ValueError):
            named_array.dims = new_dims
    else:
        named_array.dims = new_dims
        assert named_array.dims == tuple(new_dims)


def test_duck_array_class() -> None:
    def test_duck_array_typevar(a: duckarray[Any, _DType]) -> duckarray[Any, _DType]:
        # Mypy checks a is valid:
        b: duckarray[Any, _DType] = a

        # Runtime check if valid:
        if isinstance(b, _arrayfunction_or_api):
            return b
        else:
            raise TypeError(f"a ({type(a)}) is not a valid _arrayfunction or _arrayapi")

    numpy_a: NDArray[np.int64]
    numpy_a = np.array([2.1, 4], dtype=np.dtype(np.int64))
    test_duck_array_typevar(numpy_a)

    masked_a: np.ma.MaskedArray[Any, np.dtype[np.int64]]
    masked_a = np.ma.asarray([2.1, 4], dtype=np.dtype(np.int64))  # type: ignore[no-untyped-call]
    test_duck_array_typevar(masked_a)

    custom_a: CustomArrayIndexable[Any, np.dtype[np.int64]]
    custom_a = CustomArrayIndexable(numpy_a)
    test_duck_array_typevar(custom_a)

    # Test numpy's array api:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            r"The numpy.array_api submodule is still experimental",
            category=UserWarning,
        )
        import numpy.array_api as nxp

    # TODO: nxp doesn't use dtype typevars, so can only use Any for the moment:
    arrayapi_a: duckarray[Any, Any]  #  duckarray[Any, np.dtype[np.int64]]
    arrayapi_a = nxp.asarray([2.1, 4], dtype=np.dtype(np.int64))
    test_duck_array_typevar(arrayapi_a)


def test_new_namedarray() -> None:
    dtype_float = np.dtype(np.float32)
    narr_float: NamedArray[Any, np.dtype[np.float32]]
    narr_float = NamedArray(("x",), np.array([1.5, 3.2], dtype=dtype_float))
    assert narr_float.dtype == dtype_float

    dtype_int = np.dtype(np.int8)
    narr_int: NamedArray[Any, np.dtype[np.int8]]
    narr_int = narr_float._new(("x",), np.array([1, 3], dtype=dtype_int))
    assert narr_int.dtype == dtype_int

    # Test with a subclass:
    class Variable(
        NamedArray[_ShapeType_co, _DType_co], Generic[_ShapeType_co, _DType_co]
    ):
        @overload
        def _new(
            self,
            dims: _DimsLike | Default = ...,
            data: duckarray[Any, _DType] = ...,
            attrs: _AttrsLike | Default = ...,
        ) -> Variable[Any, _DType]:
            ...

        @overload
        def _new(
            self,
            dims: _DimsLike | Default = ...,
            data: Default = ...,
            attrs: _AttrsLike | Default = ...,
        ) -> Variable[_ShapeType_co, _DType_co]:
            ...

        def _new(
            self,
            dims: _DimsLike | Default = _default,
            data: duckarray[Any, _DType] | Default = _default,
            attrs: _AttrsLike | Default = _default,
        ) -> Variable[Any, _DType] | Variable[_ShapeType_co, _DType_co]:
            dims_ = copy.copy(self._dims) if dims is _default else dims

            attrs_: Mapping[Any, Any] | None
            if attrs is _default:
                attrs_ = None if self._attrs is None else self._attrs.copy()
            else:
                attrs_ = attrs

            if data is _default:
                return type(self)(dims_, copy.copy(self._data), attrs_)
            else:
                cls_ = cast("type[Variable[Any, _DType]]", type(self))
                return cls_(dims_, data, attrs_)

    var_float: Variable[Any, np.dtype[np.float32]]
    var_float = Variable(("x",), np.array([1.5, 3.2], dtype=dtype_float))
    assert var_float.dtype == dtype_float

    var_int: Variable[Any, np.dtype[np.int8]]
    var_int = var_float._new(("x",), np.array([1, 3], dtype=dtype_int))
    assert var_int.dtype == dtype_int


def test_replace_namedarray() -> None:
    dtype_float = np.dtype(np.float32)
    np_val: np.ndarray[Any, np.dtype[np.float32]]
    np_val = np.array([1.5, 3.2], dtype=dtype_float)
    np_val2: np.ndarray[Any, np.dtype[np.float32]]
    np_val2 = 2 * np_val

    narr_float: NamedArray[Any, np.dtype[np.float32]]
    narr_float = NamedArray(("x",), np_val)
    assert narr_float.dtype == dtype_float

    narr_float2: NamedArray[Any, np.dtype[np.float32]]
    narr_float2 = NamedArray(("x",), np_val2)
    assert narr_float2.dtype == dtype_float

    # Test with a subclass:
    class Variable(
        NamedArray[_ShapeType_co, _DType_co], Generic[_ShapeType_co, _DType_co]
    ):
        @overload
        def _new(
            self,
            dims: _DimsLike | Default = ...,
            data: duckarray[Any, _DType] = ...,
            attrs: _AttrsLike | Default = ...,
        ) -> Variable[Any, _DType]:
            ...

        @overload
        def _new(
            self,
            dims: _DimsLike | Default = ...,
            data: Default = ...,
            attrs: _AttrsLike | Default = ...,
        ) -> Variable[_ShapeType_co, _DType_co]:
            ...

        def _new(
            self,
            dims: _DimsLike | Default = _default,
            data: duckarray[Any, _DType] | Default = _default,
            attrs: _AttrsLike | Default = _default,
        ) -> Variable[Any, _DType] | Variable[_ShapeType_co, _DType_co]:
            dims_ = copy.copy(self._dims) if dims is _default else dims

            attrs_: Mapping[Any, Any] | None
            if attrs is _default:
                attrs_ = None if self._attrs is None else self._attrs.copy()
            else:
                attrs_ = attrs

            if data is _default:
                return type(self)(dims_, copy.copy(self._data), attrs_)
            else:
                cls_ = cast("type[Variable[Any, _DType]]", type(self))
                return cls_(dims_, data, attrs_)

    var_float: Variable[Any, np.dtype[np.float32]]
    var_float = Variable(("x",), np_val)
    assert var_float.dtype == dtype_float

    var_float2: Variable[Any, np.dtype[np.float32]]
    var_float2 = var_float._replace(("x",), np_val2)
    assert var_float2.dtype == dtype_float
