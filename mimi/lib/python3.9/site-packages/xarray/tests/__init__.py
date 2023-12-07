from __future__ import annotations

import importlib
import platform
import warnings
from contextlib import contextmanager, nullcontext
from unittest import mock  # noqa: F401

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal  # noqa: F401
from packaging.version import Version
from pandas.testing import assert_frame_equal  # noqa: F401

import xarray.testing
from xarray import Dataset
from xarray.core import utils
from xarray.core.duck_array_ops import allclose_or_equiv  # noqa: F401
from xarray.core.indexing import ExplicitlyIndexed
from xarray.core.options import set_options
from xarray.testing import (  # noqa: F401
    assert_chunks_equal,
    assert_duckarray_allclose,
    assert_duckarray_equal,
)

# import mpl and change the backend before other mpl imports
try:
    import matplotlib as mpl

    # Order of imports is important here.
    # Using a different backend makes Travis CI work
    mpl.use("Agg")
except ImportError:
    pass

# https://github.com/pydata/xarray/issues/7322
warnings.filterwarnings("ignore", "'urllib3.contrib.pyopenssl' module is deprecated")
warnings.filterwarnings("ignore", "Deprecated call to `pkg_resources.declare_namespace")
warnings.filterwarnings("ignore", "pkg_resources is deprecated as an API")

arm_xfail = pytest.mark.xfail(
    platform.machine() == "aarch64" or "arm" in platform.machine(),
    reason="expected failure on ARM",
)


def _importorskip(
    modname: str, minversion: str | None = None
) -> tuple[bool, pytest.MarkDecorator]:
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            if Version(mod.__version__) < Version(minversion):
                raise ImportError("Minimum version not satisfied")
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, func


has_matplotlib, requires_matplotlib = _importorskip("matplotlib")
has_scipy, requires_scipy = _importorskip("scipy")
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="'cgi' is deprecated and slated for removal in Python 3.13",
        category=DeprecationWarning,
    )

    has_pydap, requires_pydap = _importorskip("pydap.client")
has_netCDF4, requires_netCDF4 = _importorskip("netCDF4")
has_h5netcdf, requires_h5netcdf = _importorskip("h5netcdf")
has_pynio, requires_pynio = _importorskip("Nio")
has_cftime, requires_cftime = _importorskip("cftime")
has_dask, requires_dask = _importorskip("dask")
has_bottleneck, requires_bottleneck = _importorskip("bottleneck")
has_rasterio, requires_rasterio = _importorskip("rasterio")
has_zarr, requires_zarr = _importorskip("zarr")
has_fsspec, requires_fsspec = _importorskip("fsspec")
has_iris, requires_iris = _importorskip("iris")
has_numbagg, requires_numbagg = _importorskip("numbagg", "0.4.0")
has_seaborn, requires_seaborn = _importorskip("seaborn")
has_sparse, requires_sparse = _importorskip("sparse")
has_cupy, requires_cupy = _importorskip("cupy")
has_cartopy, requires_cartopy = _importorskip("cartopy")
has_pint, requires_pint = _importorskip("pint")
has_numexpr, requires_numexpr = _importorskip("numexpr")
has_flox, requires_flox = _importorskip("flox")


# some special cases
has_scipy_or_netCDF4 = has_scipy or has_netCDF4
requires_scipy_or_netCDF4 = pytest.mark.skipif(
    not has_scipy_or_netCDF4, reason="requires scipy or netCDF4"
)
# _importorskip does not work for development versions
has_pandas_version_two = Version(pd.__version__).major >= 2
requires_pandas_version_two = pytest.mark.skipif(
    not has_pandas_version_two, reason="requires pandas 2.0.0"
)
has_h5netcdf_ros3 = _importorskip("h5netcdf", "1.3.0")
requires_h5netcdf_ros3 = pytest.mark.skipif(
    not has_h5netcdf_ros3[0], reason="requires h5netcdf 1.3.0"
)

# change some global options for tests
set_options(warn_for_unclosed_files=True)

if has_dask:
    import dask


class CountingScheduler:
    """Simple dask scheduler counting the number of computes.

    Reference: https://stackoverflow.com/questions/53289286/"""

    def __init__(self, max_computes=0):
        self.total_computes = 0
        self.max_computes = max_computes

    def __call__(self, dsk, keys, **kwargs):
        self.total_computes += 1
        if self.total_computes > self.max_computes:
            raise RuntimeError(
                "Too many computes. Total: %d > max: %d."
                % (self.total_computes, self.max_computes)
            )
        return dask.get(dsk, keys, **kwargs)


def raise_if_dask_computes(max_computes=0):
    # return a dummy context manager so that this can be used for non-dask objects
    if not has_dask:
        return nullcontext()
    scheduler = CountingScheduler(max_computes)
    return dask.config.set(scheduler=scheduler)


flaky = pytest.mark.flaky
network = pytest.mark.network


class UnexpectedDataAccess(Exception):
    pass


class InaccessibleArray(utils.NDArrayMixin, ExplicitlyIndexed):
    """Disallows any loading."""

    def __init__(self, array):
        self.array = array

    def get_duck_array(self):
        raise UnexpectedDataAccess("Tried accessing data")

    def __array__(self, dtype: np.typing.DTypeLike = None):
        raise UnexpectedDataAccess("Tried accessing data")

    def __getitem__(self, key):
        raise UnexpectedDataAccess("Tried accessing data.")


class FirstElementAccessibleArray(InaccessibleArray):
    def __getitem__(self, key):
        tuple_idxr = key.tuple
        if len(tuple_idxr) > 1:
            raise UnexpectedDataAccess("Tried accessing more than one element.")
        return self.array[tuple_idxr]


class DuckArrayWrapper(utils.NDArrayMixin):
    """Array-like that prevents casting to array.
    Modeled after cupy."""

    def __init__(self, array: np.ndarray):
        self.array = array

    def __getitem__(self, key):
        return type(self)(self.array[key])

    def __array__(self, dtype: np.typing.DTypeLike = None):
        raise UnexpectedDataAccess("Tried accessing data")

    def __array_namespace__(self):
        """Present to satisfy is_duck_array test."""


class ReturnItem:
    def __getitem__(self, key):
        return key


class IndexerMaker:
    def __init__(self, indexer_cls):
        self._indexer_cls = indexer_cls

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        return self._indexer_cls(key)


def source_ndarray(array):
    """Given an ndarray, return the base object which holds its memory, or the
    object itself.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "DatetimeIndex.base")
        warnings.filterwarnings("ignore", "TimedeltaIndex.base")
        base = getattr(array, "base", np.asarray(array).base)
    if base is None:
        base = array
    return base


@contextmanager
def assert_no_warnings():
    with warnings.catch_warnings(record=True) as record:
        yield record
        assert len(record) == 0, "got unexpected warning(s)"


# Internal versions of xarray's test functions that validate additional
# invariants


def assert_equal(a, b, check_default_indexes=True):
    __tracebackhide__ = True
    xarray.testing.assert_equal(a, b)
    xarray.testing._assert_internal_invariants(a, check_default_indexes)
    xarray.testing._assert_internal_invariants(b, check_default_indexes)


def assert_identical(a, b, check_default_indexes=True):
    __tracebackhide__ = True
    xarray.testing.assert_identical(a, b)
    xarray.testing._assert_internal_invariants(a, check_default_indexes)
    xarray.testing._assert_internal_invariants(b, check_default_indexes)


def assert_allclose(a, b, check_default_indexes=True, **kwargs):
    __tracebackhide__ = True
    xarray.testing.assert_allclose(a, b, **kwargs)
    xarray.testing._assert_internal_invariants(a, check_default_indexes)
    xarray.testing._assert_internal_invariants(b, check_default_indexes)


def create_test_data(seed: int | None = None, add_attrs: bool = True) -> Dataset:
    rs = np.random.RandomState(seed)
    _vars = {
        "var1": ["dim1", "dim2"],
        "var2": ["dim1", "dim2"],
        "var3": ["dim3", "dim1"],
    }
    _dims = {"dim1": 8, "dim2": 9, "dim3": 10}

    obj = Dataset()
    obj["dim2"] = ("dim2", 0.5 * np.arange(_dims["dim2"]))
    obj["dim3"] = ("dim3", list("abcdefghij"))
    obj["time"] = ("time", pd.date_range("2000-01-01", periods=20))
    for v, dims in sorted(_vars.items()):
        data = rs.normal(size=tuple(_dims[d] for d in dims))
        obj[v] = (dims, data)
        if add_attrs:
            obj[v].attrs = {"foo": "variable"}
    obj.coords["numbers"] = (
        "dim3",
        np.array([0, 1, 2, 0, 0, 1, 1, 2, 2, 3], dtype="int64"),
    )
    obj.encoding = {"foo": "bar"}
    assert all(obj.data.flags.writeable for obj in obj.variables.values())
    return obj


_CFTIME_CALENDARS = [
    "365_day",
    "360_day",
    "julian",
    "all_leap",
    "366_day",
    "gregorian",
    "proleptic_gregorian",
    "standard",
]
