from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from packaging.version import Version

from xarray.core.utils import is_duck_array, is_scalar, module_available

integer_types = (int, np.integer)

if TYPE_CHECKING:
    ModType = Literal["dask", "pint", "cupy", "sparse", "cubed"]
    DuckArrayTypes = tuple[type[Any], ...]  # TODO: improve this? maybe Generic


class DuckArrayModule:
    """
    Solely for internal isinstance and version checks.

    Motivated by having to only import pint when required (as pint currently imports xarray)
    https://github.com/pydata/xarray/pull/5561#discussion_r664815718
    """

    module: ModuleType | None
    version: Version
    type: DuckArrayTypes
    available: bool

    def __init__(self, mod: ModType) -> None:
        duck_array_module: ModuleType | None
        duck_array_version: Version
        duck_array_type: DuckArrayTypes
        try:
            duck_array_module = import_module(mod)
            duck_array_version = Version(duck_array_module.__version__)

            if mod == "dask":
                duck_array_type = (import_module("dask.array").Array,)
            elif mod == "pint":
                duck_array_type = (duck_array_module.Quantity,)
            elif mod == "cupy":
                duck_array_type = (duck_array_module.ndarray,)
            elif mod == "sparse":
                duck_array_type = (duck_array_module.SparseArray,)
            elif mod == "cubed":
                duck_array_type = (duck_array_module.Array,)
            else:
                raise NotImplementedError

        except (ImportError, AttributeError):  # pragma: no cover
            duck_array_module = None
            duck_array_version = Version("0.0.0")
            duck_array_type = ()

        self.module = duck_array_module
        self.version = duck_array_version
        self.type = duck_array_type
        self.available = duck_array_module is not None


_cached_duck_array_modules: dict[ModType, DuckArrayModule] = {}


def _get_cached_duck_array_module(mod: ModType) -> DuckArrayModule:
    if mod not in _cached_duck_array_modules:
        duckmod = DuckArrayModule(mod)
        _cached_duck_array_modules[mod] = duckmod
        return duckmod
    else:
        return _cached_duck_array_modules[mod]


def array_type(mod: ModType) -> DuckArrayTypes:
    """Quick wrapper to get the array class of the module."""
    return _get_cached_duck_array_module(mod).type


def mod_version(mod: ModType) -> Version:
    """Quick wrapper to get the version of the module."""
    return _get_cached_duck_array_module(mod).version


def is_dask_collection(x):
    if module_available("dask"):
        from dask.base import is_dask_collection

        return is_dask_collection(x)
    return False


def is_duck_dask_array(x):
    return is_duck_array(x) and is_dask_collection(x)


def is_chunked_array(x) -> bool:
    return is_duck_dask_array(x) or (is_duck_array(x) and hasattr(x, "chunks"))


def is_0d_dask_array(x):
    return is_duck_dask_array(x) and is_scalar(x)


def to_numpy(data) -> np.ndarray:
    from xarray.core.indexing import ExplicitlyIndexed
    from xarray.core.parallelcompat import get_chunked_array_type

    if isinstance(data, ExplicitlyIndexed):
        data = data.get_duck_array()

    # TODO first attempt to call .to_numpy() once some libraries implement it
    if hasattr(data, "chunks"):
        chunkmanager = get_chunked_array_type(data)
        data, *_ = chunkmanager.compute(data)
    if isinstance(data, array_type("cupy")):
        data = data.get()
    # pint has to be imported dynamically as pint imports xarray
    if isinstance(data, array_type("pint")):
        data = data.magnitude
    if isinstance(data, array_type("sparse")):
        data = data.todense()
    data = np.asarray(data)

    return data


def to_duck_array(data):
    from xarray.core.indexing import ExplicitlyIndexed

    if isinstance(data, ExplicitlyIndexed):
        return data.get_duck_array()
    elif is_duck_array(data):
        return data
    else:
        return np.asarray(data)
