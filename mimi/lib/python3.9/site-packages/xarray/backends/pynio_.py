from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    _normalize_path,
)
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.locks import (
    HDF5_LOCK,
    NETCDFC_LOCK,
    SerializableLock,
    combine_locks,
    ensure_lock,
)
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import Frozen, FrozenDict, close_on_error
from xarray.core.variable import Variable

if TYPE_CHECKING:
    import os
    from io import BufferedIOBase

    from xarray.core.dataset import Dataset

# PyNIO can invoke netCDF libraries internally
# Add a dedicated lock just in case NCL as well isn't thread-safe.
NCL_LOCK = SerializableLock()
PYNIO_LOCK = combine_locks([HDF5_LOCK, NETCDFC_LOCK, NCL_LOCK])


class NioArrayWrapper(BackendArray):
    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name
        array = self.get_array()
        self.shape = array.shape
        self.dtype = np.dtype(array.typecode())

    def get_array(self, needs_lock=True):
        ds = self.datastore._manager.acquire(needs_lock)
        return ds.variables[self.variable_name]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem
        )

    def _getitem(self, key):
        with self.datastore.lock:
            array = self.get_array(needs_lock=False)

            if key == () and self.ndim == 0:
                return array.get_value()

            return array[key]


class NioDataStore(AbstractDataStore):
    """Store for accessing datasets via PyNIO"""

    def __init__(self, filename, mode="r", lock=None, **kwargs):
        import Nio

        warnings.warn(
            "The PyNIO backend is Deprecated and will be removed from Xarray in a future release. "
            "See https://github.com/pydata/xarray/issues/4491 for more information",
            DeprecationWarning,
        )

        if lock is None:
            lock = PYNIO_LOCK
        self.lock = ensure_lock(lock)
        self._manager = CachingFileManager(
            Nio.open_file, filename, lock=lock, mode=mode, kwargs=kwargs
        )
        # xarray provides its own support for FillValue,
        # so turn off PyNIO's support for the same.
        self.ds.set_option("MaskedArrayMode", "MaskedNever")

    @property
    def ds(self):
        return self._manager.acquire()

    def open_store_variable(self, name, var):
        data = indexing.LazilyIndexedArray(NioArrayWrapper(name, self))
        return Variable(var.dimensions, data, var.attributes)

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.variables.items()
        )

    def get_attrs(self):
        return Frozen(self.ds.attributes)

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)

    def get_encoding(self):
        return {
            "unlimited_dims": {k for k in self.ds.dimensions if self.ds.unlimited(k)}
        }

    def close(self):
        self._manager.close()


class PynioBackendEntrypoint(BackendEntrypoint):
    """
    PyNIO backend

        .. deprecated:: 0.20.0

        Deprecated as PyNIO is no longer supported. See
        https://github.com/pydata/xarray/issues/4491 for more information
    """

    def open_dataset(  # type: ignore[override]  # allow LSP violation, not supporting **kwargs
        self,
        filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables: str | Iterable[str] | None = None,
        use_cftime=None,
        decode_timedelta=None,
        mode="r",
        lock=None,
    ) -> Dataset:
        filename_or_obj = _normalize_path(filename_or_obj)
        store = NioDataStore(
            filename_or_obj,
            mode=mode,
            lock=lock,
        )

        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
        return ds


BACKEND_ENTRYPOINTS["pynio"] = ("Nio", PynioBackendEntrypoint)
