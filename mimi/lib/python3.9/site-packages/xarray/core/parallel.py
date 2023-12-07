from __future__ import annotations

import collections
import itertools
import operator
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from xarray.core.alignment import align
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.pycompat import is_dask_collection

if TYPE_CHECKING:
    from xarray.core.types import T_Xarray


def unzip(iterable):
    return zip(*iterable)


def assert_chunks_compatible(a: Dataset, b: Dataset):
    a = a.unify_chunks()
    b = b.unify_chunks()

    for dim in set(a.chunks).intersection(set(b.chunks)):
        if a.chunks[dim] != b.chunks[dim]:
            raise ValueError(f"Chunk sizes along dimension {dim!r} are not equal.")


def check_result_variables(
    result: DataArray | Dataset, expected: Mapping[str, Any], kind: str
):
    if kind == "coords":
        nice_str = "coordinate"
    elif kind == "data_vars":
        nice_str = "data"

    # check that coords and data variables are as expected
    missing = expected[kind] - set(getattr(result, kind))
    if missing:
        raise ValueError(
            "Result from applying user function does not contain "
            f"{nice_str} variables {missing}."
        )
    extra = set(getattr(result, kind)) - expected[kind]
    if extra:
        raise ValueError(
            "Result from applying user function has unexpected "
            f"{nice_str} variables {extra}."
        )


def dataset_to_dataarray(obj: Dataset) -> DataArray:
    if not isinstance(obj, Dataset):
        raise TypeError(f"Expected Dataset, got {type(obj)}")

    if len(obj.data_vars) > 1:
        raise TypeError(
            "Trying to convert Dataset with more than one data variable to DataArray"
        )

    return next(iter(obj.data_vars.values()))


def dataarray_to_dataset(obj: DataArray) -> Dataset:
    # only using _to_temp_dataset would break
    # func = lambda x: x.to_dataset()
    # since that relies on preserving name.
    if obj.name is None:
        dataset = obj._to_temp_dataset()
    else:
        dataset = obj.to_dataset()
    return dataset


def make_meta(obj):
    """If obj is a DataArray or Dataset, return a new object of the same type and with
    the same variables and dtypes, but where all variables have size 0 and numpy
    backend.
    If obj is neither a DataArray nor Dataset, return it unaltered.
    """
    if isinstance(obj, DataArray):
        obj_array = obj
        obj = dataarray_to_dataset(obj)
    elif isinstance(obj, Dataset):
        obj_array = None
    else:
        return obj

    from dask.array.utils import meta_from_array

    meta = Dataset()
    for name, variable in obj.variables.items():
        meta_obj = meta_from_array(variable.data, ndim=variable.ndim)
        meta[name] = (variable.dims, meta_obj, variable.attrs)
    meta.attrs = obj.attrs
    meta = meta.set_coords(obj.coords)

    if obj_array is not None:
        return dataset_to_dataarray(meta)
    return meta


def infer_template(
    func: Callable[..., T_Xarray], obj: DataArray | Dataset, *args, **kwargs
) -> T_Xarray:
    """Infer return object by running the function on meta objects."""
    meta_args = [make_meta(arg) for arg in (obj,) + args]

    try:
        template = func(*meta_args, **kwargs)
    except Exception as e:
        raise Exception(
            "Cannot infer object returned from running user provided function. "
            "Please supply the 'template' kwarg to map_blocks."
        ) from e

    if not isinstance(template, (Dataset, DataArray)):
        raise TypeError(
            "Function must return an xarray DataArray or Dataset. Instead it returned "
            f"{type(template)}"
        )

    return template


def make_dict(x: DataArray | Dataset) -> dict[Hashable, Any]:
    """Map variable name to numpy(-like) data
    (Dataset.to_dict() is too complicated).
    """
    if isinstance(x, DataArray):
        x = x._to_temp_dataset()

    return {k: v.data for k, v in x.variables.items()}


def _get_chunk_slicer(dim: Hashable, chunk_index: Mapping, chunk_bounds: Mapping):
    if dim in chunk_index:
        which_chunk = chunk_index[dim]
        return slice(chunk_bounds[dim][which_chunk], chunk_bounds[dim][which_chunk + 1])
    return slice(None)


def map_blocks(
    func: Callable[..., T_Xarray],
    obj: DataArray | Dataset,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    template: DataArray | Dataset | None = None,
) -> T_Xarray:
    """Apply a function to each block of a DataArray or Dataset.

    .. warning::
        This function is experimental and its signature may change.

    Parameters
    ----------
    func : callable
        User-provided function that accepts a DataArray or Dataset as its first
        parameter ``obj``. The function will receive a subset or 'block' of ``obj`` (see below),
        corresponding to one chunk along each chunked dimension. ``func`` will be
        executed as ``func(subset_obj, *subset_args, **kwargs)``.

        This function must return either a single DataArray or a single Dataset.

        This function cannot add a new chunked dimension.
    obj : DataArray, Dataset
        Passed to the function as its first argument, one block at a time.
    args : sequence
        Passed to func after unpacking and subsetting any xarray objects by blocks.
        xarray objects in args must be aligned with obj, otherwise an error is raised.
    kwargs : mapping
        Passed verbatim to func after unpacking. xarray objects, if any, will not be
        subset to blocks. Passing dask collections in kwargs is not allowed.
    template : DataArray or Dataset, optional
        xarray object representing the final result after compute is called. If not provided,
        the function will be first run on mocked-up data, that looks like ``obj`` but
        has sizes 0, to determine properties of the returned object such as dtype,
        variable names, attributes, new dimensions and new indexes (if any).
        ``template`` must be provided if the function changes the size of existing dimensions.
        When provided, ``attrs`` on variables in `template` are copied over to the result. Any
        ``attrs`` set by ``func`` will be ignored.

    Returns
    -------
    A single DataArray or Dataset with dask backend, reassembled from the outputs of the
    function.

    Notes
    -----
    This function is designed for when ``func`` needs to manipulate a whole xarray object
    subset to each block. Each block is loaded into memory. In the more common case where
    ``func`` can work on numpy arrays, it is recommended to use ``apply_ufunc``.

    If none of the variables in ``obj`` is backed by dask arrays, calling this function is
    equivalent to calling ``func(obj, *args, **kwargs)``.

    See Also
    --------
    dask.array.map_blocks, xarray.apply_ufunc, xarray.Dataset.map_blocks
    xarray.DataArray.map_blocks

    Examples
    --------
    Calculate an anomaly from climatology using ``.groupby()``. Using
    ``xr.map_blocks()`` allows for parallel operations with knowledge of ``xarray``,
    its indices, and its methods like ``.groupby()``.

    >>> def calculate_anomaly(da, groupby_type="time.month"):
    ...     gb = da.groupby(groupby_type)
    ...     clim = gb.mean(dim="time")
    ...     return gb - clim
    ...
    >>> time = xr.cftime_range("1990-01", "1992-01", freq="ME")
    >>> month = xr.DataArray(time.month, coords={"time": time}, dims=["time"])
    >>> np.random.seed(123)
    >>> array = xr.DataArray(
    ...     np.random.rand(len(time)),
    ...     dims=["time"],
    ...     coords={"time": time, "month": month},
    ... ).chunk()
    >>> array.map_blocks(calculate_anomaly, template=array).compute()
    <xarray.DataArray (time: 24)>
    array([ 0.12894847,  0.11323072, -0.0855964 , -0.09334032,  0.26848862,
            0.12382735,  0.22460641,  0.07650108, -0.07673453, -0.22865714,
           -0.19063865,  0.0590131 , -0.12894847, -0.11323072,  0.0855964 ,
            0.09334032, -0.26848862, -0.12382735, -0.22460641, -0.07650108,
            0.07673453,  0.22865714,  0.19063865, -0.0590131 ])
    Coordinates:
      * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
        month    (time) int64 1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12

    Note that one must explicitly use ``args=[]`` and ``kwargs={}`` to pass arguments
    to the function being applied in ``xr.map_blocks()``:

    >>> array.map_blocks(
    ...     calculate_anomaly,
    ...     kwargs={"groupby_type": "time.year"},
    ...     template=array,
    ... )  # doctest: +ELLIPSIS
    <xarray.DataArray (time: 24)>
    dask.array<<this-array>-calculate_anomaly, shape=(24,), dtype=float64, chunksize=(24,), chunktype=numpy.ndarray>
    Coordinates:
      * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
        month    (time) int64 dask.array<chunksize=(24,), meta=np.ndarray>
    """

    def _wrapper(
        func: Callable,
        args: list,
        kwargs: dict,
        arg_is_array: Iterable[bool],
        expected: dict,
    ):
        """
        Wrapper function that receives datasets in args; converts to dataarrays when necessary;
        passes these to the user function `func` and checks returned objects for expected shapes/sizes/etc.
        """

        converted_args = [
            dataset_to_dataarray(arg) if is_array else arg
            for is_array, arg in zip(arg_is_array, args)
        ]

        result = func(*converted_args, **kwargs)

        # check all dims are present
        missing_dimensions = set(expected["shapes"]) - set(result.sizes)
        if missing_dimensions:
            raise ValueError(
                f"Dimensions {missing_dimensions} missing on returned object."
            )

        # check that index lengths and values are as expected
        for name, index in result._indexes.items():
            if name in expected["shapes"]:
                if result.sizes[name] != expected["shapes"][name]:
                    raise ValueError(
                        f"Received dimension {name!r} of length {result.sizes[name]}. "
                        f"Expected length {expected['shapes'][name]}."
                    )
            if name in expected["indexes"]:
                expected_index = expected["indexes"][name]
                if not index.equals(expected_index):
                    raise ValueError(
                        f"Expected index {name!r} to be {expected_index!r}. Received {index!r} instead."
                    )

        # check that all expected variables were returned
        check_result_variables(result, expected, "coords")
        if isinstance(result, Dataset):
            check_result_variables(result, expected, "data_vars")

        return make_dict(result)

    if template is not None and not isinstance(template, (DataArray, Dataset)):
        raise TypeError(
            f"template must be a DataArray or Dataset. Received {type(template).__name__} instead."
        )
    if not isinstance(args, Sequence):
        raise TypeError("args must be a sequence (for example, a list or tuple).")
    if kwargs is None:
        kwargs = {}
    elif not isinstance(kwargs, Mapping):
        raise TypeError("kwargs must be a mapping (for example, a dict)")

    for value in kwargs.values():
        if is_dask_collection(value):
            raise TypeError(
                "Cannot pass dask collections in kwargs yet. Please compute or "
                "load values before passing to map_blocks."
            )

    if not is_dask_collection(obj):
        return func(obj, *args, **kwargs)

    try:
        import dask
        import dask.array
        from dask.highlevelgraph import HighLevelGraph

    except ImportError:
        pass

    all_args = [obj] + list(args)
    is_xarray = [isinstance(arg, (Dataset, DataArray)) for arg in all_args]
    is_array = [isinstance(arg, DataArray) for arg in all_args]

    # there should be a better way to group this. partition?
    xarray_indices, xarray_objs = unzip(
        (index, arg) for index, arg in enumerate(all_args) if is_xarray[index]
    )
    others = [
        (index, arg) for index, arg in enumerate(all_args) if not is_xarray[index]
    ]

    # all xarray objects must be aligned. This is consistent with apply_ufunc.
    aligned = align(*xarray_objs, join="exact")
    xarray_objs = tuple(
        dataarray_to_dataset(arg) if isinstance(arg, DataArray) else arg
        for arg in aligned
    )

    _, npargs = unzip(
        sorted(list(zip(xarray_indices, xarray_objs)) + others, key=lambda x: x[0])
    )

    # check that chunk sizes are compatible
    input_chunks = dict(npargs[0].chunks)
    input_indexes = dict(npargs[0]._indexes)
    for arg in xarray_objs[1:]:
        assert_chunks_compatible(npargs[0], arg)
        input_chunks.update(arg.chunks)
        input_indexes.update(arg._indexes)

    if template is None:
        # infer template by providing zero-shaped arrays
        template = infer_template(func, aligned[0], *args, **kwargs)
        template_indexes = set(template._indexes)
        preserved_indexes = template_indexes & set(input_indexes)
        new_indexes = template_indexes - set(input_indexes)
        indexes = {dim: input_indexes[dim] for dim in preserved_indexes}
        indexes.update({k: template._indexes[k] for k in new_indexes})
        output_chunks: Mapping[Hashable, tuple[int, ...]] = {
            dim: input_chunks[dim] for dim in template.dims if dim in input_chunks
        }

    else:
        # template xarray object has been provided with proper sizes and chunk shapes
        indexes = dict(template._indexes)
        output_chunks = template.chunksizes
        if not output_chunks:
            raise ValueError(
                "Provided template has no dask arrays. "
                " Please construct a template with appropriately chunked dask arrays."
            )

    for dim in output_chunks:
        if dim in input_chunks and len(input_chunks[dim]) != len(output_chunks[dim]):
            raise ValueError(
                "map_blocks requires that one block of the input maps to one block of output. "
                f"Expected number of output chunks along dimension {dim!r} to be {len(input_chunks[dim])}. "
                f"Received {len(output_chunks[dim])} instead. Please provide template if not provided, or "
                "fix the provided template."
            )

    if isinstance(template, DataArray):
        result_is_array = True
        template_name = template.name
        template = template._to_temp_dataset()
    elif isinstance(template, Dataset):
        result_is_array = False
    else:
        raise TypeError(
            f"func output must be DataArray or Dataset; got {type(template)}"
        )

    # We're building a new HighLevelGraph hlg. We'll have one new layer
    # for each variable in the dataset, which is the result of the
    # func applied to the values.

    graph: dict[Any, Any] = {}
    new_layers: collections.defaultdict[str, dict[Any, Any]] = collections.defaultdict(
        dict
    )
    gname = f"{dask.utils.funcname(func)}-{dask.base.tokenize(npargs[0], args, kwargs)}"

    # map dims to list of chunk indexes
    ichunk = {dim: range(len(chunks_v)) for dim, chunks_v in input_chunks.items()}
    # mapping from chunk index to slice bounds
    input_chunk_bounds = {
        dim: np.cumsum((0,) + chunks_v) for dim, chunks_v in input_chunks.items()
    }
    output_chunk_bounds = {
        dim: np.cumsum((0,) + chunks_v) for dim, chunks_v in output_chunks.items()
    }

    def subset_dataset_to_block(
        graph: dict, gname: str, dataset: Dataset, input_chunk_bounds, chunk_index
    ):
        """
        Creates a task that subsets an xarray dataset to a block determined by chunk_index.
        Block extents are determined by input_chunk_bounds.
        Also subtasks that subset the constituent variables of a dataset.
        """

        # this will become [[name1, variable1],
        #                   [name2, variable2],
        #                   ...]
        # which is passed to dict and then to Dataset
        data_vars = []
        coords = []

        chunk_tuple = tuple(chunk_index.values())
        for name, variable in dataset.variables.items():
            # make a task that creates tuple of (dims, chunk)
            if dask.is_dask_collection(variable.data):
                # recursively index into dask_keys nested list to get chunk
                chunk = variable.__dask_keys__()
                for dim in variable.dims:
                    chunk = chunk[chunk_index[dim]]

                chunk_variable_task = (f"{name}-{gname}-{chunk[0]!r}",) + chunk_tuple
                graph[chunk_variable_task] = (
                    tuple,
                    [variable.dims, chunk, variable.attrs],
                )
            else:
                # non-dask array possibly with dimensions chunked on other variables
                # index into variable appropriately
                subsetter = {
                    dim: _get_chunk_slicer(dim, chunk_index, input_chunk_bounds)
                    for dim in variable.dims
                }
                subset = variable.isel(subsetter)
                chunk_variable_task = (
                    f"{name}-{gname}-{dask.base.tokenize(subset)}",
                ) + chunk_tuple
                graph[chunk_variable_task] = (
                    tuple,
                    [subset.dims, subset, subset.attrs],
                )

            # this task creates dict mapping variable name to above tuple
            if name in dataset._coord_names:
                coords.append([name, chunk_variable_task])
            else:
                data_vars.append([name, chunk_variable_task])

        return (Dataset, (dict, data_vars), (dict, coords), dataset.attrs)

    # iterate over all possible chunk combinations
    for chunk_tuple in itertools.product(*ichunk.values()):
        # mapping from dimension name to chunk index
        chunk_index = dict(zip(ichunk.keys(), chunk_tuple))

        blocked_args = [
            subset_dataset_to_block(graph, gname, arg, input_chunk_bounds, chunk_index)
            if isxr
            else arg
            for isxr, arg in zip(is_xarray, npargs)
        ]

        # expected["shapes", "coords", "data_vars", "indexes"] are used to
        # raise nice error messages in _wrapper
        expected = {}
        # input chunk 0 along a dimension maps to output chunk 0 along the same dimension
        # even if length of dimension is changed by the applied function
        expected["shapes"] = {
            k: output_chunks[k][v] for k, v in chunk_index.items() if k in output_chunks
        }
        expected["data_vars"] = set(template.data_vars.keys())  # type: ignore[assignment]
        expected["coords"] = set(template.coords.keys())  # type: ignore[assignment]
        expected["indexes"] = {
            dim: indexes[dim][_get_chunk_slicer(dim, chunk_index, output_chunk_bounds)]
            for dim in indexes
        }

        from_wrapper = (gname,) + chunk_tuple
        graph[from_wrapper] = (_wrapper, func, blocked_args, kwargs, is_array, expected)

        # mapping from variable name to dask graph key
        var_key_map: dict[Hashable, str] = {}
        for name, variable in template.variables.items():
            if name in indexes:
                continue
            gname_l = f"{name}-{gname}"
            var_key_map[name] = gname_l

            key: tuple[Any, ...] = (gname_l,)
            for dim in variable.dims:
                if dim in chunk_index:
                    key += (chunk_index[dim],)
                else:
                    # unchunked dimensions in the input have one chunk in the result
                    # output can have new dimensions with exactly one chunk
                    key += (0,)

            # We're adding multiple new layers to the graph:
            # The first new layer is the result of the computation on
            # the array.
            # Then we add one layer per variable, which extracts the
            # result for that variable, and depends on just the first new
            # layer.
            new_layers[gname_l][key] = (operator.getitem, from_wrapper, name)

    hlg = HighLevelGraph.from_collections(
        gname,
        graph,
        dependencies=[arg for arg in npargs if dask.is_dask_collection(arg)],
    )

    # This adds in the getitems for each variable in the dataset.
    hlg = HighLevelGraph(
        {**hlg.layers, **new_layers},
        dependencies={
            **hlg.dependencies,
            **{name: {gname} for name in new_layers.keys()},
        },
    )

    # TODO: benbovy - flexible indexes: make it work with custom indexes
    # this will need to pass both indexes and coords to the Dataset constructor
    result = Dataset(
        coords={k: idx.to_pandas_index() for k, idx in indexes.items()},
        attrs=template.attrs,
    )

    for index in result._indexes:
        result[index].attrs = template[index].attrs
        result[index].encoding = template[index].encoding

    for name, gname_l in var_key_map.items():
        dims = template[name].dims
        var_chunks = []
        for dim in dims:
            if dim in output_chunks:
                var_chunks.append(output_chunks[dim])
            elif dim in result._indexes:
                var_chunks.append((result.sizes[dim],))
            elif dim in template.dims:
                # new unindexed dimension
                var_chunks.append((template.sizes[dim],))

        data = dask.array.Array(
            hlg, name=gname_l, chunks=var_chunks, dtype=template[name].dtype
        )
        result[name] = (dims, data, template[name].attrs)
        result[name].encoding = template[name].encoding

    result = result.set_coords(template._coord_names)

    if result_is_array:
        da = dataset_to_dataarray(result)
        da.name = template_name
        return da  # type: ignore[return-value]
    return result  # type: ignore[return-value]
