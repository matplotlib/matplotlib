from __future__ import absolute_import

import numpy as np


# Copy-pasted from numpy.lib.stride_tricks 1.11.2.
def _maybe_view_as_subclass(original_array, new_array):
    if type(original_array) is not type(new_array):
        # if input was an ndarray subclass and subclasses were OK,
        # then view the result as that subclass.
        new_array = new_array.view(type=type(original_array))
        # Since we have done something akin to a view from original_array, we
        # should let the subclass finalize (if it has it implemented, i.e., is
        # not None).
        if new_array.__array_finalize__:
            new_array.__array_finalize__(original_array)
    return new_array


# Copy-pasted from numpy.lib.stride_tricks 1.11.2.
def _broadcast_to(array, shape, subok, readonly):
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    array = np.array(array, copy=False, subok=subok)
    if not shape and array.shape:
        raise ValueError('cannot broadcast a non-scalar to a scalar array')
    if any(size < 0 for size in shape):
        raise ValueError('all elements of broadcast shape must be non-'
                         'negative')
    needs_writeable = not readonly and array.flags.writeable
    extras = ['reduce_ok'] if needs_writeable else []
    op_flag = 'readwrite' if needs_writeable else 'readonly'
    broadcast = np.nditer(
        (array,), flags=['multi_index', 'refs_ok', 'zerosize_ok'] + extras,
        op_flags=[op_flag], itershape=shape, order='C').itviews[0]
    result = _maybe_view_as_subclass(array, broadcast)
    if needs_writeable and not result.flags.writeable:
        result.flags.writeable = True
    return result


# Copy-pasted from numpy.lib.stride_tricks 1.11.2.
def broadcast_to(array, shape, subok=False):
    """Broadcast an array to a new shape.

    Parameters
    ----------
    array : array_like
        The array to broadcast.
    shape : tuple
        The shape of the desired array.
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned array will be forced to be a base-class array (default).

    Returns
    -------
    broadcast : array
        A readonly view on the original array with the given shape. It is
        typically not contiguous. Furthermore, more than one element of a
        broadcasted array may refer to a single memory location.

    Raises
    ------
    ValueError
        If the array is not compatible with the new shape according to NumPy's
        broadcasting rules.

    Notes
    -----
    .. versionadded:: 1.10.0

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> np.broadcast_to(x, (3, 3))
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    """
    return _broadcast_to(array, shape, subok=subok, readonly=True)
