"""
Helper functions for managing the Matplotlib API.

This documentation is only relevant for Matplotlib developers, not for users.

.. warning:

    This module and its submodules are for internal use only.  Do not use them
    in your own code.  We may change the API at any time with no warning.

"""
import inspect
import itertools
import re
import sys
import warnings

from .deprecation import (
    deprecated, warn_deprecated,
    rename_parameter, delete_parameter, make_keyword_only,
    deprecate_method_override, deprecate_privatize_attribute,
    suppress_matplotlib_deprecation_warning,
    MatplotlibDeprecationWarning)


class classproperty:
    """
    Like `property`, but also triggers on access via the class, and it is the
    *class* that's passed as argument.

    Examples
    --------
    ::

        class C:
            @classproperty
            def foo(cls):
                return cls.__name__

        assert C.foo == "C"
    """

    def __init__(self, fget, fset=None, fdel=None, doc=None):
        self._fget = fget
        if fset is not None or fdel is not None:
            raise ValueError('classproperty only implements fget.')
        self.fset = fset
        self.fdel = fdel
        # docs are ignored for now
        self._doc = doc

    def __get__(self, instance, owner):
        return self._fget(owner)

    @property
    def fget(self):
        return self._fget


def check_in_list(_values, *, _print_supported_values=True, **kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* is in *_values*.

    Parameters
    ----------
    _values : iterable
        Sequence of values to check on.
    _print_supported_values : bool, default: True
        Whether to print *_values* when raising ValueError.
    **kwargs : dict
        *key, value* pairs as keyword arguments to find in *_values*.

    Raises
    ------
    ValueError
        If any *value* in *kwargs* is not found in *_values*.

    Examples
    --------
    >>> _api.check_in_list(["foo", "bar"], arg=arg, other_arg=other_arg)
    """
    values = _values
    for key, val in kwargs.items():
        if val not in values:
            if _print_supported_values:
                raise ValueError(
                    f"{val!r} is not a valid value for {key}; "
                    f"supported values are {', '.join(map(repr, values))}")
            else:
                raise ValueError(f"{val!r} is not a valid value for {key}")


def check_shape(_shape, **kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* has the shape
    *_shape*, if not, raise an appropriate ValueError.

    *None* in the shape is treated as a "free" size that can have any length.
    e.g. (None, 2) -> (N, 2)

    The values checked must be numpy arrays.

    Examples
    --------
    To check for (N, 2) shaped arrays

    >>> _api.check_shape((None, 2), arg=arg, other_arg=other_arg)
    """
    target_shape = _shape
    for k, v in kwargs.items():
        data_shape = v.shape

        if len(target_shape) != len(data_shape) or any(
                t not in [s, None]
                for t, s in zip(target_shape, data_shape)
        ):
            dim_labels = iter(itertools.chain(
                'MNLIJKLH',
                (f"D{i}" for i in itertools.count())))
            text_shape = ", ".join((str(n)
                                    if n is not None
                                    else next(dim_labels)
                                    for n in target_shape))

            raise ValueError(
                f"{k!r} must be {len(target_shape)}D "
                f"with shape ({text_shape}). "
                f"Your input has shape {v.shape}."
            )


def check_getitem(_mapping, **kwargs):
    """
    *kwargs* must consist of a single *key, value* pair.  If *key* is in
    *_mapping*, return ``_mapping[value]``; else, raise an appropriate
    ValueError.

    Examples
    --------
    >>> _api.check_getitem({"foo": "bar"}, arg=arg)
    """
    mapping = _mapping
    if len(kwargs) != 1:
        raise ValueError("check_getitem takes a single keyword argument")
    (k, v), = kwargs.items()
    try:
        return mapping[v]
    except KeyError:
        raise ValueError(
            "{!r} is not a valid value for {}; supported values are {}"
            .format(v, k, ', '.join(map(repr, mapping)))) from None


def warn_external(message, category=None):
    """
    `warnings.warn` wrapper that sets *stacklevel* to "outside Matplotlib".

    The original emitter of the warning can be obtained by patching this
    function back to `warnings.warn`, i.e. ``_api.warn_external =
    warnings.warn`` (or ``functools.partial(warnings.warn, stacklevel=2)``,
    etc.).
    """
    frame = sys._getframe()
    for stacklevel in itertools.count(1):  # lgtm[py/unused-loop-variable]
        if frame is None:
            # when called in embedded context may hit frame is None
            break
        if not re.match(r"\A(matplotlib|mpl_toolkits)(\Z|\.(?!tests\.))",
                        # Work around sphinx-gallery not setting __name__.
                        frame.f_globals.get("__name__", "")):
            break
        frame = frame.f_back
    warnings.warn(message, category, stacklevel)


def validate_arg_types(arg_names, cls):
    """
    A decorator that converts the arguments given by *arg_names* to *cls*, and
    raises an error if that's not possible.

    Notes
    -----
    - Each argument in *arg_names* is casted to cls(argument) before the
      original function is called.
    - The default value in the function signature is allowed through, even if
      it cannot be cast to *cls*. As an example, this is helpful when using
      `None` to denote no value being passed.
    """
    from matplotlib.rcsetup import _make_type_validator
    validator = _make_type_validator(cls)

    def outer(func):
        """
        *func* is the decorated function.
        """
        sig = inspect.signature(func)
        for arg_name in arg_names:
            if arg_name not in sig.parameters:
                raise ValueError(f'Argument name {arg_name} not in function signature')

        def inner(*args, **kwargs):
            """
            *args* and *kwargs* are those passed to the decorated function.
            """
            sig_bound = sig.bind(*args, **kwargs)
            for arg_name in arg_names:
                if arg_name not in sig_bound.arguments:
                    continue
                val = sig_bound.arguments[arg_name]
                if val == sig.parameters[arg_name].default:
                    # Allow the default value through
                    continue
                try:
                    # Try to cast to the desired class
                    sig_bound.arguments[arg_name] = cls(val)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Could not convert argument '{arg_name}' "
                                     f"with value '{val}' "
                                     f"to type {cls}.") from e
            return func(*sig_bound.args, **sig_bound.kwargs)

        return inner

    return outer
