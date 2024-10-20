# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""A peculiar method of monkeypatching C++ binding classes with Python methods."""

from __future__ import annotations

import inspect
import platform
from typing import Any, Callable, Protocol, TypeVar


class AugmentedCallable(Protocol):
    """Protocol for any method, with attached booleans."""

    _augment_override_cpp: bool
    _augment_if_no_cpp: bool

    def __call__(self, *args, **kwargs) -> Any:
        """Any function."""  # pragma: no cover


def augment_override_cpp(fn: AugmentedCallable) -> AugmentedCallable:
    """Replace the C++ implementation, if there is one."""
    fn._augment_override_cpp = True
    return fn


def augment_if_no_cpp(fn: AugmentedCallable) -> AugmentedCallable:
    """Provide a Python implementation if no C++ implementation exists."""
    fn._augment_if_no_cpp = True
    return fn


def _is_inherited_method(meth: Callable) -> bool:
    # Augmenting a C++ with a method that cls inherits from the Python
    # object is never what we want.
    return meth.__qualname__.startswith('object.')


def _is_augmentable(m: Any) -> bool:
    return (
        inspect.isfunction(m) and not _is_inherited_method(m)
    ) or inspect.isdatadescriptor(m)


Tcpp = TypeVar('Tcpp')
T = TypeVar('T')


def augments(cls_cpp: type[Tcpp]):
    """Attach methods of a Python support class to an existing class.

    This monkeypatches all methods defined in the support class onto an
    existing class. Example:

    .. code-block:: python

        @augments(ClassDefinedInCpp)
        class SupportClass:
            def foo(self):
                pass

    The Python method 'foo' will be monkeypatched on ClassDefinedInCpp. SupportClass
    has no meaning on its own and should not be used, but gets returned from
    this function so IDE code inspection doesn't get too confused.

    We don't subclass because it's much more convenient to monkeypatch Python
    methods onto the existing Python binding of the C++ class. For one thing,
    this allows the implementation to be moved from Python to C++ or vice
    versa. It saves having to implement an intermediate Python subclass and then
    ensures that the C++ superclass never 'leaks' to pikepdf users. Finally,
    wrapper classes and subclasses can become problematic if the call stack
    crosses the C++/Python boundary multiple times.

    Any existing methods may be used, regardless of whether they are defined
    elsewhere in the support class or in the target class.

    For data fields to work, the target class must be
    tagged ``py::dynamic_attr`` in pybind11.

    Strictly, the target class does not have to be C++ or derived from pybind11.
    This works on pure Python classes too.

    THIS DOES NOT work for class methods.

    (Alternative ideas: https://github.com/pybind/pybind11/issues/1074)
    """
    OVERRIDE_WHITELIST = {'__eq__', '__hash__', '__repr__'}
    if platform.python_implementation() == 'PyPy':
        # Either PyPy or pybind11's interface to PyPy automatically adds a __getattr__
        OVERRIDE_WHITELIST |= {'__getattr__'}  # pragma: no cover

    def class_augment(cls: type[T], cls_cpp: type[Tcpp] = cls_cpp) -> type[T]:
        # inspect.getmembers has different behavior on PyPy - in particular it seems
        # that a typical PyPy class like cls will have more methods that it considers
        # methods than CPython does. Our predicate should take care of this.
        for name, member in inspect.getmembers(cls, predicate=_is_augmentable):
            if name == '__weakref__':
                continue
            if (
                hasattr(cls_cpp, name)
                and hasattr(cls, name)
                and name not in getattr(cls, '__abstractmethods__', set())
                and name not in OVERRIDE_WHITELIST
                and not getattr(getattr(cls, name), '_augment_override_cpp', False)
            ):
                if getattr(getattr(cls, name), '_augment_if_no_cpp', False):
                    # If tagged as "augment if no C++", we only want the binding to be
                    # applied when the primary class does not provide a C++
                    # implementation. Usually this would be a function that not is
                    # provided by pybind11 in some template.
                    continue

                # If the original C++ class and Python support class both define the
                # same name, we generally have a conflict, because this is augmentation
                # not inheritance. However, if the method provided by the support class
                # is an abstract method, then we can consider the C++ version the
                # implementation. Also, pybind11 provides defaults for __eq__,
                # __hash__ and __repr__ that we often do want to override directly.

                raise RuntimeError(
                    f"C++ {cls_cpp} and Python {cls} both define the same "
                    f"non-abstract method {name}: "
                    f"{getattr(cls_cpp, name, '')!r}, "
                    f"{getattr(cls, name, '')!r}"
                )
            if inspect.isfunction(member):
                if hasattr(cls_cpp, name):
                    # If overriding a C++ named method, make a copy of the original
                    # method. This is so that the Python override can call the C++
                    # implementation if it needs to.
                    setattr(cls_cpp, f"_cpp{name}", getattr(cls_cpp, name))
                setattr(cls_cpp, name, member)
                installed_member = getattr(cls_cpp, name)
                installed_member.__qualname__ = member.__qualname__.replace(
                    cls.__name__, cls_cpp.__name__
                )
            elif inspect.isdatadescriptor(member):
                setattr(cls_cpp, name, member)

        def disable_init(self):
            # Prevent initialization of the support class
            raise NotImplementedError(self.__class__.__name__ + '.__init__')

        cls.__init__ = disable_init  # type: ignore
        return cls

    return class_augment
