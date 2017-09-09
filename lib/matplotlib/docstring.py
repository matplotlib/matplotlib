from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from matplotlib import cbook


class Substitution(object):
    """A decorator that performs %-substitution on an object's docstring.

    This decorator should be robust even if obj.__doc__ is None (for example,
    if -OO was passed to the interpreter)

    Usage: construct a docstring.Substitution with a sequence or dictionary
    suitable for performing substitution; then decorate a suitable function
    with the constructed object, e.g.::

        sub_author_name = Substitution(author='Jason')

        @sub_author_name
        def some_function(x):
            "%(author)s wrote this function"

        # note that some_function.__doc__ is now "Jason wrote this function"

    One can also use positional arguments::

        sub_first_last_names = Substitution('Edgar Allen', 'Poe')

        @sub_first_last_names
        def some_function(x):
            "%s %s wrote the Raven"
    """
    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError("Only positional or keyword args are allowed")
        self.params = args or kwargs

    def __call__(self, func):
        if func.__doc__:
            if six.PY2:
                getattr(func, "im_func", func).__doc__ %= self.params
            else:
                func.__doc__ %= self.params
        return func

    def update(self, *args, **kwargs):
        """Assume self.params is a dict and update it with supplied args."""
        self.params.update(*args, **kwargs)

    @classmethod
    @cbook.deprecated("2.2")
    def from_params(cls, params):
        """
        In the case where the params is a mutable sequence (list or
        dictionary) and it may change before this class is called, one may
        explicitly use a reference to the params rather than using *args or
        **kwargs which will copy the values and not reference them.
        """
        result = cls()
        result.params = params
        return result


@cbook.deprecated("2.2")
class Appender(object):
    """
    A function decorator that will append an addendum to the docstring
    of the target function.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).

    Usage: construct a docstring.Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.

    add_copyright = Appender("Copyright (c) 2009", join='\n')

    @add_copyright
    def my_dog(has='fleas'):
        "This docstring will have a copyright below"
        pass
    """
    def __init__(self, addendum, join=''):
        self.addendum = addendum
        self.join = join

    def __call__(self, func):
        if func.__doc__:
            func.__doc__ = self.join.join([func.__doc__, self.addendum])
        return func


@cbook.deprecated("2.2")
def dedent(func):
    """Dedent a docstring (if present)."""
    if func.__doc__:
        if six.PY2:
            getattr(func, "im_func", func).__doc__ = cbook.dedent(func.__doc__)
        else:
            func.__doc__ = cbook.dedent(func.__doc__)
    return func


@cbook.deprecated("2.2")
def copy(source):
    """A decorator that copies the docstring from the source (if present)."""
    def decorator(target):
        if source.__doc__:
            target.__doc__ = source.__doc__
        return target
    return decorator

# Create a decorator that will house the various documentation that is reused
# throughout Matplotlib.
interpd = Substitution()


def dedent_interpd(func):
    """Decorator that dedents and interpolates an object's docstring.
    """
    if func.__doc__:
        if six.PY2:
            getattr(func, "im_func", func).__doc__ = cbook.dedent(func.__doc__)
        else:
            func.__doc__ = cbook.dedent(func.__doc__)
    return interpd(func)


@cbook.deprecated("2.2")
def copy_dedent(source):
    """A decorator that copies the dedented docstring from the source."""
    def decorator(func):
        if source.__doc__:
            dedent(source)
        return func
    return decorator
