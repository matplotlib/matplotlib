import inspect

from matplotlib import cbook


class Substitution:
    """
    A decorator that performs %-substitution on an object's docstring.

    This decorator should be robust even if ``obj.__doc__`` is None (for
    example, if -OO was passed to the interpreter).

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
            raise TypeError("Only positional or keyword args are allowed")
        self.params = args or kwargs

    def __call__(self, func):
        if func.__doc__:
            func.__doc__ %= self.params
        return func

    def update(self, *args, **kwargs):
        """
        Update ``self.params`` (which must be a dict) with the supplied args.
        """
        self.params.update(*args, **kwargs)

    @classmethod
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


@cbook.deprecated("3.1")
class Appender:
    r"""
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
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = func.__doc__ and self.join.join(docitems)
        return func


@cbook.deprecated("3.1", alternative="inspect.getdoc()")
def dedent(func):
    "Dedent a docstring (if present)"
    func.__doc__ = func.__doc__ and cbook.dedent(func.__doc__)
    return func


def copy(source):
    "Copy a docstring from another source function (if present)"
    def do_copy(target):
        if source.__doc__:
            target.__doc__ = source.__doc__
        return target
    return do_copy


# Create a decorator that will house the various docstring snippets reused
# throughout Matplotlib.
interpd = Substitution()


def dedent_interpd(func):
    """Dedent *func*'s docstring, then interpolate it with ``interpd``."""
    func.__doc__ = inspect.getdoc(func)
    return interpd(func)


@cbook.deprecated("3.1", alternative="docstring.copy() and cbook.dedent()")
def copy_dedent(source):
    """A decorator that will copy the docstring from the source and
    then dedent it"""
    # note the following is ugly because "Python is not a functional
    # language" - GVR. Perhaps one day, functools.compose will exist.
    #  or perhaps not.
    #  http://mail.python.org/pipermail/patches/2007-February/021687.html
    return lambda target: dedent(copy(source)(target))
