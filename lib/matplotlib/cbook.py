"""
A collection of utility functions and classes.  Originally, many
(but not all) were from the Python Cookbook -- hence the name cbook.

This module is safe to import from anywhere within matplotlib;
it imports matplotlib only at runtime.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import xrange, zip
from itertools import repeat
import collections
import datetime
import errno
from functools import reduce
import glob
import gzip
import io
import locale
import os
import re
import sys
import time
import traceback
import types
import warnings
from weakref import ref, WeakKeyDictionary

import numpy as np
import numpy.ma as ma


class MatplotlibDeprecationWarning(UserWarning):
    """
    A class for issuing deprecation warnings for Matplotlib users.

    In light of the fact that Python builtin DeprecationWarnings are ignored
    by default as of Python 2.7 (see link below), this class was put in to
    allow for the signaling of deprecation, but via UserWarnings which are not
    ignored by default.

    https://docs.python.org/dev/whatsnew/2.7.html#the-future-for-python-2-x
    """
    pass

mplDeprecation = MatplotlibDeprecationWarning


def _generate_deprecation_message(since, message='', name='',
                                  alternative='', pending=False,
                                  obj_type='attribute'):

    if not message:
        altmessage = ''

        if pending:
            message = (
                'The %(func)s %(obj_type)s will be deprecated in a '
                'future version.')
        else:
            message = (
                'The %(func)s %(obj_type)s was deprecated in version '
                '%(since)s.')
        if alternative:
            altmessage = ' Use %s instead.' % alternative

        message = ((message % {
            'func': name,
            'name': name,
            'alternative': alternative,
            'obj_type': obj_type,
            'since': since}) +
            altmessage)

    return message


def warn_deprecated(
        since, message='', name='', alternative='', pending=False,
        obj_type='attribute'):
    """
    Used to display deprecation warning in a standard way.

    Parameters
    ----------
    since : str
        The release at which this API became deprecated.

    message : str, optional
        Override the default deprecation message.  The format
        specifier `%(func)s` may be used for the name of the function,
        and `%(alternative)s` may be used in the deprecation message
        to insert the name of an alternative to the deprecated
        function.  `%(obj_type)` may be used to insert a friendly name
        for the type of object being deprecated.

    name : str, optional
        The name of the deprecated function; if not provided the name
        is automatically determined from the passed in function,
        though this is useful in the case of renamed functions, where
        the new function is just assigned to the name of the
        deprecated function.  For example::

            def new_function():
                ...
            oldFunction = new_function

    alternative : str, optional
        An alternative function that the user may use in place of the
        deprecated function.  The deprecation warning will tell the user about
        this alternative if provided.

    pending : bool, optional
        If True, uses a PendingDeprecationWarning instead of a
        DeprecationWarning.

    obj_type : str, optional
        The object type being deprecated.

    Examples
    --------

        Basic example::

            # To warn of the deprecation of "matplotlib.name_of_module"
            warn_deprecated('1.4.0', name='matplotlib.name_of_module',
                            obj_type='module')

    """
    message = _generate_deprecation_message(
        since, message, name, alternative, pending, obj_type)

    warnings.warn(message, mplDeprecation, stacklevel=1)


def deprecated(since, message='', name='', alternative='', pending=False,
               obj_type='function'):
    """
    Decorator to mark a function as deprecated.

    Parameters
    ----------
    since : str
        The release at which this API became deprecated.  This is
        required.

    message : str, optional
        Override the default deprecation message.  The format
        specifier `%(func)s` may be used for the name of the function,
        and `%(alternative)s` may be used in the deprecation message
        to insert the name of an alternative to the deprecated
        function.  `%(obj_type)` may be used to insert a friendly name
        for the type of object being deprecated.

    name : str, optional
        The name of the deprecated function; if not provided the name
        is automatically determined from the passed in function,
        though this is useful in the case of renamed functions, where
        the new function is just assigned to the name of the
        deprecated function.  For example::

            def new_function():
                ...
            oldFunction = new_function

    alternative : str, optional
        An alternative function that the user may use in place of the
        deprecated function.  The deprecation warning will tell the user about
        this alternative if provided.

    pending : bool, optional
        If True, uses a PendingDeprecationWarning instead of a
        DeprecationWarning.

    Examples
    --------

        Basic example::

            @deprecated('1.4.0')
            def the_function_to_deprecate():
                pass

    """
    def deprecate(func, message=message, name=name, alternative=alternative,
                  pending=pending):
        import functools
        import textwrap

        if isinstance(func, classmethod):
            func = func.__func__
            is_classmethod = True
        else:
            is_classmethod = False

        if not name:
            name = func.__name__

        message = _generate_deprecation_message(
            since, message, name, alternative, pending, obj_type)

        @functools.wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(message, mplDeprecation, stacklevel=2)

            return func(*args, **kwargs)

        old_doc = deprecated_func.__doc__
        if not old_doc:
            old_doc = ''
        old_doc = textwrap.dedent(old_doc).strip('\n')
        message = message.strip()
        new_doc = (('\n.. deprecated:: %(since)s'
                    '\n    %(message)s\n\n' %
                    {'since': since, 'message': message}) + old_doc)
        if not old_doc:
            # This is to prevent a spurious 'unexected unindent' warning from
            # docutils when the original docstring was blank.
            new_doc += r'\ '

        deprecated_func.__doc__ = new_doc

        if is_classmethod:
            deprecated_func = classmethod(deprecated_func)
        return deprecated_func

    return deprecate


# On some systems, locale.getpreferredencoding returns None,
# which can break unicode; and the sage project reports that
# some systems have incorrect locale specifications, e.g.,
# an encoding instead of a valid locale name.  Another
# pathological case that has been reported is an empty string.

# On some systems, getpreferredencoding sets the locale, which has
# side effects.  Passing False eliminates those side effects.

def unicode_safe(s):
    import matplotlib

    if isinstance(s, bytes):
        try:
            preferredencoding = locale.getpreferredencoding(
                matplotlib.rcParams['axes.formatter.use_locale']).strip()
            if not preferredencoding:
                preferredencoding = None
        except (ValueError, ImportError, AttributeError):
            preferredencoding = None

        if preferredencoding is None:
            return six.text_type(s)
        else:
            return six.text_type(s, preferredencoding)
    return s


class converter(object):
    """
    Base class for handling string -> python type with support for
    missing values
    """
    def __init__(self, missing='Null', missingval=None):
        self.missing = missing
        self.missingval = missingval

    def __call__(self, s):
        if s == self.missing:
            return self.missingval
        return s

    def is_missing(self, s):
        return not s.strip() or s == self.missing


class tostr(converter):
    """convert to string or None"""
    def __init__(self, missing='Null', missingval=''):
        converter.__init__(self, missing=missing, missingval=missingval)


class todatetime(converter):
    """convert to a datetime or None"""
    def __init__(self, fmt='%Y-%m-%d', missing='Null', missingval=None):
        'use a :func:`time.strptime` format string for conversion'
        converter.__init__(self, missing, missingval)
        self.fmt = fmt

    def __call__(self, s):
        if self.is_missing(s):
            return self.missingval
        tup = time.strptime(s, self.fmt)
        return datetime.datetime(*tup[:6])


class todate(converter):
    """convert to a date or None"""
    def __init__(self, fmt='%Y-%m-%d', missing='Null', missingval=None):
        """use a :func:`time.strptime` format string for conversion"""
        converter.__init__(self, missing, missingval)
        self.fmt = fmt

    def __call__(self, s):
        if self.is_missing(s):
            return self.missingval
        tup = time.strptime(s, self.fmt)
        return datetime.date(*tup[:3])


class tofloat(converter):
    """convert to a float or None"""
    def __init__(self, missing='Null', missingval=None):
        converter.__init__(self, missing)
        self.missingval = missingval

    def __call__(self, s):
        if self.is_missing(s):
            return self.missingval
        return float(s)


class toint(converter):
    """convert to an int or None"""
    def __init__(self, missing='Null', missingval=None):
        converter.__init__(self, missing)

    def __call__(self, s):
        if self.is_missing(s):
            return self.missingval
        return int(s)


class _BoundMethodProxy(object):
    """
    Our own proxy object which enables weak references to bound and unbound
    methods and arbitrary callables. Pulls information about the function,
    class, and instance out of a bound method. Stores a weak reference to the
    instance to support garbage collection.

    @organization: IBM Corporation
    @copyright: Copyright (c) 2005, 2006 IBM Corporation
    @license: The BSD License

    Minor bugfixes by Michael Droettboom
    """
    def __init__(self, cb):
        self._hash = hash(cb)
        self._destroy_callbacks = []
        try:
            try:
                if six.PY3:
                    self.inst = ref(cb.__self__, self._destroy)
                else:
                    self.inst = ref(cb.im_self, self._destroy)
            except TypeError:
                self.inst = None
            if six.PY3:
                self.func = cb.__func__
                self.klass = cb.__self__.__class__
            else:
                self.func = cb.im_func
                self.klass = cb.im_class
        except AttributeError:
            self.inst = None
            self.func = cb
            self.klass = None

    def add_destroy_callback(self, callback):
        self._destroy_callbacks.append(_BoundMethodProxy(callback))

    def _destroy(self, wk):
        for callback in self._destroy_callbacks:
            try:
                callback(self)
            except ReferenceError:
                pass

    def __getstate__(self):
        d = self.__dict__.copy()
        # de-weak reference inst
        inst = d['inst']
        if inst is not None:
            d['inst'] = inst()
        return d

    def __setstate__(self, statedict):
        self.__dict__ = statedict
        inst = statedict['inst']
        # turn inst back into a weakref
        if inst is not None:
            self.inst = ref(inst)

    def __call__(self, *args, **kwargs):
        """
        Proxy for a call to the weak referenced object. Take
        arbitrary params to pass to the callable.

        Raises `ReferenceError`: When the weak reference refers to
        a dead object
        """
        if self.inst is not None and self.inst() is None:
            raise ReferenceError
        elif self.inst is not None:
            # build a new instance method with a strong reference to the
            # instance

            mtd = types.MethodType(self.func, self.inst())

        else:
            # not a bound method, just return the func
            mtd = self.func
        # invoke the callable and return the result
        return mtd(*args, **kwargs)

    def __eq__(self, other):
        """
        Compare the held function and instance with that held by
        another proxy.
        """
        try:
            if self.inst is None:
                return self.func == other.func and other.inst is None
            else:
                return self.func == other.func and self.inst() == other.inst()
        except Exception:
            return False

    def __ne__(self, other):
        """
        Inverse of __eq__.
        """
        return not self.__eq__(other)

    def __hash__(self):
        return self._hash


class CallbackRegistry(object):
    """
    Handle registering and disconnecting for a set of signals and
    callbacks:

        >>> def oneat(x):
        ...    print('eat', x)
        >>> def ondrink(x):
        ...    print('drink', x)

        >>> from matplotlib.cbook import CallbackRegistry
        >>> callbacks = CallbackRegistry()

        >>> id_eat = callbacks.connect('eat', oneat)
        >>> id_drink = callbacks.connect('drink', ondrink)

        >>> callbacks.process('drink', 123)
        drink 123
        >>> callbacks.process('eat', 456)
        eat 456
        >>> callbacks.process('be merry', 456) # nothing will be called
        >>> callbacks.disconnect(id_eat)
        >>> callbacks.process('eat', 456)      # nothing will be called

    In practice, one should always disconnect all callbacks when they
    are no longer needed to avoid dangling references (and thus memory
    leaks).  However, real code in matplotlib rarely does so, and due
    to its design, it is rather difficult to place this kind of code.
    To get around this, and prevent this class of memory leaks, we
    instead store weak references to bound methods only, so when the
    destination object needs to die, the CallbackRegistry won't keep
    it alive.  The Python stdlib weakref module can not create weak
    references to bound methods directly, so we need to create a proxy
    object to handle weak references to bound methods (or regular free
    functions).  This technique was shared by Peter Parente on his
    `"Mindtrove" blog
    <http://mindtrove.info/python-weak-references/>`_.
    """
    def __init__(self):
        self.callbacks = dict()
        self._cid = 0
        self._func_cid_map = {}

    def __getstate__(self):
        # We cannot currently pickle the callables in the registry, so
        # return an empty dictionary.
        return {}

    def __setstate__(self, state):
        # re-initialise an empty callback registry
        self.__init__()

    def connect(self, s, func):
        """
        register *func* to be called when a signal *s* is generated
        func will be called
        """
        self._func_cid_map.setdefault(s, WeakKeyDictionary())
        # Note proxy not needed in python 3.
        # TODO rewrite this when support for python2.x gets dropped.
        proxy = _BoundMethodProxy(func)
        if proxy in self._func_cid_map[s]:
            return self._func_cid_map[s][proxy]

        proxy.add_destroy_callback(self._remove_proxy)
        self._cid += 1
        cid = self._cid
        self._func_cid_map[s][proxy] = cid
        self.callbacks.setdefault(s, dict())
        self.callbacks[s][cid] = proxy
        return cid

    def _remove_proxy(self, proxy):
        for signal, proxies in list(six.iteritems(self._func_cid_map)):
            try:
                del self.callbacks[signal][proxies[proxy]]
            except KeyError:
                pass

            if len(self.callbacks[signal]) == 0:
                del self.callbacks[signal]
                del self._func_cid_map[signal]

    def disconnect(self, cid):
        """
        disconnect the callback registered with callback id *cid*
        """
        for eventname, callbackd in list(six.iteritems(self.callbacks)):
            try:
                del callbackd[cid]
            except KeyError:
                continue
            else:
                for signal, functions in list(
                        six.iteritems(self._func_cid_map)):
                    for function, value in list(six.iteritems(functions)):
                        if value == cid:
                            del functions[function]
                return

    def process(self, s, *args, **kwargs):
        """
        process signal *s*.  All of the functions registered to receive
        callbacks on *s* will be called with *\*args* and *\*\*kwargs*
        """
        if s in self.callbacks:
            for cid, proxy in list(six.iteritems(self.callbacks[s])):
                try:
                    proxy(*args, **kwargs)
                except ReferenceError:
                    self._remove_proxy(proxy)


class silent_list(list):
    """
    override repr when returning a list of matplotlib artists to
    prevent long, meaningless output.  This is meant to be used for a
    homogeneous list of a given type
    """
    def __init__(self, type, seq=None):
        self.type = type
        if seq is not None:
            self.extend(seq)

    def __repr__(self):
        return '<a list of %d %s objects>' % (len(self), self.type)

    def __str__(self):
        return repr(self)

    def __getstate__(self):
        # store a dictionary of this SilentList's state
        return {'type': self.type, 'seq': self[:]}

    def __setstate__(self, state):
        self.type = state['type']
        self.extend(state['seq'])


class IgnoredKeywordWarning(UserWarning):
    """
    A class for issuing warnings about keyword arguments that will be ignored
    by matplotlib
    """
    pass


def local_over_kwdict(local_var, kwargs, *keys):
    """
    Enforces the priority of a local variable over potentially conflicting
    argument(s) from a kwargs dict. The following possible output values are
    considered in order of priority:

        local_var > kwargs[keys[0]] > ... > kwargs[keys[-1]]

    The first of these whose value is not None will be returned. If all are
    None then None will be returned. Each key in keys will be removed from the
    kwargs dict in place.

    Parameters
    ----------
        local_var: any object
            The local variable (highest priority)

        kwargs: dict
            Dictionary of keyword arguments; modified in place

        keys: str(s)
            Name(s) of keyword arguments to process, in descending order of
            priority

    Returns
    -------
        out: any object
            Either local_var or one of kwargs[key] for key in keys

    Raises
    ------
        IgnoredKeywordWarning
            For each key in keys that is removed from kwargs but not used as
            the output value

    """
    out = local_var
    for key in keys:
        kwarg_val = kwargs.pop(key, None)
        if kwarg_val is not None:
            if out is None:
                out = kwarg_val
            else:
                warnings.warn('"%s" keyword argument will be ignored' % key,
                              IgnoredKeywordWarning)
    return out


def strip_math(s):
    """remove latex formatting from mathtext"""
    remove = (r'\mathdefault', r'\rm', r'\cal', r'\tt', r'\it', '\\', '{', '}')
    s = s[1:-1]
    for r in remove:
        s = s.replace(r, '')
    return s


class Bunch(object):
    """
    Often we want to just collect a bunch of stuff together, naming each
    item of the bunch; a dictionary's OK for that, but a small do- nothing
    class is even handier, and prettier to use.  Whenever you want to
    group a few variables::

      >>> point = Bunch(datum=2, squared=4, coord=12)
      >>> point.datum

      By: Alex Martelli
      From: https://code.activestate.com/recipes/121294/
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __repr__(self):
        keys = six.iterkeys(self.__dict__)
        return 'Bunch(%s)' % ', '.join(['%s=%s' % (k, self.__dict__[k])
                                        for k
                                        in keys])


def unique(x):
    """Return a list of unique elements of *x*"""
    return list(six.iterkeys(dict([(val, 1) for val in x])))


def iterable(obj):
    """return true if *obj* is iterable"""
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def is_string_like(obj):
    """Return True if *obj* looks like a string"""
    if isinstance(obj, six.string_types):
        return True
    # numpy strings are subclass of str, ma strings are not
    if ma.isMaskedArray(obj):
        if obj.ndim == 0 and obj.dtype.kind in 'SU':
            return True
        else:
            return False
    try:
        obj + ''
    except:
        return False
    return True


def is_sequence_of_strings(obj):
    """Returns true if *obj* is iterable and contains strings"""
    if not iterable(obj):
        return False
    if is_string_like(obj) and not isinstance(obj, np.ndarray):
        try:
            obj = obj.values
        except AttributeError:
            # not pandas
            return False
    for o in obj:
        if not is_string_like(o):
            return False
    return True


def is_hashable(obj):
    """Returns true if *obj* can be hashed"""
    try:
        hash(obj)
    except TypeError:
        return False
    return True


def is_writable_file_like(obj):
    """return true if *obj* looks like a file object with a *write* method"""
    return hasattr(obj, 'write') and six.callable(obj.write)


def file_requires_unicode(x):
    """
    Returns `True` if the given writable file-like object requires Unicode
    to be written to it.
    """
    try:
        x.write(b'')
    except TypeError:
        return True
    else:
        return False


def is_scalar(obj):
    """return true if *obj* is not string like and is not iterable"""
    return not is_string_like(obj) and not iterable(obj)


def is_numlike(obj):
    """return true if *obj* looks like a number"""
    try:
        obj + 1
    except:
        return False
    else:
        return True


def to_filehandle(fname, flag='rU', return_opened=False):
    """
    *fname* can be a filename or a file handle.  Support for gzipped
    files is automatic, if the filename ends in .gz.  *flag* is a
    read/write flag for :func:`file`
    """
    if is_string_like(fname):
        if fname.endswith('.gz'):
            # get rid of 'U' in flag for gzipped files.
            flag = flag.replace('U', '')
            fh = gzip.open(fname, flag)
        elif fname.endswith('.bz2'):
            # get rid of 'U' in flag for bz2 files
            flag = flag.replace('U', '')
            import bz2
            fh = bz2.BZ2File(fname, flag)
        else:
            fh = open(fname, flag)
        opened = True
    elif hasattr(fname, 'seek'):
        fh = fname
        opened = False
    else:
        raise ValueError('fname must be a string or file handle')
    if return_opened:
        return fh, opened
    return fh


def is_scalar_or_string(val):
    """Return whether the given object is a scalar or string like."""
    return is_string_like(val) or not iterable(val)


def _string_to_bool(s):
    if not is_string_like(s):
        return s
    if s == 'on':
        return True
    if s == 'off':
        return False
    raise ValueError("string argument must be either 'on' or 'off'")


def get_sample_data(fname, asfileobj=True):
    """
    Return a sample data file.  *fname* is a path relative to the
    `mpl-data/sample_data` directory.  If *asfileobj* is `True`
    return a file object, otherwise just a file path.

    Set the rc parameter examples.directory to the directory where we should
    look, if sample_data files are stored in a location different than
    default (which is 'mpl-data/sample_data` at the same level of 'matplotlib`
    Python module files).

    If the filename ends in .gz, the file is implicitly ungzipped.
    """
    import matplotlib

    if matplotlib.rcParams['examples.directory']:
        root = matplotlib.rcParams['examples.directory']
    else:
        root = os.path.join(matplotlib._get_data_path(), 'sample_data')
    path = os.path.join(root, fname)

    if asfileobj:
        if (os.path.splitext(fname)[-1].lower() in
                ('.csv', '.xrc', '.txt')):
            mode = 'r'
        else:
            mode = 'rb'

        base, ext = os.path.splitext(fname)
        if ext == '.gz':
            return gzip.open(path, mode)
        else:
            return open(path, mode)
    else:
        return path


def flatten(seq, scalarp=is_scalar_or_string):
    """
    Returns a generator of flattened nested containers

    For example:

        >>> from matplotlib.cbook import flatten
        >>> l = (('John', ['Hunter']), (1, 23), [[([42, (5, 23)], )]])
        >>> print(list(flatten(l)))
        ['John', 'Hunter', 1, 23, 42, 5, 23]

    By: Composite of Holger Krekel and Luther Blissett
    From: https://code.activestate.com/recipes/121294/
    and Recipe 1.12 in cookbook
    """
    for item in seq:
        if scalarp(item):
            yield item
        else:
            for subitem in flatten(item, scalarp):
                yield subitem


class Sorter(object):
    """
    Sort by attribute or item

    Example usage::

      sort = Sorter()

      list = [(1, 2), (4, 8), (0, 3)]
      dict = [{'a': 3, 'b': 4}, {'a': 5, 'b': 2}, {'a': 0, 'b': 0},
              {'a': 9, 'b': 9}]


      sort(list)       # default sort
      sort(list, 1)    # sort by index 1
      sort(dict, 'a')  # sort a list of dicts by key 'a'

    """

    def _helper(self, data, aux, inplace):
        aux.sort()
        result = [data[i] for junk, i in aux]
        if inplace:
            data[:] = result
        return result

    def byItem(self, data, itemindex=None, inplace=1):
        if itemindex is None:
            if inplace:
                data.sort()
                result = data
            else:
                result = data[:]
                result.sort()
            return result
        else:
            aux = [(data[i][itemindex], i) for i in range(len(data))]
            return self._helper(data, aux, inplace)

    def byAttribute(self, data, attributename, inplace=1):
        aux = [(getattr(data[i], attributename), i) for i in range(len(data))]
        return self._helper(data, aux, inplace)

    # a couple of handy synonyms
    sort = byItem
    __call__ = byItem


class Xlator(dict):
    """
    All-in-one multiple-string-substitution class

    Example usage::

      text = "Larry Wall is the creator of Perl"
      adict = {
      "Larry Wall" : "Guido van Rossum",
      "creator" : "Benevolent Dictator for Life",
      "Perl" : "Python",
      }

      print(multiple_replace(adict, text))

      xlat = Xlator(adict)
      print(xlat.xlat(text))
    """

    def _make_regex(self):
        """ Build re object based on the keys of the current dictionary """
        return re.compile("|".join(map(re.escape, list(six.iterkeys(self)))))

    def __call__(self, match):
        """ Handler invoked for each regex *match* """
        return self[match.group(0)]

    def xlat(self, text):
        """ Translate *text*, returns the modified text. """
        return self._make_regex().sub(self, text)


def soundex(name, len=4):
    """ soundex module conforming to Odell-Russell algorithm """

    # digits holds the soundex values for the alphabet
    soundex_digits = '01230120022455012623010202'
    sndx = ''
    fc = ''

    # Translate letters in name to soundex digits
    for c in name.upper():
        if c.isalpha():
            if not fc:
                fc = c   # Remember first letter
            d = soundex_digits[ord(c) - ord('A')]
            # Duplicate consecutive soundex digits are skipped
            if not sndx or (d != sndx[-1]):
                sndx += d

    # Replace first digit with first letter
    sndx = fc + sndx[1:]

    # Remove all 0s from the soundex code
    sndx = sndx.replace('0', '')

    # Return soundex code truncated or 0-padded to len characters
    return (sndx + (len * '0'))[:len]


class Null(object):
    """ Null objects always and reliably "do nothing." """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __str__(self):
        return "Null()"

    def __repr__(self):
        return "Null()"

    if six.PY3:
        def __bool__(self):
            return 0
    else:
        def __nonzero__(self):
            return 0

    def __getattr__(self, name):
        return self

    def __setattr__(self, name, value):
        return self

    def __delattr__(self, name):
        return self


def mkdirs(newdir, mode=0o777):
    """
    make directory *newdir* recursively, and set *mode*.  Equivalent to ::

        > mkdir -p NEWDIR
        > chmod MODE NEWDIR
    """
    # this functionality is now in core python as of 3.2
    # LPY DROP
    if six.PY3:
        os.makedirs(newdir, mode=mode, exist_ok=True)
    else:
        try:
            os.makedirs(newdir, mode=mode)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

class GetRealpathAndStat(object):
    def __init__(self):
        self._cache = {}

    def __call__(self, path):
        result = self._cache.get(path)
        if result is None:
            realpath = os.path.realpath(path)
            if sys.platform == 'win32':
                stat_key = realpath
            else:
                stat = os.stat(realpath)
                stat_key = (stat.st_ino, stat.st_dev)
            result = realpath, stat_key
            self._cache[path] = result
        return result
get_realpath_and_stat = GetRealpathAndStat()


def dict_delall(d, keys):
    """delete all of the *keys* from the :class:`dict` *d*"""
    for key in keys:
        try:
            del d[key]
        except KeyError:
            pass


class RingBuffer(object):
    """ class that implements a not-yet-full buffer """
    def __init__(self, size_max):
        self.max = size_max
        self.data = []

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.cur] = x
            self.cur = (self.cur + 1) % self.max

        def get(self):
            """ return list of elements in correct order """
            return self.data[self.cur:] + self.data[:self.cur]

    def append(self, x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = __Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data

    def __get_item__(self, i):
        return self.data[i % len(self.data)]


def get_split_ind(seq, N):
    """
    *seq* is a list of words.  Return the index into seq such that::

        len(' '.join(seq[:ind])<=N

    .
    """

    s_len = 0
    # todo: use Alex's xrange pattern from the cbook for efficiency
    for (word, ind) in zip(seq, xrange(len(seq))):
        s_len += len(word) + 1  # +1 to account for the len(' ')
        if s_len >= N:
            return ind
    return len(seq)


def wrap(prefix, text, cols):
    """wrap *text* with *prefix* at length *cols*"""
    pad = ' ' * len(prefix.expandtabs())
    available = cols - len(pad)

    seq = text.split(' ')
    Nseq = len(seq)
    ind = 0
    lines = []
    while ind < Nseq:
        lastInd = ind
        ind += get_split_ind(seq[ind:], available)
        lines.append(seq[lastInd:ind])

    # add the prefix to the first line, pad with spaces otherwise
    ret = prefix + ' '.join(lines[0]) + '\n'
    for line in lines[1:]:
        ret += pad + ' '.join(line) + '\n'
    return ret

# A regular expression used to determine the amount of space to
# remove.  It looks for the first sequence of spaces immediately
# following the first newline, or at the beginning of the string.
_find_dedent_regex = re.compile("(?:(?:\n\r?)|^)( *)\S")
# A cache to hold the regexs that actually remove the indent.
_dedent_regex = {}


def dedent(s):
    """
    Remove excess indentation from docstring *s*.

    Discards any leading blank lines, then removes up to n whitespace
    characters from each line, where n is the number of leading
    whitespace characters in the first line. It differs from
    textwrap.dedent in its deletion of leading blank lines and its use
    of the first non-blank line to determine the indentation.

    It is also faster in most cases.
    """
    # This implementation has a somewhat obtuse use of regular
    # expressions.  However, this function accounted for almost 30% of
    # matplotlib startup time, so it is worthy of optimization at all
    # costs.

    if not s:      # includes case of s is None
        return ''

    match = _find_dedent_regex.match(s)
    if match is None:
        return s

    # This is the number of spaces to remove from the left-hand side.
    nshift = match.end(1) - match.start(1)
    if nshift == 0:
        return s

    # Get a regex that will remove *up to* nshift spaces from the
    # beginning of each line.  If it isn't in the cache, generate it.
    unindent = _dedent_regex.get(nshift, None)
    if unindent is None:
        unindent = re.compile("\n\r? {0,%d}" % nshift)
        _dedent_regex[nshift] = unindent

    result = unindent.sub("\n", s).strip()
    return result


def listFiles(root, patterns='*', recurse=1, return_folders=0):
    """
    Recursively list files

    from Parmar and Martelli in the Python Cookbook
    """
    import os.path
    import fnmatch
    # Expand patterns from semicolon-separated string to list
    pattern_list = patterns.split(';')
    results = []

    for dirname, dirs, files in os.walk(root):
        # Append to results all relevant files (and perhaps folders)
        for name in files:
            fullname = os.path.normpath(os.path.join(dirname, name))
            if return_folders or os.path.isfile(fullname):
                for pattern in pattern_list:
                    if fnmatch.fnmatch(name, pattern):
                        results.append(fullname)
                        break
        # Block recursion if recursion was disallowed
        if not recurse:
            break

    return results


def get_recursive_filelist(args):
    """
    Recurse all the files and dirs in *args* ignoring symbolic links
    and return the files as a list of strings
    """
    files = []

    for arg in args:
        if os.path.isfile(arg):
            files.append(arg)
            continue
        if os.path.isdir(arg):
            newfiles = listFiles(arg, recurse=1, return_folders=1)
            files.extend(newfiles)

    return [f for f in files if not os.path.islink(f)]


def pieces(seq, num=2):
    """Break up the *seq* into *num* tuples"""
    start = 0
    while 1:
        item = seq[start:start + num]
        if not len(item):
            break
        yield item
        start += num


def exception_to_str(s=None):
    if six.PY3:
        sh = io.StringIO()
    else:
        sh = io.BytesIO()
    if s is not None:
        print(s, file=sh)
    traceback.print_exc(file=sh)
    return sh.getvalue()


def allequal(seq):
    """
    Return *True* if all elements of *seq* compare equal.  If *seq* is
    0 or 1 length, return *True*
    """
    if len(seq) < 2:
        return True
    val = seq[0]
    for i in xrange(1, len(seq)):
        thisval = seq[i]
        if thisval != val:
            return False
    return True


def alltrue(seq):
    """
    Return *True* if all elements of *seq* evaluate to *True*.  If
    *seq* is empty, return *False*.
    """
    if not len(seq):
        return False
    for val in seq:
        if not val:
            return False
    return True


def onetrue(seq):
    """
    Return *True* if one element of *seq* is *True*.  It *seq* is
    empty, return *False*.
    """
    if not len(seq):
        return False
    for val in seq:
        if val:
            return True
    return False


def allpairs(x):
    """
    return all possible pairs in sequence *x*

    Condensed by Alex Martelli from this thread_ on c.l.python

    .. _thread: http://groups.google.com/groups?q=all+pairs+group:*python*&hl=en&lr=&ie=UTF-8&selm=mailman.4028.1096403649.5135.python-list%40python.org&rnum=1
    """
    return [(s, f) for i, f in enumerate(x) for s in x[i + 1:]]


class maxdict(dict):
    """
    A dictionary with a maximum size; this doesn't override all the
    relevant methods to constrain the size, just setitem, so use with
    caution
    """
    def __init__(self, maxsize):
        dict.__init__(self)
        self.maxsize = maxsize
        self._killkeys = []

    def __setitem__(self, k, v):
        if k not in self:
            if len(self) >= self.maxsize:
                del self[self._killkeys[0]]
                del self._killkeys[0]
            self._killkeys.append(k)
        dict.__setitem__(self, k, v)


class Stack(object):
    """
    Implement a stack where elements can be pushed on and you can move
    back and forth.  But no pop.  Should mimic home / back / forward
    in a browser
    """

    def __init__(self, default=None):
        self.clear()
        self._default = default

    def __call__(self):
        """return the current element, or None"""
        if not len(self._elements):
            return self._default
        else:
            return self._elements[self._pos]

    def __len__(self):
        return self._elements.__len__()

    def __getitem__(self, ind):
        return self._elements.__getitem__(ind)

    def forward(self):
        """move the position forward and return the current element"""
        n = len(self._elements)
        if self._pos < n - 1:
            self._pos += 1
        return self()

    def back(self):
        """move the position back and return the current element"""
        if self._pos > 0:
            self._pos -= 1
        return self()

    def push(self, o):
        """
        push object onto stack at current position - all elements
        occurring later than the current position are discarded
        """
        self._elements = self._elements[:self._pos + 1]
        self._elements.append(o)
        self._pos = len(self._elements) - 1
        return self()

    def home(self):
        """push the first element onto the top of the stack"""
        if not len(self._elements):
            return
        self.push(self._elements[0])
        return self()

    def empty(self):
        return len(self._elements) == 0

    def clear(self):
        """empty the stack"""
        self._pos = -1
        self._elements = []

    def bubble(self, o):
        """
        raise *o* to the top of the stack and return *o*.  *o* must be
        in the stack
        """

        if o not in self._elements:
            raise ValueError('Unknown element o')
        old = self._elements[:]
        self.clear()
        bubbles = []
        for thiso in old:
            if thiso == o:
                bubbles.append(thiso)
            else:
                self.push(thiso)
        for thiso in bubbles:
            self.push(o)
        return o

    def remove(self, o):
        'remove element *o* from the stack'
        if o not in self._elements:
            raise ValueError('Unknown element o')
        old = self._elements[:]
        self.clear()
        for thiso in old:
            if thiso == o:
                continue
            else:
                self.push(thiso)


def popall(seq):
    'empty a list'
    for i in xrange(len(seq)):
        seq.pop()


def finddir(o, match, case=False):
    """
    return all attributes of *o* which match string in match.  if case
    is True require an exact case match.
    """
    if case:
        names = [(name, name) for name in dir(o) if is_string_like(name)]
    else:
        names = [(name.lower(), name) for name in dir(o)
                 if is_string_like(name)]
        match = match.lower()
    return [orig for name, orig in names if name.find(match) >= 0]


def reverse_dict(d):
    """reverse the dictionary -- may lose data if values are not unique!"""
    return dict([(v, k) for k, v in six.iteritems(d)])


def restrict_dict(d, keys):
    """
    Return a dictionary that contains those keys that appear in both
    d and keys, with values from d.
    """
    return dict([(k, v) for (k, v) in six.iteritems(d) if k in keys])


def report_memory(i=0):  # argument may go away
    """return the memory consumed by process"""
    from matplotlib.compat.subprocess import Popen, PIPE
    pid = os.getpid()
    if sys.platform == 'sunos5':
        try:
            a2 = Popen(str('ps -p %d -o osz') % pid, shell=True,
                       stdout=PIPE).stdout.readlines()
        except OSError:
            raise NotImplementedError(
                "report_memory works on Sun OS only if "
                "the 'ps' program is found")
        mem = int(a2[-1].strip())
    elif sys.platform.startswith('linux'):
        try:
            a2 = Popen(str('ps -p %d -o rss,sz') % pid, shell=True,
                       stdout=PIPE).stdout.readlines()
        except OSError:
            raise NotImplementedError(
                "report_memory works on Linux only if "
                "the 'ps' program is found")
        mem = int(a2[1].split()[1])
    elif sys.platform.startswith('darwin'):
        try:
            a2 = Popen(str('ps -p %d -o rss,vsz') % pid, shell=True,
                       stdout=PIPE).stdout.readlines()
        except OSError:
            raise NotImplementedError(
                "report_memory works on Mac OS only if "
                "the 'ps' program is found")
        mem = int(a2[1].split()[0])
    elif sys.platform.startswith('win'):
        try:
            a2 = Popen([str("tasklist"), "/nh", "/fi", "pid eq %d" % pid],
                       stdout=PIPE).stdout.read()
        except OSError:
            raise NotImplementedError(
                "report_memory works on Windows only if "
                "the 'tasklist' program is found")
        mem = int(a2.strip().split()[-2].replace(',', ''))
    else:
        raise NotImplementedError(
            "We don't have a memory monitor for %s" % sys.platform)
    return mem

_safezip_msg = 'In safezip, len(args[0])=%d but len(args[%d])=%d'


def safezip(*args):
    """make sure *args* are equal len before zipping"""
    Nx = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != Nx:
            raise ValueError(_safezip_msg % (Nx, i + 1, len(arg)))
    return list(zip(*args))


def issubclass_safe(x, klass):
    """return issubclass(x, klass) and return False on a TypeError"""

    try:
        return issubclass(x, klass)
    except TypeError:
        return False


def safe_masked_invalid(x, copy=False):
    x = np.array(x, subok=True, copy=copy)
    if not x.dtype.isnative:
        # Note that the argument to `byteswap` is 'inplace',
        # thus if we have already made a copy, do the byteswap in
        # place, else make a copy with the byte order swapped.
        # Be explicit that we are swapping the byte order of the dtype
        x = x.byteswap(copy).newbyteorder('S')

    try:
        xm = np.ma.masked_invalid(x, copy=False)
        xm.shrink_mask()
    except TypeError:
        return x
    return xm


class MemoryMonitor(object):
    def __init__(self, nmax=20000):
        self._nmax = nmax
        self._mem = np.zeros((self._nmax,), np.int32)
        self.clear()

    def clear(self):
        self._n = 0
        self._overflow = False

    def __call__(self):
        mem = report_memory()
        if self._n < self._nmax:
            self._mem[self._n] = mem
            self._n += 1
        else:
            self._overflow = True
        return mem

    def report(self, segments=4):
        n = self._n
        segments = min(n, segments)
        dn = int(n / segments)
        ii = list(xrange(0, n, dn))
        ii[-1] = n - 1
        print()
        print('memory report: i, mem, dmem, dmem/nloops')
        print(0, self._mem[0])
        for i in range(1, len(ii)):
            di = ii[i] - ii[i - 1]
            if di == 0:
                continue
            dm = self._mem[ii[i]] - self._mem[ii[i - 1]]
            print('%5d %5d %3d %8.3f' % (ii[i], self._mem[ii[i]],
                                         dm, dm / float(di)))
        if self._overflow:
            print("Warning: array size was too small for the number of calls.")

    def xy(self, i0=0, isub=1):
        x = np.arange(i0, self._n, isub)
        return x, self._mem[i0:self._n:isub]

    def plot(self, i0=0, isub=1, fig=None):
        if fig is None:
            from .pylab import figure
            fig = figure()

        ax = fig.add_subplot(111)
        ax.plot(*self.xy(i0, isub))
        fig.canvas.draw()


def print_cycles(objects, outstream=sys.stdout, show_progress=False):
    """
    *objects*
        A list of objects to find cycles in.  It is often useful to
        pass in gc.garbage to find the cycles that are preventing some
        objects from being garbage collected.

    *outstream*
        The stream for output.

    *show_progress*
        If True, print the number of objects reached as they are found.
    """
    import gc
    from types import FrameType

    def print_path(path):
        for i, step in enumerate(path):
            # next "wraps around"
            next = path[(i + 1) % len(path)]

            outstream.write("   %s -- " % str(type(step)))
            if isinstance(step, dict):
                for key, val in six.iteritems(step):
                    if val is next:
                        outstream.write("[%s]" % repr(key))
                        break
                    if key is next:
                        outstream.write("[key] = %s" % repr(val))
                        break
            elif isinstance(step, list):
                outstream.write("[%d]" % step.index(next))
            elif isinstance(step, tuple):
                outstream.write("( tuple )")
            else:
                outstream.write(repr(step))
            outstream.write(" ->\n")
        outstream.write("\n")

    def recurse(obj, start, all, current_path):
        if show_progress:
            outstream.write("%d\r" % len(all))

        all[id(obj)] = None

        referents = gc.get_referents(obj)
        for referent in referents:
            # If we've found our way back to the start, this is
            # a cycle, so print it out
            if referent is start:
                print_path(current_path)

            # Don't go back through the original list of objects, or
            # through temporary references to the object, since those
            # are just an artifact of the cycle detector itself.
            elif referent is objects or isinstance(referent, FrameType):
                continue

            # We haven't seen this object before, so recurse
            elif id(referent) not in all:
                recurse(referent, start, all, current_path + [obj])

    for obj in objects:
        outstream.write("Examining: %r\n" % (obj,))
        recurse(obj, obj, {}, [])


class Grouper(object):
    """
    This class provides a lightweight way to group arbitrary objects
    together into disjoint sets when a full-blown graph data structure
    would be overkill.

    Objects can be joined using :meth:`join`, tested for connectedness
    using :meth:`joined`, and all disjoint sets can be retreived by
    using the object as an iterator.

    The objects being joined must be hashable and weak-referenceable.

    For example:

        >>> from matplotlib.cbook import Grouper
        >>> class Foo(object):
        ...     def __init__(self, s):
        ...         self.s = s
        ...     def __repr__(self):
        ...         return self.s
        ...
        >>> a, b, c, d, e, f = [Foo(x) for x in 'abcdef']
        >>> grp = Grouper()
        >>> grp.join(a, b)
        >>> grp.join(b, c)
        >>> grp.join(d, e)
        >>> sorted(map(tuple, grp))
        [(a, b, c), (d, e)]
        >>> grp.joined(a, b)
        True
        >>> grp.joined(a, c)
        True
        >>> grp.joined(a, d)
        False

    """
    def __init__(self, init=()):
        mapping = self._mapping = {}
        for x in init:
            mapping[ref(x)] = [ref(x)]

    def __contains__(self, item):
        return ref(item) in self._mapping

    def clean(self):
        """
        Clean dead weak references from the dictionary
        """
        mapping = self._mapping
        to_drop = [key for key in mapping if key() is None]
        for key in to_drop:
            val = mapping.pop(key)
            val.remove(key)

    def join(self, a, *args):
        """
        Join given arguments into the same set.  Accepts one or more
        arguments.
        """
        mapping = self._mapping
        set_a = mapping.setdefault(ref(a), [ref(a)])

        for arg in args:
            set_b = mapping.get(ref(arg))
            if set_b is None:
                set_a.append(ref(arg))
                mapping[ref(arg)] = set_a
            elif set_b is not set_a:
                if len(set_b) > len(set_a):
                    set_a, set_b = set_b, set_a
                set_a.extend(set_b)
                for elem in set_b:
                    mapping[elem] = set_a

        self.clean()

    def joined(self, a, b):
        """
        Returns True if *a* and *b* are members of the same set.
        """
        self.clean()

        mapping = self._mapping
        try:
            return mapping[ref(a)] is mapping[ref(b)]
        except KeyError:
            return False

    def remove(self, a):
        self.clean()

        mapping = self._mapping
        seta = mapping.pop(ref(a), None)
        if seta is not None:
            seta.remove(ref(a))

    def __iter__(self):
        """
        Iterate over each of the disjoint sets as a list.

        The iterator is invalid if interleaved with calls to join().
        """
        self.clean()

        class Token:
            pass
        token = Token()

        # Mark each group as we come across if by appending a token,
        # and don't yield it twice
        for group in six.itervalues(self._mapping):
            if not group[-1] is token:
                yield [x() for x in group]
                group.append(token)

        # Cleanup the tokens
        for group in six.itervalues(self._mapping):
            if group[-1] is token:
                del group[-1]

    def get_siblings(self, a):
        """
        Returns all of the items joined with *a*, including itself.
        """
        self.clean()

        siblings = self._mapping.get(ref(a), [ref(a)])
        return [x() for x in siblings]


def simple_linear_interpolation(a, steps):
    if steps == 1:
        return a

    steps = int(np.floor(steps))
    new_length = ((len(a) - 1) * steps) + 1
    new_shape = list(a.shape)
    new_shape[0] = new_length
    result = np.zeros(new_shape, a.dtype)

    result[0] = a[0]
    a0 = a[0:-1]
    a1 = a[1:]
    delta = ((a1 - a0) / steps)
    for i in range(1, steps):
        result[i::steps] = delta * i + a0
    result[steps::steps] = a1

    return result


def recursive_remove(path):
    if os.path.isdir(path):
        for fname in (glob.glob(os.path.join(path, '*')) +
                      glob.glob(os.path.join(path, '.*'))):
            if os.path.isdir(fname):
                recursive_remove(fname)
                os.removedirs(fname)
            else:
                os.remove(fname)
        #os.removedirs(path)
    else:
        os.remove(path)


def delete_masked_points(*args):
    """
    Find all masked and/or non-finite points in a set of arguments,
    and return the arguments with only the unmasked points remaining.

    Arguments can be in any of 5 categories:

    1) 1-D masked arrays
    2) 1-D ndarrays
    3) ndarrays with more than one dimension
    4) other non-string iterables
    5) anything else

    The first argument must be in one of the first four categories;
    any argument with a length differing from that of the first
    argument (and hence anything in category 5) then will be
    passed through unchanged.

    Masks are obtained from all arguments of the correct length
    in categories 1, 2, and 4; a point is bad if masked in a masked
    array or if it is a nan or inf.  No attempt is made to
    extract a mask from categories 2, 3, and 4 if :meth:`np.isfinite`
    does not yield a Boolean array.

    All input arguments that are not passed unchanged are returned
    as ndarrays after removing the points or rows corresponding to
    masks in any of the arguments.

    A vastly simpler version of this function was originally
    written as a helper for Axes.scatter().

    """
    if not len(args):
        return ()
    if (is_string_like(args[0]) or not iterable(args[0])):
        raise ValueError("First argument must be a sequence")
    nrecs = len(args[0])
    margs = []
    seqlist = [False] * len(args)
    for i, x in enumerate(args):
        if (not is_string_like(x)) and iterable(x) and len(x) == nrecs:
            seqlist[i] = True
            if ma.isMA(x):
                if x.ndim > 1:
                    raise ValueError("Masked arrays must be 1-D")
            else:
                x = np.asarray(x)
        margs.append(x)
    masks = []    # list of masks that are True where good
    for i, x in enumerate(margs):
        if seqlist[i]:
            if x.ndim > 1:
                continue  # Don't try to get nan locations unless 1-D.
            if ma.isMA(x):
                masks.append(~ma.getmaskarray(x))  # invert the mask
                xd = x.data
            else:
                xd = x
            try:
                mask = np.isfinite(xd)
                if isinstance(mask, np.ndarray):
                    masks.append(mask)
            except:  # Fixme: put in tuple of possible exceptions?
                pass
    if len(masks):
        mask = reduce(np.logical_and, masks)
        igood = mask.nonzero()[0]
        if len(igood) < nrecs:
            for i, x in enumerate(margs):
                if seqlist[i]:
                    margs[i] = x.take(igood, axis=0)
    for i, x in enumerate(margs):
        if seqlist[i] and ma.isMA(x):
            margs[i] = x.filled()
    return margs


def boxplot_stats(X, whis=1.5, bootstrap=None, labels=None,
                  autorange=False):
    """
    Returns list of dictionaries of statistics used to draw a series
    of box and whisker plots. The `Returns` section enumerates the
    required keys of the dictionary. Users can skip this function and
    pass a user-defined set of dictionaries to the new `axes.bxp` method
    instead of relying on MPL to do the calculations.

    Parameters
    ----------
    X : array-like
        Data that will be represented in the boxplots. Should have 2 or
        fewer dimensions.

    whis : float, string, or sequence (default = 1.5)
        As a float, determines the reach of the whiskers past the first
        and third quartiles (e.g., Q3 + whis*IQR, QR = interquartile
        range, Q3-Q1). Beyond the whiskers, data are considered outliers
        and are plotted as individual points. This can be set this to an
        ascending sequence of percentile (e.g., [5, 95]) to set the
        whiskers at specific percentiles of the data. Finally, `whis`
        can be the string ``'range'`` to force the whiskers to the
        minimum and maximum of the data. In the edge case that the 25th
        and 75th percentiles are equivalent, `whis` can be automatically
        set to ``'range'`` via the `autorange` option.

    bootstrap : int, optional
        Number of times the confidence intervals around the median
        should be bootstrapped (percentile method).

    labels : array-like, optional
        Labels for each dataset. Length must be compatible with
        dimensions of `X`.

    autorange : bool, optional (False)
        When `True` and the data are distributed such that the  25th and
        75th percentiles are equal, ``whis`` is set to ``'range'`` such
        that the whisker ends are at the minimum and maximum of the
        data.

    Returns
    -------
    bxpstats : list of dict
        A list of dictionaries containing the results for each column
        of data. Keys of each dictionary are the following:

        ========   ===================================
        Key        Value Description
        ========   ===================================
        label      tick label for the boxplot
        mean       arithemetic mean value
        med        50th percentile
        q1         first quartile (25th percentile)
        q3         third quartile (75th percentile)
        cilo       lower notch around the median
        cihi       upper notch around the median
        whislo     end of the lower whisker
        whishi     end of the upper whisker
        fliers     outliers
        ========   ===================================

    Notes
    -----
    Non-bootstrapping approach to confidence interval uses Gaussian-
    based asymptotic approximation:

    .. math::

        \mathrm{med} \pm 1.57 \\times \\frac{\mathrm{iqr}}{\sqrt{N}}

    General approach from:
    McGill, R., Tukey, J.W., and Larsen, W.A. (1978) "Variations of
    Boxplots", The American Statistician, 32:12-16.

    """

    def _bootstrap_median(data, N=5000):
        # determine 95% confidence intervals of the median
        M = len(data)
        percentiles = [2.5, 97.5]

        ii = np.random.randint(M, size=(N, M))
        bsData = x[ii]
        estimate = np.median(bsData, axis=1, overwrite_input=True)

        CI = np.percentile(estimate, percentiles)
        return CI

    def _compute_conf_interval(data, med, iqr, bootstrap):
        if bootstrap is not None:
            # Do a bootstrap estimate of notch locations.
            # get conf. intervals around median
            CI = _bootstrap_median(data, N=bootstrap)
            notch_min = CI[0]
            notch_max = CI[1]
        else:

            N = len(data)
            notch_min = med - 1.57 * iqr / np.sqrt(N)
            notch_max = med + 1.57 * iqr / np.sqrt(N)

        return notch_min, notch_max

    # output is a list of dicts
    bxpstats = []

    # convert X to a list of lists
    X = _reshape_2D(X)

    ncols = len(X)
    if labels is None:
        labels = repeat(None)
    elif len(labels) != ncols:
        raise ValueError("Dimensions of labels and X must be compatible")

    input_whis = whis
    for ii, (x, label) in enumerate(zip(X, labels), start=0):

        # empty dict
        stats = {}
        if label is not None:
            stats['label'] = label

        # restore whis to the input values in case it got changed in the loop
        whis = input_whis

        # note tricksyness, append up here and then mutate below
        bxpstats.append(stats)

        # if empty, bail
        if len(x) == 0:
            stats['fliers'] = np.array([])
            stats['mean'] = np.nan
            stats['med'] = np.nan
            stats['q1'] = np.nan
            stats['q3'] = np.nan
            stats['cilo'] = np.nan
            stats['cihi'] = np.nan
            stats['whislo'] = np.nan
            stats['whishi'] = np.nan
            stats['med'] = np.nan
            continue

        # up-convert to an array, just to be safe
        x = np.asarray(x)

        # arithmetic mean
        stats['mean'] = np.mean(x)

        # medians and quartiles
        q1, med, q3 = np.percentile(x, [25, 50, 75])

        # interquartile range
        stats['iqr'] = q3 - q1
        if stats['iqr'] == 0 and autorange:
            whis = 'range'

        # conf. interval around median
        stats['cilo'], stats['cihi'] = _compute_conf_interval(
            x, med, stats['iqr'], bootstrap
        )

        # lowest/highest non-outliers
        if np.isscalar(whis):
            if np.isreal(whis):
                loval = q1 - whis * stats['iqr']
                hival = q3 + whis * stats['iqr']
            elif whis in ['range', 'limit', 'limits', 'min/max']:
                loval = np.min(x)
                hival = np.max(x)
            else:
                whismsg = ('whis must be a float, valid string, or '
                           'list of percentiles')
                raise ValueError(whismsg)
        else:
            loval = np.percentile(x, whis[0])
            hival = np.percentile(x, whis[1])

        # get high extreme
        wiskhi = np.compress(x <= hival, x)
        if len(wiskhi) == 0 or np.max(wiskhi) < q3:
            stats['whishi'] = q3
        else:
            stats['whishi'] = np.max(wiskhi)

        # get low extreme
        wisklo = np.compress(x >= loval, x)
        if len(wisklo) == 0 or np.min(wisklo) > q1:
            stats['whislo'] = q1
        else:
            stats['whislo'] = np.min(wisklo)

        # compute a single array of outliers
        stats['fliers'] = np.hstack([
            np.compress(x < stats['whislo'], x),
            np.compress(x > stats['whishi'], x)
        ])

        # add in the remaining stats
        stats['q1'], stats['med'], stats['q3'] = q1, med, q3


    return bxpstats


# FIXME I don't think this is used anywhere
def unmasked_index_ranges(mask, compressed=True):
    """
    Find index ranges where *mask* is *False*.

    *mask* will be flattened if it is not already 1-D.

    Returns Nx2 :class:`numpy.ndarray` with each row the start and stop
    indices for slices of the compressed :class:`numpy.ndarray`
    corresponding to each of *N* uninterrupted runs of unmasked
    values.  If optional argument *compressed* is *False*, it returns
    the start and stop indices into the original :class:`numpy.ndarray`,
    not the compressed :class:`numpy.ndarray`.  Returns *None* if there
    are no unmasked values.

    Example::

      y = ma.array(np.arange(5), mask = [0,0,1,0,0])
      ii = unmasked_index_ranges(ma.getmaskarray(y))
      # returns array [[0,2,] [2,4,]]

      y.compressed()[ii[1,0]:ii[1,1]]
      # returns array [3,4,]

      ii = unmasked_index_ranges(ma.getmaskarray(y), compressed=False)
      # returns array [[0, 2], [3, 5]]

      y.filled()[ii[1,0]:ii[1,1]]
      # returns array [3,4,]

    Prior to the transforms refactoring, this was used to support
    masked arrays in Line2D.
    """
    mask = mask.reshape(mask.size)
    m = np.concatenate(((1,), mask, (1,)))
    indices = np.arange(len(mask) + 1)
    mdif = m[1:] - m[:-1]
    i0 = np.compress(mdif == -1, indices)
    i1 = np.compress(mdif == 1, indices)
    assert len(i0) == len(i1)
    if len(i1) == 0:
        return None  # Maybe this should be np.zeros((0,2), dtype=int)
    if not compressed:
        return np.concatenate((i0[:, np.newaxis], i1[:, np.newaxis]), axis=1)
    seglengths = i1 - i0
    breakpoints = np.cumsum(seglengths)
    ic0 = np.concatenate(((0,), breakpoints[:-1]))
    ic1 = breakpoints
    return np.concatenate((ic0[:, np.newaxis], ic1[:, np.newaxis]), axis=1)

# a dict to cross-map linestyle arguments
_linestyles = [('-', 'solid'),
               ('--', 'dashed'),
               ('-.', 'dashdot'),
               (':', 'dotted')]

ls_mapper = dict(_linestyles)
# The ls_mapper maps short codes for line style to their full name used
# by backends
# The reverse mapper is for mapping full names to short ones
ls_mapper_r = dict([(ls[1], ls[0]) for ls in _linestyles])


def align_iterators(func, *iterables):
    """
    This generator takes a bunch of iterables that are ordered by func
    It sends out ordered tuples::

       (func(row), [rows from all iterators matching func(row)])

    It is used by :func:`matplotlib.mlab.recs_join` to join record arrays
    """
    class myiter:
        def __init__(self, it):
            self.it = it
            self.key = self.value = None
            self.iternext()

        def iternext(self):
            try:
                self.value = next(self.it)
                self.key = func(self.value)
            except StopIteration:
                self.value = self.key = None

        def __call__(self, key):
            retval = None
            if key == self.key:
                retval = self.value
                self.iternext()
            elif self.key and key > self.key:
                raise ValueError("Iterator has been left behind")
            return retval

    # This can be made more efficient by not computing the minimum key for each
    # iteration
    iters = [myiter(it) for it in iterables]
    minvals = minkey = True
    while True:
        minvals = ([_f for _f in [it.key for it in iters] if _f])
        if minvals:
            minkey = min(minvals)
            yield (minkey, [it(minkey) for it in iters])
        else:
            break


def is_math_text(s):
    # Did we find an even number of non-escaped dollar signs?
    # If so, treat is as math text.
    try:
        s = six.text_type(s)
    except UnicodeDecodeError:
        raise ValueError(
            "matplotlib display text must have all code points < 128 or use "
            "Unicode strings")

    dollar_count = s.count(r'$') - s.count(r'\$')
    even_dollars = (dollar_count > 0 and dollar_count % 2 == 0)

    return even_dollars


def _check_1d(x):
    '''
    Converts a sequence of less than 1 dimension, to an array of 1
    dimension; leaves everything else untouched.
    '''
    if not hasattr(x, 'shape') or len(x.shape) < 1:
        return np.atleast_1d(x)
    else:
        try:
            x[:, None]
            return x
        except (IndexError, TypeError):
            return np.atleast_1d(x)


def _reshape_2D(X):
    """
    Converts a non-empty list or an ndarray of two or fewer dimensions
    into a list of iterable objects so that in

        for v in _reshape_2D(X):

    v is iterable and can be used to instantiate a 1D array.
    """
    if hasattr(X, 'shape'):
        # one item
        if len(X.shape) == 1:
            if hasattr(X[0], 'shape'):
                X = list(X)
            else:
                X = [X, ]

        # several items
        elif len(X.shape) == 2:
            nrows, ncols = X.shape
            if nrows == 1:
                X = [X]
            elif ncols == 1:
                X = [X.ravel()]
            else:
                X = [X[:, i] for i in xrange(ncols)]
        else:
            raise ValueError("input `X` must have 2 or fewer dimensions")

    if not hasattr(X[0], '__len__'):
        X = [X]
    else:
        X = [np.ravel(x) for x in X]

    return X


def violin_stats(X, method, points=100):
    """
    Returns a list of dictionaries of data which can be used to draw a series
    of violin plots. See the `Returns` section below to view the required keys
    of the dictionary. Users can skip this function and pass a user-defined set
    of dictionaries to the `axes.vplot` method instead of using MPL to do the
    calculations.

    Parameters
    ----------
    X : array-like
        Sample data that will be used to produce the gaussian kernel density
        estimates. Must have 2 or fewer dimensions.

    method : callable
        The method used to calculate the kernel density estimate for each
        column of data. When called via `method(v, coords)`, it should
        return a vector of the values of the KDE evaluated at the values
        specified in coords.

    points : scalar, default = 100
        Defines the number of points to evaluate each of the gaussian kernel
        density estimates at.

    Returns
    -------

    A list of dictionaries containing the results for each column of data.
    The dictionaries contain at least the following:

        - coords: A list of scalars containing the coordinates this particular
          kernel density estimate was evaluated at.
        - vals: A list of scalars containing the values of the kernel density
          estimate at each of the coordinates given in `coords`.
        - mean: The mean value for this column of data.
        - median: The median value for this column of data.
        - min: The minimum value for this column of data.
        - max: The maximum value for this column of data.
    """

    # List of dictionaries describing each of the violins.
    vpstats = []

    # Want X to be a list of data sequences
    X = _reshape_2D(X)

    for x in X:
        # Dictionary of results for this distribution
        stats = {}

        # Calculate basic stats for the distribution
        min_val = np.min(x)
        max_val = np.max(x)

        # Evaluate the kernel density estimate
        coords = np.linspace(min_val, max_val, points)
        stats['vals'] = method(x, coords)
        stats['coords'] = coords

        # Store additional statistics for this distribution
        stats['mean'] = np.mean(x)
        stats['median'] = np.median(x)
        stats['min'] = min_val
        stats['max'] = max_val

        # Append to output
        vpstats.append(stats)

    return vpstats


class _NestedClassGetter(object):
    # recipe from http://stackoverflow.com/a/11493777/741316
    """
    When called with the containing class as the first argument,
    and the name of the nested class as the second argument,
    returns an instance of the nested class.
    """
    def __call__(self, containing_class, class_name):
        nested_class = getattr(containing_class, class_name)

        # make an instance of a simple object (this one will do), for which we
        # can change the __class__ later on.
        nested_instance = _NestedClassGetter()

        # set the class of the instance, the __init__ will never be called on
        # the class but the original state will be set later on by pickle.
        nested_instance.__class__ = nested_class
        return nested_instance


class _InstanceMethodPickler(object):
    """
    Pickle cannot handle instancemethod saving. _InstanceMethodPickler
    provides a solution to this.
    """
    def __init__(self, instancemethod):
        """Takes an instancemethod as its only argument."""
        if six.PY3:
            self.parent_obj = instancemethod.__self__
            self.instancemethod_name = instancemethod.__func__.__name__
        else:
            self.parent_obj = instancemethod.im_self
            self.instancemethod_name = instancemethod.im_func.__name__

    def get_instancemethod(self):
        return getattr(self.parent_obj, self.instancemethod_name)


def _step_validation(x, *args):
    """
    Helper function of `pts_to_*step` functions

    This function does all of the normalization required to the
    input and generate the template for output


    """
    args = tuple(np.asanyarray(y) for y in args)
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1 dimensional")
    if len(args) == 0:
        raise ValueError("At least one Y value must be passed")

    return np.vstack((x, ) + args)


def pts_to_prestep(x, *args):
    """
    Covert continuous line to pre-steps

    Given a set of N points convert to 2 N -1 points
    which when connected linearly give a step function
    which changes values at the begining the intervals.

    Parameters
    ----------
    x : array
        The x location of the steps

    y1, y2, ... : array
        Any number of y arrays to be turned into steps.
        All must be the same length as ``x``

    Returns
    -------
    x, y1, y2, .. : array
        The x and y values converted to steps in the same order
        as the input.  If the input is length ``N``, each of these arrays
        will be length ``2N + 1``


    Examples
    --------
    >> x_s, y1_s, y2_s = pts_to_prestep(x, y1, y2)
    """
    # do normalization
    vertices = _step_validation(x, *args)
    # create the output array
    steps = np.zeros((vertices.shape[0], 2 * len(x) - 1), np.float)
    # do the to step conversion logic
    steps[0, 0::2], steps[0, 1::2] = vertices[0, :], vertices[0, :-1]
    steps[1:, 0::2], steps[1:, 1:-1:2] = vertices[1:, :], vertices[1:, 1:]
    # convert 2D array back to tuple
    return tuple(steps)


def pts_to_poststep(x, *args):
    """
    Covert continuous line to pre-steps

    Given a set of N points convert to 2 N -1 points
    which when connected linearly give a step function
    which changes values at the begining the intervals.

    Parameters
    ----------
    x : array
        The x location of the steps

    y1, y2, ... : array
        Any number of y arrays to be turned into steps.
        All must be the same length as ``x``

    Returns
    -------
    x, y1, y2, .. : array
        The x and y values converted to steps in the same order
        as the input.  If the input is length ``N``, each of these arrays
        will be length ``2N + 1``


    Examples
    --------
    >> x_s, y1_s, y2_s = pts_to_prestep(x, y1, y2)
    """
    # do normalization
    vertices = _step_validation(x, *args)
    # create the output array
    steps = ma.zeros((vertices.shape[0], 2 * len(x) - 1), np.float)
    # do the to step conversion logic
    steps[0, ::2], steps[0, 1:-1:2] = vertices[0, :], vertices[0, 1:]
    steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :-1]

    # convert 2D array back to tuple
    return tuple(steps)


def pts_to_midstep(x, *args):
    """
    Covert continuous line to pre-steps

    Given a set of N points convert to 2 N -1 points
    which when connected linearly give a step function
    which changes values at the begining the intervals.

    Parameters
    ----------
    x : array
        The x location of the steps

    y1, y2, ... : array
        Any number of y arrays to be turned into steps.
        All must be the same length as ``x``

    Returns
    -------
    x, y1, y2, .. : array
        The x and y values converted to steps in the same order
        as the input.  If the input is length ``N``, each of these arrays
        will be length ``2N + 1``


    Examples
    --------
    >> x_s, y1_s, y2_s = pts_to_prestep(x, y1, y2)
    """
    # do normalization
    vertices = _step_validation(x, *args)
    # create the output array
    steps = ma.zeros((vertices.shape[0], 2 * len(x)), np.float)
    steps[0, 1:-1:2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
    steps[0, 2::2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
    steps[0, 0] = vertices[0, 0]
    steps[0, -1] = vertices[0, -1]
    steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :]

    # convert 2D array back to tuple
    return tuple(steps)

STEP_LOOKUP_MAP = {'pre': pts_to_prestep,
                   'post': pts_to_poststep,
                   'mid': pts_to_midstep,
                   'step-pre': pts_to_prestep,
                   'step-post': pts_to_poststep,
                   'step-mid': pts_to_midstep}


def index_of(y):
    """
    A helper function to get the index of an input to plot
    against if x values are not explicitly given.

    Tries to get `y.index` (works if this is a pd.Series), if that
    fails, return np.arange(y.shape[0]).

    This will be extended in the future to deal with more types of
    labeled data.

    Parameters
    ----------
    y : scalar or array-like
        The proposed y-value

    Returns
    -------
    x, y : ndarray
       The x and y values to plot.
    """
    try:
        return y.index.values, y.values
    except AttributeError:
        y = np.atleast_1d(y)
        return np.arange(y.shape[0], dtype=float), y


def safe_first_element(obj):
    if isinstance(obj, collections.Iterator):
        # needed to accept `array.flat` as input.
        # np.flatiter reports as an instance of collections.Iterator
        # but can still be indexed via [].
        # This has the side effect of re-setting the iterator, but
        # that is acceptable.
        try:
            return obj[0]
        except TypeError:
            pass
        raise RuntimeError("matplotlib does not support generators "
                           "as input")
    return next(iter(obj))


def normalize_kwargs(kw, alias_mapping=None, required=(), forbidden=(),
                     allowed=None):
    """Helper function to normalize kwarg inputs

    The order they are resolved are:

     1. aliasing
     2. required
     3. forbidden
     4. allowed

    This order means that only the canonical names need appear in
    `allowed`, `forbidden`, `required`

    Parameters
    ----------

    alias_mapping, dict, optional
        A mapping between a canonical name to a list of
        aliases, in order of precedence from lowest to highest.

        If the canonical value is not in the list it is assumed to have
        the highest priority.

    required : iterable, optional
        A tuple of fields that must be in kwargs.

    forbidden : iterable, optional
        A list of keys which may not be in kwargs

    allowed : tuple, optional
        A tuple of allowed fields.  If this not None, then raise if
        `kw` contains any keys not in the union of `required`
        and `allowed`.  To allow only the required fields pass in
        ``()`` for `allowed`

    Raises
    ------
    TypeError
        To match what python raises if invalid args/kwargs are passed to
        a callable.

    """
    # deal with default value of alias_mapping
    if alias_mapping is None:
        alias_mapping = dict()

    # make a local so we can pop
    kw = dict(kw)
    # output dictionary
    ret = dict()

    # hit all alias mappings
    for canonical, alias_list in six.iteritems(alias_mapping):

        # the alias lists are ordered from lowest to highest priority
        # so we know to use the last value in this list
        tmp = []
        seen = []
        for a in alias_list:
            try:
                tmp.append(kw.pop(a))
                seen.append(a)
            except KeyError:
                pass
        # if canonical is not in the alias_list assume highest priority
        if canonical not in alias_list:
            try:
                tmp.append(kw.pop(canonical))
                seen.append(canonical)
            except KeyError:
                pass
        # if we found anything in this set of aliases put it in the return
        # dict
        if tmp:
            ret[canonical] = tmp[-1]
            if len(tmp) > 1:
                warnings.warn("Saw kwargs {seen!r} which are all aliases for "
                              "{canon!r}.  Kept value from {used!r}".format(
                                  seen=seen, canon=canonical, used=seen[-1]))

    # at this point we know that all keys which are aliased are removed, update
    # the return dictionary from the cleaned local copy of the input
    ret.update(kw)

    fail_keys = [k for k in required if k not in ret]
    if fail_keys:
        raise TypeError("The required keys {keys!r} "
                        "are not in kwargs".format(keys=fail_keys))

    fail_keys = [k for k in forbidden if k in ret]
    if fail_keys:
        raise TypeError("The forbidden keys {keys!r} "
                        "are in kwargs".format(keys=fail_keys))

    if allowed is not None:
        allowed_set = set(required) | set(allowed)
        fail_keys = [k for k in ret if k not in allowed_set]
        if fail_keys:
            raise TypeError("kwargs contains {keys!r} which are not in "
                            "the required {req!r} or "
                            "allowed {allow!r} keys".format(
                                keys=fail_keys, req=required,
                                allow=allowed))

    return ret


def get_label(y, default_name):
    try:
        return y.name
    except AttributeError:
        return default_name

# Numpy > 1.6.x deprecates putmask in favor of the new copyto.
# So long as we support versions 1.6.x and less, we need the
# following local version of putmask.  We choose to make a
# local version of putmask rather than of copyto because the
# latter includes more functionality than the former. Therefore
# it is easy to make a local version that gives full putmask
# behavior, but duplicating the full copyto behavior would be
# more difficult.

try:
    np.copyto
except AttributeError:
    _putmask = np.putmask
else:
    def _putmask(a, mask, values):
        return np.copyto(a, values, where=mask)

_lockstr = """\
LOCKERROR: matplotlib is trying to acquire the lock
    {!r}
and has failed.  This maybe due to any other process holding this
lock.  If you are sure no other matplotlib process is running try
removing these folders and trying again.
"""


class Locked(object):
    """
    Context manager to handle locks.

    Based on code from conda.

    (c) 2012-2013 Continuum Analytics, Inc. / https://www.continuum.io/
    All Rights Reserved

    conda is distributed under the terms of the BSD 3-clause license.
    Consult LICENSE_CONDA or https://opensource.org/licenses/BSD-3-Clause.
    """
    LOCKFN = '.matplotlib_lock'

    class TimeoutError(RuntimeError):
        pass

    def __init__(self, path):
        self.path = path
        self.end = "-" + str(os.getpid())
        self.lock_path = os.path.join(self.path, self.LOCKFN + self.end)
        self.pattern = os.path.join(self.path, self.LOCKFN + '-*')
        self.remove = True

    def __enter__(self):
        retries = 50
        sleeptime = 0.1
        while retries:
            files = glob.glob(self.pattern)
            if files and not files[0].endswith(self.end):
                time.sleep(sleeptime)
                retries -= 1
            else:
                break
        else:
            err_str = _lockstr.format(self.pattern)
            raise self.TimeoutError(err_str)

        if not files:
            try:
                os.makedirs(self.lock_path)
            except OSError:
                pass
        else:  # PID lock already here --- someone else will remove it.
            self.remove = False

    def __exit__(self, exc_type, exc_value, traceback):
        if self.remove:
            for path in self.lock_path, self.path:
                try:
                    os.rmdir(path)
                except OSError:
                    pass
