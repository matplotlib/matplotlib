"""
A collection of utility functions and classes.  Originally, many
(but not all) were from the Python Cookbook -- hence the name cbook.

This module is safe to import from anywhere within matplotlib;
it imports matplotlib only at runtime.
"""

import collections
import collections.abc
import contextlib
import datetime
import errno
import functools
import glob
import gzip
import io
import itertools
import locale
import numbers
import operator
import os
from pathlib import Path
import re
import sys
import time
import traceback
import types
import warnings
import weakref
from weakref import WeakMethod

import numpy as np

import matplotlib
from .deprecation import (
    mplDeprecation, deprecated, warn_deprecated, MatplotlibDeprecationWarning)


@deprecated("3.0")
def unicode_safe(s):

    if isinstance(s, bytes):
        try:
            # On some systems, locale.getpreferredencoding returns None,
            # which can break unicode; and the sage project reports that
            # some systems have incorrect locale specifications, e.g.,
            # an encoding instead of a valid locale name.  Another
            # pathological case that has been reported is an empty string.
            # On some systems, getpreferredencoding sets the locale, which has
            # side effects.  Passing False eliminates those side effects.
            preferredencoding = locale.getpreferredencoding(
                matplotlib.rcParams['axes.formatter.use_locale']).strip()
            if not preferredencoding:
                preferredencoding = None
        except (ValueError, ImportError, AttributeError):
            preferredencoding = None

        if preferredencoding is None:
            return str(s)
        else:
            return str(s, preferredencoding)
    return s


def _exception_printer(exc):
    traceback.print_exc()


class _StrongRef:
    """
    Wrapper similar to a weakref, but keeping a strong reference to the object.
    """

    def __init__(self, obj):
        self._obj = obj

    def __call__(self):
        return self._obj

    def __eq__(self, other):
        return isinstance(other, _StrongRef) and self._obj == other._obj

    def __hash__(self):
        return hash(self._obj)


class CallbackRegistry(object):
    """Handle registering and disconnecting for a set of signals and callbacks:

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

    In practice, one should always disconnect all callbacks when they are
    no longer needed to avoid dangling references (and thus memory leaks).
    However, real code in Matplotlib rarely does so, and due to its design,
    it is rather difficult to place this kind of code.  To get around this,
    and prevent this class of memory leaks, we instead store weak references
    to bound methods only, so when the destination object needs to die, the
    CallbackRegistry won't keep it alive.

    Parameters
    ----------
    exception_handler : callable, optional
       If provided must have signature ::

          def handler(exc: Exception) -> None:

       If not None this function will be called with any `Exception`
       subclass raised by the callbacks in `CallbackRegistry.process`.
       The handler may either consume the exception or re-raise.

       The callable must be pickle-able.

       The default handler is ::

          def h(exc):
              traceback.print_exc()
    """

    # We maintain two mappings:
    #   callbacks: signal -> {cid -> callback}
    #   _func_cid_map: signal -> {callback -> cid}
    # (actually, callbacks are weakrefs to the actual callbacks).

    def __init__(self, exception_handler=_exception_printer):
        self.exception_handler = exception_handler
        self.callbacks = {}
        self._cid_gen = itertools.count()
        self._func_cid_map = {}

    # In general, callbacks may not be pickled; thus, we simply recreate an
    # empty dictionary at unpickling.  In order to ensure that `__setstate__`
    # (which just defers to `__init__`) is called, `__getstate__` must
    # return a truthy value (for pickle protocol>=3, i.e. Py3, the
    # *actual* behavior is that `__setstate__` will be called as long as
    # `__getstate__` does not return `None`, but this is undocumented -- see
    # http://bugs.python.org/issue12290).

    def __getstate__(self):
        return {'exception_handler': self.exception_handler}

    def __setstate__(self, state):
        self.__init__(**state)

    def connect(self, s, func):
        """Register *func* to be called when signal *s* is generated.
        """
        self._func_cid_map.setdefault(s, {})
        try:
            proxy = WeakMethod(func, self._remove_proxy)
        except TypeError:
            proxy = _StrongRef(func)
        if proxy in self._func_cid_map[s]:
            return self._func_cid_map[s][proxy]

        cid = next(self._cid_gen)
        self._func_cid_map[s][proxy] = cid
        self.callbacks.setdefault(s, {})
        self.callbacks[s][cid] = proxy
        return cid

    def _remove_proxy(self, proxy):
        for signal, proxies in list(self._func_cid_map.items()):
            try:
                del self.callbacks[signal][proxies[proxy]]
            except KeyError:
                pass
            if len(self.callbacks[signal]) == 0:
                del self.callbacks[signal]
                del self._func_cid_map[signal]

    def disconnect(self, cid):
        """Disconnect the callback registered with callback id *cid*.
        """
        for eventname, callbackd in list(self.callbacks.items()):
            try:
                del callbackd[cid]
            except KeyError:
                continue
            else:
                for signal, functions in list(self._func_cid_map.items()):
                    for function, value in list(functions.items()):
                        if value == cid:
                            del functions[function]
                return

    def process(self, s, *args, **kwargs):
        """
        Process signal *s*.

        All of the functions registered to receive callbacks on *s* will be
        called with ``*args`` and ``**kwargs``.
        """
        for cid, ref in list(self.callbacks.get(s, {}).items()):
            func = ref()
            if func is not None:
                try:
                    func(*args, **kwargs)
                # this does not capture KeyboardInterrupt, SystemExit,
                # and GeneratorExit
                except Exception as exc:
                    if self.exception_handler is not None:
                        self.exception_handler(exc)
                    else:
                        raise


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

    __str__ = __repr__

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


@deprecated('3.0', alternative='types.SimpleNamespace')
class Bunch(types.SimpleNamespace):
    """
    Often we want to just collect a bunch of stuff together, naming each
    item of the bunch; a dictionary's OK for that, but a small do- nothing
    class is even handier, and prettier to use.  Whenever you want to
    group a few variables::

      >>> point = Bunch(datum=2, squared=4, coord=12)
      >>> point.datum
    """
    pass


def iterable(obj):
    """return true if *obj* is iterable"""
    try:
        iter(obj)
    except TypeError:
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
    return callable(getattr(obj, 'write', None))


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


@deprecated('3.0', 'isinstance(..., numbers.Number)')
def is_numlike(obj):
    """return true if *obj* looks like a number"""
    return isinstance(obj, (numbers.Number, np.number))


def to_filehandle(fname, flag='rU', return_opened=False, encoding=None):
    """
    *fname* can be an `os.PathLike` or a file handle.  Support for gzipped
    files is automatic, if the filename ends in .gz.  *flag* is a
    read/write flag for :func:`file`
    """
    if isinstance(fname, getattr(os, "PathLike", ())):
        fname = os.fspath(fname)
    if isinstance(fname, str):
        if fname.endswith('.gz'):
            # get rid of 'U' in flag for gzipped files.
            flag = flag.replace('U', '')
            fh = gzip.open(fname, flag)
        elif fname.endswith('.bz2'):
            # python may not be complied with bz2 support,
            # bury import until we need it
            import bz2
            # get rid of 'U' in flag for bz2 files
            flag = flag.replace('U', '')
            fh = bz2.BZ2File(fname, flag)
        else:
            fh = open(fname, flag, encoding=encoding)
        opened = True
    elif hasattr(fname, 'seek'):
        fh = fname
        opened = False
    else:
        raise ValueError('fname must be a PathLike or file handle')
    if return_opened:
        return fh, opened
    return fh


@contextlib.contextmanager
def open_file_cm(path_or_file, mode="r", encoding=None):
    r"""Pass through file objects and context-manage `.PathLike`\s."""
    fh, opened = to_filehandle(path_or_file, mode, True, encoding)
    if opened:
        with fh:
            yield fh
    else:
        yield fh


def is_scalar_or_string(val):
    """Return whether the given object is a scalar or string like."""
    return isinstance(val, str) or not iterable(val)


def _string_to_bool(s):
    """Parses the string argument as a boolean"""
    if not isinstance(s, str):
        return bool(s)
    warn_deprecated("2.2", "Passing one of 'on', 'true', 'off', 'false' as a "
                    "boolean is deprecated; use an actual boolean "
                    "(True/False) instead.")
    if s.lower() in ['on', 'true']:
        return True
    if s.lower() in ['off', 'false']:
        return False
    raise ValueError('String "%s" must be one of: '
                     '"on", "off", "true", or "false"' % s)


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
    # Don't trigger deprecation warning when just fetching.
    if dict.__getitem__(matplotlib.rcParams, 'examples.directory'):
        root = matplotlib.rcParams['examples.directory']
    else:
        root = os.path.join(matplotlib._get_data_path(), 'sample_data')
    path = os.path.join(root, fname)

    if asfileobj:
        if os.path.splitext(fname)[-1].lower() in ['.csv', '.xrc', '.txt']:
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
        if scalarp(item) or item is None:
            yield item
        else:
            yield from flatten(item, scalarp)


@deprecated("3.0")
def mkdirs(newdir, mode=0o777):
    """
    make directory *newdir* recursively, and set *mode*.  Equivalent to ::

        > mkdir -p NEWDIR
        > chmod MODE NEWDIR
    """
    # this functionality is now in core python as of 3.2
    # LPY DROP
    os.makedirs(newdir, mode=mode, exist_ok=True)


@deprecated('3.0')
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


@functools.lru_cache()
def get_realpath_and_stat(path):
    realpath = os.path.realpath(path)
    stat = os.stat(realpath)
    stat_key = (stat.st_ino, stat.st_dev)
    return realpath, stat_key


# A regular expression used to determine the amount of space to
# remove.  It looks for the first sequence of spaces immediately
# following the first newline, or at the beginning of the string.
_find_dedent_regex = re.compile(r"(?:(?:\n\r?)|^)( *)\S")
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


@deprecated("3.0")
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
    Stack of elements with a movable cursor.

    Mimics home/back/forward in a web browser.
    """

    def __init__(self, default=None):
        self.clear()
        self._default = default

    def __call__(self):
        """Return the current element, or None."""
        if not len(self._elements):
            return self._default
        else:
            return self._elements[self._pos]

    def __len__(self):
        return len(self._elements)

    def __getitem__(self, ind):
        return self._elements[ind]

    def forward(self):
        """Move the position forward and return the current element."""
        self._pos = min(self._pos + 1, len(self._elements) - 1)
        return self()

    def back(self):
        """Move the position back and return the current element."""
        if self._pos > 0:
            self._pos -= 1
        return self()

    def push(self, o):
        """
        Push *o* to the stack at current position.  Discard all later elements.

        *o* is returned.
        """
        self._elements = self._elements[:self._pos + 1] + [o]
        self._pos = len(self._elements) - 1
        return self()

    def home(self):
        """
        Push the first element onto the top of the stack.

        The first element is returned.
        """
        if not len(self._elements):
            return
        self.push(self._elements[0])
        return self()

    def empty(self):
        """Return whether the stack is empty."""
        return len(self._elements) == 0

    def clear(self):
        """Empty the stack."""
        self._pos = -1
        self._elements = []

    def bubble(self, o):
        """
        Raise *o* to the top of the stack.  *o* must be present in the stack.

        *o* is returned.
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
        """Remove *o* from the stack."""
        if o not in self._elements:
            raise ValueError('Unknown element o')
        old = self._elements[:]
        self.clear()
        for thiso in old:
            if thiso != o:
                self.push(thiso)


def report_memory(i=0):  # argument may go away
    """return the memory consumed by process"""
    from subprocess import Popen, PIPE
    pid = os.getpid()
    if sys.platform == 'sunos5':
        try:
            a2 = Popen(['ps', '-p', '%d' % pid, '-o', 'osz'],
                       stdout=PIPE).stdout.readlines()
        except OSError:
            raise NotImplementedError(
                "report_memory works on Sun OS only if "
                "the 'ps' program is found")
        mem = int(a2[-1].strip())
    elif sys.platform == 'linux':
        try:
            a2 = Popen(['ps', '-p', '%d' % pid, '-o', 'rss,sz'],
                       stdout=PIPE).stdout.readlines()
        except OSError:
            raise NotImplementedError(
                "report_memory works on Linux only if "
                "the 'ps' program is found")
        mem = int(a2[1].split()[1])
    elif sys.platform == 'darwin':
        try:
            a2 = Popen(['ps', '-p', '%d' % pid, '-o', 'rss,vsz'],
                       stdout=PIPE).stdout.readlines()
        except OSError:
            raise NotImplementedError(
                "report_memory works on Mac OS only if "
                "the 'ps' program is found")
        mem = int(a2[1].split()[0])
    elif sys.platform == 'win32':
        try:
            a2 = Popen(["tasklist", "/nh", "/fi", "pid eq %d" % pid],
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

            outstream.write("   %s -- " % type(step))
            if isinstance(step, dict):
                for key, val in step.items():
                    if val is next:
                        outstream.write("[{!r}]".format(key))
                        break
                    if key is next:
                        outstream.write("[key] = {!r}".format(val))
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
    using :meth:`joined`, and all disjoint sets can be retrieved by
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
        self._mapping = {weakref.ref(x): [weakref.ref(x)] for x in init}

    def __contains__(self, item):
        return weakref.ref(item) in self._mapping

    def clean(self):
        """Clean dead weak references from the dictionary."""
        mapping = self._mapping
        to_drop = [key for key in mapping if key() is None]
        for key in to_drop:
            val = mapping.pop(key)
            val.remove(key)

    def join(self, a, *args):
        """
        Join given arguments into the same set.  Accepts one or more arguments.
        """
        mapping = self._mapping
        set_a = mapping.setdefault(weakref.ref(a), [weakref.ref(a)])

        for arg in args:
            set_b = mapping.get(weakref.ref(arg), [weakref.ref(arg)])
            if set_b is not set_a:
                if len(set_b) > len(set_a):
                    set_a, set_b = set_b, set_a
                set_a.extend(set_b)
                for elem in set_b:
                    mapping[elem] = set_a

        self.clean()

    def joined(self, a, b):
        """Returns True if *a* and *b* are members of the same set."""
        self.clean()
        return (self._mapping.get(weakref.ref(a), object())
                is self._mapping.get(weakref.ref(b)))

    def remove(self, a):
        self.clean()
        set_a = self._mapping.pop(weakref.ref(a), None)
        if set_a:
            set_a.remove(weakref.ref(a))

    def __iter__(self):
        """
        Iterate over each of the disjoint sets as a list.

        The iterator is invalid if interleaved with calls to join().
        """
        self.clean()
        unique_groups = {id(group): group for group in self._mapping.values()}
        for group in unique_groups.values():
            yield [x() for x in group]

    def get_siblings(self, a):
        """Returns all of the items joined with *a*, including itself."""
        self.clean()
        siblings = self._mapping.get(weakref.ref(a), [weakref.ref(a)])
        return [x() for x in siblings]


def simple_linear_interpolation(a, steps):
    """
    Resample an array with ``steps - 1`` points between original point pairs.

    Parameters
    ----------
    a : array, shape (n, ...)
    steps : int

    Returns
    -------
    array, shape ``((n - 1) * steps + 1, ...)``

    Along each column of *a*, ``(steps - 1)`` points are introduced between
    each original values; the values are linearly interpolated.
    """
    fps = a.reshape((len(a), -1))
    xp = np.arange(len(a)) * steps
    x = np.arange((len(a) - 1) * steps + 1)
    return (np.column_stack([np.interp(x, xp, fp) for fp in fps.T])
            .reshape((len(x),) + a.shape[1:]))


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
    if is_scalar_or_string(args[0]):
        raise ValueError("First argument must be a sequence")
    nrecs = len(args[0])
    margs = []
    seqlist = [False] * len(args)
    for i, x in enumerate(args):
        if not isinstance(x, str) and iterable(x) and len(x) == nrecs:
            seqlist[i] = True
            if isinstance(x, np.ma.MaskedArray):
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
            if isinstance(x, np.ma.MaskedArray):
                masks.append(~np.ma.getmaskarray(x))  # invert the mask
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
        mask = np.logical_and.reduce(masks)
        igood = mask.nonzero()[0]
        if len(igood) < nrecs:
            for i, x in enumerate(margs):
                if seqlist[i]:
                    margs[i] = x.take(igood, axis=0)
    for i, x in enumerate(margs):
        if seqlist[i] and isinstance(x, np.ma.MaskedArray):
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
        As a float, determines the reach of the whiskers to the beyond the
        first and third quartiles. In other words, where IQR is the
        interquartile range (`Q3-Q1`), the upper whisker will extend to last
        datum less than `Q3 + whis*IQR`). Similarly, the lower whisker will
        extend to the first datum greater than `Q1 - whis*IQR`.
        Beyond the whiskers, data are considered outliers
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
        When `True` and the data are distributed such that the 25th and 75th
        percentiles are equal, ``whis`` is set to ``'range'`` such that the
        whisker ends are at the minimum and maximum of the data.

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

        \\mathrm{med} \\pm 1.57 \\times \\frac{\\mathrm{iqr}}{\\sqrt{N}}

    General approach from:
    McGill, R., Tukey, J.W., and Larsen, W.A. (1978) "Variations of
    Boxplots", The American Statistician, 32:12-16.

    """

    def _bootstrap_median(data, N=5000):
        # determine 95% confidence intervals of the median
        M = len(data)
        percentiles = [2.5, 97.5]

        bs_index = np.random.randint(M, size=(N, M))
        bsData = data[bs_index]
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
    X = _reshape_2D(X, "X")

    ncols = len(X)
    if labels is None:
        labels = itertools.repeat(None)
    elif len(labels) != ncols:
        raise ValueError("Dimensions of labels and X must be compatible")

    input_whis = whis
    for ii, (x, label) in enumerate(zip(X, labels)):

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
                raise ValueError('whis must be a float, valid string, or list '
                                 'of percentiles')
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


# The ls_mapper maps short codes for line style to their full name used by
# backends; the reverse mapper is for mapping full names to short ones.
ls_mapper = {'-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted'}
ls_mapper_r = {v: k for k, v in ls_mapper.items()}


@deprecated('2.2')
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


def contiguous_regions(mask):
    """
    Return a list of (ind0, ind1) such that mask[ind0:ind1].all() is
    True and we cover all such regions
    """
    mask = np.asarray(mask, dtype=bool)

    if not mask.size:
        return []

    # Find the indices of region changes, and correct offset
    idx, = np.nonzero(mask[:-1] != mask[1:])
    idx += 1

    # List operations are faster for moderately sized arrays
    idx = idx.tolist()

    # Add first and/or last index if needed
    if mask[0]:
        idx = [0] + idx
    if mask[-1]:
        idx.append(len(mask))

    return list(zip(idx[::2], idx[1::2]))


def is_math_text(s):
    # Did we find an even number of non-escaped dollar signs?
    # If so, treat is as math text.
    s = str(s)
    dollar_count = s.count(r'$') - s.count(r'\$')
    even_dollars = (dollar_count > 0 and dollar_count % 2 == 0)
    return even_dollars


def _to_unmasked_float_array(x):
    """
    Convert a sequence to a float array; if input was a masked array, masked
    values are converted to nans.
    """
    if hasattr(x, 'mask'):
        return np.ma.asarray(x, float).filled(np.nan)
    else:
        return np.asarray(x, float)


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


def _reshape_2D(X, name):
    """
    Use Fortran ordering to convert ndarrays and lists of iterables to lists of
    1D arrays.

    Lists of iterables are converted by applying `np.asarray` to each of their
    elements.  1D ndarrays are returned in a singleton list containing them.
    2D ndarrays are converted to the list of their *columns*.

    *name* is used to generate the error message for invalid inputs.
    """
    # Iterate over columns for ndarrays, over rows otherwise.
    X = np.atleast_1d(X.T if isinstance(X, np.ndarray) else np.asarray(X))
    if X.ndim == 1 and X.dtype.type != np.object_:
        # 1D array of scalars: directly return it.
        return [X]
    elif X.ndim in [1, 2]:
        # 2D array, or 1D array of iterables: flatten them first.
        return [np.reshape(x, -1) for x in X]
    else:
        raise ValueError("{} must have 2 or fewer dimensions".format(name))


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
    X = _reshape_2D(X, "X")

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


def pts_to_prestep(x, *args):
    """
    Convert continuous line to pre-steps.

    Given a set of ``N`` points, convert to ``2N - 1`` points, which when
    connected linearly give a step function which changes values at the
    beginning of the intervals.

    Parameters
    ----------
    x : array
        The x location of the steps. May be empty.

    y1, ..., yp : array
        y arrays to be turned into steps; all must be the same length as ``x``.

    Returns
    -------
    out : array
        The x and y values converted to steps in the same order as the input;
        can be unpacked as ``x_out, y1_out, ..., yp_out``.  If the input is
        length ``N``, each of these arrays will be length ``2N + 1``. For
        ``N=0``, the length will be 0.

    Examples
    --------
    >> x_s, y1_s, y2_s = pts_to_prestep(x, y1, y2)
    """
    steps = np.zeros((1 + len(args), max(2 * len(x) - 1, 0)))
    # In all `pts_to_*step` functions, only assign *once* using `x` and `args`,
    # as converting to an array may be expensive.
    steps[0, 0::2] = x
    steps[0, 1::2] = steps[0, 0:-2:2]
    steps[1:, 0::2] = args
    steps[1:, 1::2] = steps[1:, 2::2]
    return steps


def pts_to_poststep(x, *args):
    """
    Convert continuous line to post-steps.

    Given a set of ``N`` points convert to ``2N + 1`` points, which when
    connected linearly give a step function which changes values at the end of
    the intervals.

    Parameters
    ----------
    x : array
        The x location of the steps. May be empty.

    y1, ..., yp : array
        y arrays to be turned into steps; all must be the same length as ``x``.

    Returns
    -------
    out : array
        The x and y values converted to steps in the same order as the input;
        can be unpacked as ``x_out, y1_out, ..., yp_out``.  If the input is
        length ``N``, each of these arrays will be length ``2N + 1``. For
        ``N=0``, the length will be 0.

    Examples
    --------
    >> x_s, y1_s, y2_s = pts_to_poststep(x, y1, y2)
    """
    steps = np.zeros((1 + len(args), max(2 * len(x) - 1, 0)))
    steps[0, 0::2] = x
    steps[0, 1::2] = steps[0, 2::2]
    steps[1:, 0::2] = args
    steps[1:, 1::2] = steps[1:, 0:-2:2]
    return steps


def pts_to_midstep(x, *args):
    """
    Convert continuous line to mid-steps.

    Given a set of ``N`` points convert to ``2N`` points which when connected
    linearly give a step function which changes values at the middle of the
    intervals.

    Parameters
    ----------
    x : array
        The x location of the steps. May be empty.

    y1, ..., yp : array
        y arrays to be turned into steps; all must be the same length as
        ``x``.

    Returns
    -------
    out : array
        The x and y values converted to steps in the same order as the input;
        can be unpacked as ``x_out, y1_out, ..., yp_out``.  If the input is
        length ``N``, each of these arrays will be length ``2N``.

    Examples
    --------
    >> x_s, y1_s, y2_s = pts_to_midstep(x, y1, y2)
    """
    steps = np.zeros((1 + len(args), 2 * len(x)))
    x = np.asanyarray(x)
    steps[0, 1:-1:2] = steps[0, 2::2] = (x[:-1] + x[1:]) / 2
    steps[0, :1] = x[:1]  # Also works for zero-sized input.
    steps[0, -1:] = x[-1:]
    steps[1:, 0::2] = args
    steps[1:, 1::2] = steps[1:, 0::2]
    return steps


STEP_LOOKUP_MAP = {'default': lambda x, y: (x, y),
                   'steps': pts_to_prestep,
                   'steps-pre': pts_to_prestep,
                   'steps-post': pts_to_poststep,
                   'steps-mid': pts_to_midstep}


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
        y = _check_1d(y)
        return np.arange(y.shape[0], dtype=float), y


def safe_first_element(obj):
    if isinstance(obj, collections.abc.Iterator):
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


def sanitize_sequence(data):
    """Converts dictview object to list"""
    return (list(data) if isinstance(data, collections.abc.MappingView)
            else data)


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
    for canonical, alias_list in alias_mapping.items():

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
        allowed_set = {*required, *allowed}
        fail_keys = [k for k in ret if k not in allowed_set]
        if fail_keys:
            raise TypeError(
                "kwargs contains {keys!r} which are not in the required "
                "{req!r} or allowed {allow!r} keys".format(
                    keys=fail_keys, req=required, allow=allowed))

    return ret


def get_label(y, default_name):
    try:
        return y.name
    except AttributeError:
        return default_name


_lockstr = """\
LOCKERROR: matplotlib is trying to acquire the lock
    {!r}
and has failed.  This maybe due to any other process holding this
lock.  If you are sure no other matplotlib process is running try
removing these folders and trying again.
"""


@deprecated("3.0")
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


@contextlib.contextmanager
def _lock_path(path):
    """
    Context manager for locking a path.

    Usage::

        with _lock_path(path):
            ...

    Another thread or process that attempts to lock the same path will wait
    until this context manager is exited.

    The lock is implemented by creating a temporary file in the parent
    directory, so that directory must exist and be writable.
    """
    path = Path(path)
    lock_path = path.with_name(path.name + ".matplotlib-lock")
    retries = 50
    sleeptime = 0.1
    for _ in range(retries):
        try:
            with lock_path.open("xb"):
                break
        except FileExistsError:
            time.sleep(sleeptime)
    else:
        raise TimeoutError("""\
Lock error: Matplotlib failed to acquire the following lock file:
    {}
This maybe due to another process holding this lock file.  If you are sure no
other Matplotlib process is running, remove this file and try again.""".format(
            lock_path))
    try:
        yield
    finally:
        lock_path.unlink()


def _topmost_artist(
        artists,
        _cached_max=functools.partial(max, key=operator.attrgetter("zorder"))):
    """Get the topmost artist of a list.

    In case of a tie, return the *last* of the tied artists, as it will be
    drawn on top of the others. `max` returns the first maximum in case of
    ties, so we need to iterate over the list in reverse order.
    """
    return _cached_max(reversed(artists))


def _str_equal(obj, s):
    """Return whether *obj* is a string equal to string *s*.

    This helper solely exists to handle the case where *obj* is a numpy array,
    because in such cases, a naive ``obj == s`` would yield an array, which
    cannot be used in a boolean context.
    """
    return isinstance(obj, str) and obj == s


def _str_lower_equal(obj, s):
    """Return whether *obj* is a string equal, when lowercased, to string *s*.

    This helper solely exists to handle the case where *obj* is a numpy array,
    because in such cases, a naive ``obj == s`` would yield an array, which
    cannot be used in a boolean context.
    """
    return isinstance(obj, str) and obj.lower() == s


def _define_aliases(alias_d, cls=None):
    """Class decorator for defining property aliases.

    Use as ::

        @cbook._define_aliases({"property": ["alias", ...], ...})
        class C: ...

    For each property, if the corresponding ``get_property`` is defined in the
    class so far, an alias named ``get_alias`` will be defined; the same will
    be done for setters.  If neither the getter nor the setter exists, an
    exception will be raised.

    The alias map is stored as the ``_alias_map`` attribute on the class and
    can be used by `~.normalize_kwargs` (which assumes that higher priority
    aliases come last).
    """
    if cls is None:  # Return the actual class decorator.
        return functools.partial(_define_aliases, alias_d)

    def make_alias(name):  # Enforce a closure over *name*.
        def method(self, *args, **kwargs):
            return getattr(self, name)(*args, **kwargs)
        return method

    for prop, aliases in alias_d.items():
        exists = False
        for prefix in ["get_", "set_"]:
            if prefix + prop in vars(cls):
                exists = True
                for alias in aliases:
                    method = make_alias(prefix + prop)
                    method.__name__ = prefix + alias
                    method.__doc__ = "alias for `{}`".format(prefix + prop)
                    setattr(cls, prefix + alias, method)
        if not exists:
            raise ValueError(
                "Neither getter nor setter exists for {!r}".format(prop))

    if hasattr(cls, "_alias_map"):
        # Need to decide on conflict resolution policy.
        raise NotImplementedError("Parent class already defines aliases")
    cls._alias_map = alias_d
    return cls


def _array_perimeter(arr):
    """
    Get the elements on the perimeter of ``arr``,

    Parameters
    ----------
    arr : ndarray, shape (M, N)
        The input array

    Returns
    -------
    perimeter : ndarray, shape (2*(M - 1) + 2*(N - 1),)
        The elements on the perimeter of the array::

            [arr[0,0] ... arr[0,-1] ... arr[-1, -1] ... arr[-1,0] ...]

    Examples
    --------
    >>> i, j = np.ogrid[:3,:4]
    >>> a = i*10 + j
    >>> a
    array([[ 0,  1,  2,  3],
           [10, 11, 12, 13],
           [20, 21, 22, 23]])
    >>> _array_perimeter(a)
    array([ 0,  1,  2,  3, 13, 23, 22, 21, 20, 10])
    """
    # note we use Python's half-open ranges to avoid repeating
    # the corners
    forward = np.s_[0:-1]      # [0 ... -1)
    backward = np.s_[-1:0:-1]  # [-1 ... 0)
    return np.concatenate((
        arr[0, forward],
        arr[forward, -1],
        arr[-1, backward],
        arr[backward, 0],
    ))


@contextlib.contextmanager
def _setattr_cm(obj, **kwargs):
    """Temporarily set some attributes; restore original state at context exit.
    """
    sentinel = object()
    origs = [(attr, getattr(obj, attr, sentinel)) for attr in kwargs]
    try:
        for attr, val in kwargs.items():
            setattr(obj, attr, val)
        yield
    finally:
        for attr, orig in origs:
            if orig is sentinel:
                delattr(obj, attr)
            else:
                setattr(obj, attr, orig)


def _warn_external(message, category=None):
    """
    `warnings.warn` wrapper that sets *stacklevel* to "outside Matplotlib".

    The original emitter of the warning can be obtained by patching this
    function back to `warnings.warn`, i.e. ``cbook._warn_external =
    warnings.warn`` (or ``functools.partial(warnings.warn, stacklevel=2)``,
    etc.).
    """
    frame = sys._getframe()
    for stacklevel in itertools.count(1):
        if not re.match(r"\A(matplotlib|mpl_toolkits)(\Z|\.)",
                        frame.f_globals["__name__"]):
            break
        frame = frame.f_back
    warnings.warn(message, category, stacklevel)


class _OrderedSet(collections.abc.MutableSet):
    def __init__(self):
        self._od = collections.OrderedDict()

    def __contains__(self, key):
        return key in self._od

    def __iter__(self):
        return iter(self._od)

    def __len__(self):
        return len(self._od)

    def add(self, key):
        self._od.pop(key, None)
        self._od[key] = None

    def discard(self, key):
        self._od.pop(key, None)


# Agg's buffers are unmultiplied RGBA8888, which neither PyQt4 nor cairo
# support; however, both do support premultiplied ARGB32.


def _premultiplied_argb32_to_unmultiplied_rgba8888(buf):
    """
    Convert a premultiplied ARGB32 buffer to an unmultiplied RGBA8888 buffer.
    """
    rgba = np.take(  # .take() ensures C-contiguity of the result.
        buf,
        [2, 1, 0, 3] if sys.byteorder == "little" else [1, 2, 3, 0], axis=2)
    rgb = rgba[..., :-1]
    alpha = rgba[..., -1]
    # Un-premultiply alpha.  The formula is the same as in cairo-png.c.
    mask = alpha != 0
    for channel in np.rollaxis(rgb, -1):
        channel[mask] = (
            (channel[mask].astype(int) * 255 + alpha[mask] // 2)
            // alpha[mask])
    return rgba


def _unmultiplied_rgba8888_to_premultiplied_argb32(rgba8888):
    """
    Convert an unmultiplied RGBA8888 buffer to a premultiplied ARGB32 buffer.
    """
    if sys.byteorder == "little":
        argb32 = np.take(rgba8888, [2, 1, 0, 3], axis=2)
        rgb24 = argb32[..., :-1]
        alpha8 = argb32[..., -1:]
    else:
        argb32 = np.take(rgba8888, [3, 0, 1, 2], axis=2)
        alpha8 = argb32[..., :1]
        rgb24 = argb32[..., 1:]
    # Only bother premultiplying when the alpha channel is not fully opaque,
    # as the cost is not negligible.  The unsafe cast is needed to do the
    # multiplication in-place in an integer buffer.
    if alpha8.min() != 0xff:
        np.multiply(rgb24, alpha8 / 0xff, out=rgb24, casting="unsafe")
    return argb32
