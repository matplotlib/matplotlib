from __future__ import print_function

import os
import io
import traceback
import gzip
import re
import sys
import types
import new
import warnings
import errno
import subprocess
import numpy.ma as ma
import locale
from weakref import ref, WeakKeyDictionary

import numpy as np

import matplotlib
from matplotlib import MatplotlibDeprecationWarning as mplDeprecation

# On some systems, locale.getpreferredencoding returns None,
# which can break unicode; and the sage project reports that
# some systems have incorrect locale specifications, e.g.,
# an encoding instead of a valid locale name.  Another
# pathological case that has been reported is an empty string.

# On some systems, getpreferredencoding sets the locale, which has
# side effects.  Passing False eliminates those side effects.

if sys.version_info[0] >= 3:
    def unicode_safe(s):
        try:
            preferredencoding = locale.getpreferredencoding(
                matplotlib.rcParams['axes.formatter.use_locale']).strip()
            if not preferredencoding:
                preferredencoding = None
        except (ValueError, ImportError, AttributeError):
            preferredencoding = None

        if isinstance(s, bytes):
            if preferredencoding is None:
                return unicode(s)
            else:
                # We say "unicode" and not "str" here so it passes through
                # 2to3 correctly.
                return unicode(s, preferredencoding)
        return s
else:
    def unicode_safe(s):
        try:
            preferredencoding = locale.getpreferredencoding(
                matplotlib.rcParams['axes.formatter.use_locale']).strip()
            if not preferredencoding:
                preferredencoding = None
        except (ValueError, ImportError, AttributeError):
            preferredencoding = None

        if preferredencoding is None:
            return unicode(s)
        else:
            return unicode(s, preferredencoding)


# a dict to cross-map linestyle arguments
_linestyles = [('-', 'solid'),
               ('--', 'dashed'),
               ('-.', 'dashdot'),
               (':', 'dotted')]

ls_mapper = dict(_linestyles)
ls_mapper.update([(ls[1], ls[0]) for ls in _linestyles])


# TODO rename this method is_iterable
def iterable(obj):
    """return true if *obj* is iterable"""
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def is_writable_file_like(obj):
    """return true if *obj* looks like a file object with a *write* method"""
    return hasattr(obj, 'write') and callable(obj.write)


def is_string_like(obj):
    """Return True if *obj* looks like a string"""
    if isinstance(obj, (str, unicode)):
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


def is_numlike(obj):
    """return true if *obj* looks like a number"""
    try:
        obj + 1
    except:
        return False
    else:
        return True


def is_scalar(obj):
    """return true if *obj* is not string like and is not iterable"""
    return not is_string_like(obj) and not iterable(obj)


def is_scalar_or_string(val):
    """Return whether the given object is a scalar or string like."""
    return is_string_like(val) or not iterable(val)


def is_math_text(s):
    # Did we find an even number of non-escaped dollar signs?
    # If so, treat is as math text.
    try:
        s = unicode(s)
    except UnicodeDecodeError:
        raise ValueError(
            "matplotlib display text must have all code points < 128 or use "
            "Unicode strings")

    dollar_count = s.count(r'$') - s.count(r'\$')
    even_dollars = (dollar_count > 0 and dollar_count % 2 == 0)

    return even_dollars


def is_sequence_of_strings(obj):
    """
    Returns true if *obj* is iterable and contains strings
    """
    if not iterable(obj):
        return False
    if is_string_like(obj):
        return False
    for o in obj:
        if not is_string_like(o):
            return False
    return True


def issubclass_safe(x, klass):
    """return issubclass(x, klass) and return False on a TypeError"""

    try:
        return issubclass(x, klass)
    except TypeError:
        return False


def reverse_dict(d):
    """reverse the dictionary -- may lose data if values are not unique!"""
    return dict([(v, k) for k, v in d.iteritems()])


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


def popall(seq):
    """empty a list"""
    for i in xrange(len(seq)):
        seq.pop()


def strip_math(s):
    """remove latex formatting from mathtext"""
    remove = (r'\mathdefault', r'\rm', r'\cal', r'\tt', r'\it', '\\', '{', '}')
    s = s[1:-1]
    for r in remove:
        s = s.replace(r, '')
    return s


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
    while 1:
        minvals = ([_f for _f in [it.key for it in iters] if _f])
        if minvals:
            minkey = min(minvals)
            yield (minkey, [it(minkey) for it in iters])
        else:
            break


def mkdirs(newdir, mode=0o777):
    """
    make directory *newdir* recursively, and set *mode*.  Equivalent to ::

        > mkdir -p NEWDIR
        > chmod MODE NEWDIR
    """
    try:
        if not os.path.exists(newdir):
            parts = os.path.split(newdir)
            for i in range(1, len(parts) + 1):
                thispart = os.path.join(*parts[:i])
                if not os.path.exists(thispart):
                    os.makedirs(thispart, mode)

    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


def flatten(seq, scalarp=is_scalar_or_string):
    """
    Returns a generator of flattened nested containers

    For example:

        >>> from matplotlib.utils import flatten
        >>> l = (('John', ['Hunter']), (1, 23), [[([42, (5, 23)], )]])
        >>> print list(flatten(l))
        ['John', 'Hunter', 1, 23, 42, 5, 23]

    By: Composite of Holger Krekel and Luther Blissett
    From: http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/121294
    and Recipe 1.12 in cookbook
    """
    for item in seq:
        if scalarp(item):
            yield item
        else:
            for subitem in flatten(item, scalarp):
                yield subitem


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

        >>> from matplotlib.utils import Grouper
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
        [(d, e), (a, b, c)]
        >>> grp.joined(a, b)
        True
        >>> grp.joined(a, c)
        True
        >>> grp.joined(a, d)
        False

    """
    def __init__(self, init=[]):
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
        for group in self._mapping.itervalues():
            if not group[-1] is token:
                yield [x() for x in group]
                group.append(token)

        # Cleanup the tokens
        for group in self._mapping.itervalues():
            if group[-1] is token:
                del group[-1]

    def get_siblings(self, a):
        """
        Returns all of the items joined with *a*, including itself.
        """
        self.clean()

        siblings = self._mapping.get(ref(a), [ref(a)])
        return [x() for x in siblings]


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
        try:
            try:
                self.inst = ref(cb.im_self)
            except TypeError:
                self.inst = None
            self.func = cb.im_func
            self.klass = cb.im_class
        except AttributeError:
            self.inst = None
            self.func = cb
            self.klass = None

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
            if sys.version_info[0] >= 3:
                mtd = types.MethodType(self.func, self.inst())
            else:
                mtd = new.instancemethod(self.func, self.inst(), self.klass)
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


class CallbackRegistry:
    """
    Handle registering and disconnecting for a set of signals and
    callbacks:

        >>> def oneat(x):
        ...    print 'eat', x
        >>> def ondrink(x):
        ...    print 'drink', x

        >>> from matplotlib.utils import CallbackRegistry
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
    <http://mindtrove.info/articles/python-weak-references/>`_.

    .. deprecated:: 1.3.0
    """
    def __init__(self, *args):
        if len(args):
            warnings.warn(
                "CallbackRegistry no longer requires a list of callback "
                "types. Ignoring arguments. *args will be removed in 1.5",
                mplDeprecation)
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
        if func in self._func_cid_map[s]:
            return self._func_cid_map[s][func]

        self._cid += 1
        cid = self._cid
        self._func_cid_map[s][func] = cid
        self.callbacks.setdefault(s, dict())
        proxy = _BoundMethodProxy(func)
        self.callbacks[s][cid] = proxy
        return cid

    def disconnect(self, cid):
        """
        disconnect the callback registered with callback id *cid*
        """
        for eventname, callbackd in self.callbacks.items():
            try:
                del callbackd[cid]
            except KeyError:
                continue
            else:
                for key, value in self._func_cid_map.items():
                    if value == cid:
                        del self._func_cid_map[key]
                return

    def process(self, s, *args, **kwargs):
        """
        process signal *s*.  All of the functions registered to receive
        callbacks on *s* will be called with *\*args* and *\*\*kwargs*
        """
        if s in self.callbacks:
            for cid, proxy in self.callbacks[s].items():
                # Clean out dead references
                if proxy.inst is not None and proxy.inst() is None:
                    del self.callbacks[s][cid]
                else:
                    proxy(*args, **kwargs)


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


class Bunch:
    """
    Often we want to just collect a bunch of stuff together, naming each
    item of the bunch; a dictionary's OK for that, but a small do- nothing
    class is even handier, and prettier to use.  Whenever you want to
    group a few variables::

      >>> point = Bunch(datum=2, squared=4, coord=12)
      >>> point.datum

      By: Alex Martelli
      From: http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52308
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __repr__(self):
        keys = self.__dict__.iterkeys()
        return 'Bunch(%s)' % ', '.join(['%s=%s' % (k, self.__dict__[k])
                                        for k
                                        in keys])


_safezip_msg = 'In safezip, len(args[0])=%d but len(args[%d])=%d'


def safezip(*args):
    """make sure *args* are equal len before zipping"""
    Nx = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != Nx:
            raise ValueError(_safezip_msg % (Nx, i + 1, len(arg)))
    return zip(*args)


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


def restrict_dict(d, keys):
    """
    Return a dictionary that contains those keys that appear in both
    d and keys, with values from d.
    """
    return dict([(k, v) for (k, v) in d.iteritems() if k in keys])


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
        'return the current element, or None'
        if not len(self._elements):
            return self._default
        else:
            return self._elements[self._pos]

    def __len__(self):
        return self._elements.__len__()

    def __getitem__(self, ind):
        return self._elements.__getitem__(ind)

    def forward(self):
        'move the position forward and return the current element'
        N = len(self._elements)
        if self._pos < N - 1:
            self._pos += 1
        return self()

    def back(self):
        'move the position back and return the current element'
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
        'push the first element onto the top of the stack'
        if not len(self._elements):
            return
        self.push(self._elements[0])
        return self()

    def empty(self):
        return len(self._elements) == 0

    def clear(self):
        'empty the stack'
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


class maxdict(dict):
    """
    A dictionary with a maximum size; this doesn't override all the
    relevant methods to contrain size, just setitem, so use with
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


class GetRealpathAndStat:
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


def _check_output(*popenargs, **kwargs):
    r"""Run command with arguments and return its output as a byte
    string.

    If the exit code was non-zero it raises a CalledProcessError.  The
    CalledProcessError object will have the return code in the
    returncode
    attribute and output in the output attribute.

    The arguments are the same as for the Popen constructor.  Example::

    >>> check_output(["ls", "-l", "/dev/null"])
    'crw-rw-rw- 1 root root 1, 3 Oct 18  2007 /dev/null\n'

    The stdout argument is not allowed as it is used internally.
    To capture standard error in the result, use stderr=STDOUT.::

    >>> check_output(["/bin/sh", "-c",
    ...               "ls -l non_existent_file ; exit 0"],
    ...              stderr=STDOUT)
    'ls: non_existent_file: No such file or directory\n'
    """
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd, output=output)
    return output


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


# python2.7's subprocess provides a check_output method
if hasattr(subprocess, 'check_output'):
    check_output = subprocess.check_output
else:
    check_output = _check_output


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


def exception_to_str(s=None):

    sh = io.StringIO()
    if s is not None:
        print(s, file=sh)
    traceback.print_exc(file=sh)
    return sh.getvalue()


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


def safe_masked_invalid(x):
    x = np.asanyarray(x)
    try:
        xm = np.ma.masked_invalid(x, copy=False)
        xm.shrink_mask()
    except TypeError:
        return x
    return xm


class _InstanceMethodPickler(object):
    """
    Pickle cannot handle instancemethod saving. _InstanceMethodPickler
    provides a solution to this.
    """
    def __init__(self, instancemethod):
        """Takes an instancemethod as its only argument."""
        self.parent_obj = instancemethod.im_self
        self.instancemethod_name = instancemethod.im_func.__name__

    def get_instancemethod(self):
        return getattr(self.parent_obj, self.instancemethod_name)


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


def simple_linear_interpolation(a, steps):
    if steps == 1:
        return a

    steps = np.floor(steps)
    new_length = ((len(a) - 1) * steps) + 1
    new_shape = list(a.shape)
    new_shape[0] = new_length
    result = np.zeros(new_shape, a.dtype)

    result[0] = a[0]
    a0 = a[0:-1]
    a1 = a[1:]
    delta = ((a1 - a0) / steps)

    for i in range(1, int(steps)):
        result[i::steps] = delta * i + a0
    result[steps::steps] = a1

    return result


def report_memory(i=0):  # argument may go away
    """return the memory consumed by process"""
    from subprocess import Popen, PIPE
    pid = os.getpid()
    if sys.platform == 'sunos5':
        a2 = Popen('ps -p %d -o osz' % pid, shell=True,
                   stdout=PIPE).stdout.readlines()
        mem = int(a2[-1].strip())
    elif sys.platform.startswith('linux'):
        a2 = Popen('ps -p %d -o rss,sz' % pid, shell=True,
                   stdout=PIPE).stdout.readlines()
        mem = int(a2[1].split()[1])
    elif sys.platform.startswith('darwin'):
        a2 = Popen('ps -p %d -o rss,vsz' % pid, shell=True,
                   stdout=PIPE).stdout.readlines()
        mem = int(a2[1].split()[0])
    elif sys.platform.startswith('win'):
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
                for key, val in step.iteritems():
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
