"""
A collection of utility functions and classes.  Many (but not all)
from the Python Cookbook -- hence the name _cbook
"""
from __future__ import print_function

import datetime
import glob
import gzip
import os
import re
import sys
import threading
import time

import matplotlib

import numpy as np
from matplotlib.utils import *


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
    'convert to string or None'
    def __init__(self, missing='Null', missingval=''):
        converter.__init__(self, missing=missing, missingval=missingval)


class todatetime(converter):
    'convert to a datetime or None'
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
    'convert to a date or None'
    def __init__(self, fmt='%Y-%m-%d', missing='Null', missingval=None):
        'use a :func:`time.strptime` format string for conversion'
        converter.__init__(self, missing, missingval)
        self.fmt = fmt

    def __call__(self, s):
        if self.is_missing(s):
            return self.missingval
        tup = time.strptime(s, self.fmt)
        return datetime.date(*tup[:3])


class tofloat(converter):
    'convert to a float or None'
    def __init__(self, missing='Null', missingval=None):
        converter.__init__(self, missing)
        self.missingval = missingval

    def __call__(self, s):
        if self.is_missing(s):
            return self.missingval
        return float(s)


class toint(converter):
    'convert to an int or None'
    def __init__(self, missing='Null', missingval=None):
        converter.__init__(self, missing)

    def __call__(self, s):
        if self.is_missing(s):
            return self.missingval
        return int(s)


class Scheduler(threading.Thread):
    """
    Base class for timeout and idle scheduling
    """
    idlelock = threading.Lock()
    id = 0

    def __init__(self):
        threading.Thread.__init__(self)
        self.id = Scheduler.id
        self._stopped = False
        Scheduler.id += 1
        self._stopevent = threading.Event()

    def stop(self):
        if self._stopped:
            return
        self._stopevent.set()
        self.join()
        self._stopped = True


class Timeout(Scheduler):
    """
    Schedule recurring events with a wait time in seconds
    """
    def __init__(self, wait, func):
        Scheduler.__init__(self)
        self.wait = wait
        self.func = func

    def run(self):

        while not self._stopevent.isSet():
            self._stopevent.wait(self.wait)
            Scheduler.idlelock.acquire()
            b = self.func(self)
            Scheduler.idlelock.release()
            if not b:
                break


class Idle(Scheduler):
    """
    Schedule callbacks when scheduler is idle
    """
    # the prototype impl is a bit of a poor man's idle handler.  It
    # just implements a short wait time.  But it will provide a
    # placeholder for a proper impl ater
    waittime = 0.05

    def __init__(self, func):
        Scheduler.__init__(self)
        self.func = func

    def run(self):

        while not self._stopevent.isSet():
            self._stopevent.wait(Idle.waittime)
            Scheduler.idlelock.acquire()
            b = self.func(self)
            Scheduler.idlelock.release()
            if not b:
                break


def unique(x):
    'Return a list of unique elements of *x*'
    return dict([(val, 1) for val in x]).keys()


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
    if matplotlib.rcParams['examples.directory']:
        root = matplotlib.rcParams['examples.directory']
    else:
        root = os.path.join(os.path.dirname(__file__),
            "mpl-data", "sample_data")
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


class Sorter:
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

      print multiple_replace(adict, text)

      xlat = Xlator(adict)
      print xlat.xlat(text)
    """

    def _make_regex(self):
        """ Build re object based on the keys of the current dictionary """
        return re.compile("|".join(map(re.escape, self.iterkeys())))

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


class Null:
    """ Null objects always and reliably "do nothing." """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __str__(self):
        return "Null()"

    def __repr__(self):
        return "Null()"

    def __nonzero__(self):
        return 0

    def __getattr__(self, name):
        return self

    def __setattr__(self, name, value):
        return self

    def __delattr__(self, name):
        return self


def dict_delall(d, keys):
    'delete all of the *keys* from the :class:`dict` *d*'
    for key in keys:
        try:
            del d[key]
        except KeyError:
            pass


class RingBuffer:
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
            # FIXME __Full is not defined
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

    sLen = 0
    # todo: use Alex's xrange pattern from the cbook for efficiency
    for (word, ind) in zip(seq, xrange(len(seq))):
        sLen += len(word) + 1  # +1 to account for the len(' ')
        if sLen >= N:
            return ind
    return len(seq)


def wrap(prefix, text, cols):
    'wrap *text* with *prefix* at length *cols*'
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
    "Break up the *seq* into *num* tuples"
    start = 0
    while 1:
        item = seq[start:start + num]
        if not len(item):
            break
        yield item
        start += num


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


def report_memory(i=0):  # argument may go away
    'return the memory consumed by process'
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


class MemoryMonitor:
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
        ii = range(0, n, dn)
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
            from pylab import figure
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


def recursive_remove(path):
    if os.path.isdir(path):
        for fname in glob.glob(os.path.join(path, '*')) + \
                     glob.glob(os.path.join(path, '.*')):
            if os.path.isdir(fname):
                recursive_remove(fname)
                os.removedirs(fname)
            else:
                os.remove(fname)
        #os.removedirs(path)
    else:
        os.remove(path)


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
