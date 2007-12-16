"""
A collection of utility functions and classes.  Many (but not all)
from the Python Cookbook -- hence the name cbook
"""
from __future__ import generators
import re, os, errno, sys, StringIO, traceback, locale
import time, datetime
import numpy as npy

try: set
except NameError:
    from sets import Set as set

major, minor1, minor2, s, tmp = sys.version_info


# on some systems, locale.getpreferredencoding returns None, which can break unicode
preferredencoding = locale.getpreferredencoding()

def unicode_safe(s):
    if preferredencoding is None: return unicode(s)
    else: return unicode(s, preferredencoding)

class converter:
    """
    Base class for handling string -> python type with support for
    missing values
    """
    def __init__(self, missing='Null', missingval=None):
        self.missing = missing
        self.missingval = missingval
    def __call__(self, s):
        if s==self.missing: return self.missingval
        return s

    def is_missing(self, s):
        return not s.strip() or s==self.missing

class tostr(converter):
    'convert to string or None'
    def __init__(self, missing='Null', missingval=''):
        converter.__init__(self, missing=missing, missingval=missingval)

class todatetime(converter):
    'convert to a datetime or None'
    def __init__(self, fmt='%Y-%m-%d', missing='Null', missingval=None):
        'use a time.strptime format string for conversion'
        converter.__init__(self, missing, missingval)
        self.fmt = fmt

    def __call__(self, s):
        if self.is_missing(s): return self.missingval
        tup = time.strptime(s, self.fmt)
        return datetime.datetime(*tup[:6])



class todate(converter):
    'convert to a date or None'
    def __init__(self, fmt='%Y-%m-%d', missing='Null', missingval=None):
        'use a time.strptime format string for conversion'
        converter.__init__(self, missing, missingval)
        self.fmt = fmt
    def __call__(self, s):
        if self.is_missing(s): return self.missingval
        tup = time.strptime(s, self.fmt)
        return datetime.date(*tup[:3])

class tofloat(converter):
    'convert to a float or None'
    def __init__(self, missing='Null', missingval=None):
        converter.__init__(self, missing)
        self.missingval = missingval
    def __call__(self, s):
        if self.is_missing(s): return self.missingval
        return float(s)


class toint(converter):
    'convert to an int or None'
    def __init__(self, missing='Null', missingval=None):
        converter.__init__(self, missing)

    def __call__(self, s):
        if self.is_missing(s): return self.missingval
        return int(s)

class CallbackRegistry:
    """
    Handle registering and disconnecting for a set of signals and
    callbacks

       signals = 'eat', 'drink', 'be merry'

       def oneat(x):
           print 'eat', x

       def ondrink(x):
           print 'drink', x

       callbacks = CallbackRegistry(signals)

       ideat = callbacks.connect('eat', oneat)
       iddrink = callbacks.connect('drink', ondrink)

       #tmp = callbacks.connect('drunk', ondrink) # this will raise a ValueError

       callbacks.process('drink', 123)    # will call oneat
       callbacks.process('eat', 456)      # will call ondrink
       callbacks.process('be merry', 456) # nothing will be called
       callbacks.disconnect(ideat)        # disconnect oneat
       callbacks.process('eat', 456)      # nothing will be called

    """
    def __init__(self, signals):
        'signals is a sequence of valid signals'
        self.signals = set(signals)
        # callbacks is a dict mapping the signal to a dictionary
        # mapping callback id to the callback function
        self.callbacks = dict([(s, dict()) for s in signals])
        self._cid = 0

    def _check_signal(self, s):
        'make sure s is a valid signal or raise a ValueError'
        if s not in self.signals:
            signals = list(self.signals)
            signals.sort()
            raise ValueError('Unknown signal "%s"; valid signals are %s'%(s, signals))

    def connect(self, s, func):
        """
        register func to be called when a signal s is generated
        func will be called
        """
        self._check_signal(s)
        self._cid +=1
        self.callbacks[s][self._cid] = func
        return self._cid

    def disconnect(self, cid):
        """
        disconnect the callback registered with callback id cid
        """
        for eventname, callbackd in self.callbacks.items():
            try: del callbackd[cid]
            except KeyError: continue
            else: return

    def process(self, s, *args, **kwargs):
        """
        process signal s.  All of the functions registered to receive
        callbacks on s will be called with *args and **kwargs
        """
        self._check_signal(s)
        for func in self.callbacks[s].values():
            func(*args, **kwargs)



class silent_list(list):
    """
    override repr when returning a list of matplotlib artists to
    prevent long, meaningless output.  This is meant to be used for a
    homogeneous list of a give type
    """
    def __init__(self, type, seq=None):
        self.type = type
        if seq is not None: self.extend(seq)

    def __repr__(self):
        return '<a list of %d %s objects>' % (len(self), self.type)

    def __str__(self):
        return '<a list of %d %s objects>' % (len(self), self.type)

def strip_math(s):
    'remove latex formatting from mathtext'
    remove = (r'\rm', '\cal', '\tt', '\it', '\\', '{', '}')
    s = s[1:-1]
    for r in remove:  s = s.replace(r,'')
    return s

class Bunch:
   """
   Often we want to just collect a bunch of stuff together, naming each
   item of the bunch; a dictionary's OK for that, but a small do- nothing
   class is even handier, and prettier to use.  Whenever you want to
   group a few variables:

     >>> point = Bunch(datum=2, squared=4, coord=12)
     >>> point.datum

     By: Alex Martelli
     From: http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52308
   """
   def __init__(self, **kwds):
      self.__dict__.update(kwds)


def unique(x):
   'Return a list of unique elements of x'
   return dict([ (val, 1) for val in x]).keys()

def iterable(obj):
    try: len(obj)
    except: return 0
    return 1


def is_string_like(obj):
    if hasattr(obj, 'shape'): return 0
    try: obj + ''
    except (TypeError, ValueError): return 0
    return 1

def is_writable_file_like(obj):
    return hasattr(obj, 'write') and callable(obj.write)

def is_scalar(obj):
    return is_string_like(obj) or not iterable(obj)

def is_numlike(obj):
    try: obj+1
    except TypeError: return False
    else: return True

def to_filehandle(fname, flag='r', return_opened=False):
    """
    fname can be a filename or a file handle.  Support for gzipped
    files is automatic, if the filename ends in .gz.  flag is a
    read/write flag for file
    """
    if is_string_like(fname):
        if fname.endswith('.gz'):
            import gzip
            fh = gzip.open(fname, flag)
        else:
            fh = file(fname, flag)
        opened = True
    elif hasattr(fname, 'seek'):
        fh = fname
        opened = False
    else:
        raise ValueError('fname must be a string or file handle')
    if return_opened:
        return fh, opened
    return fh

def flatten(seq, scalarp=is_scalar):
    """
    this generator flattens nested containers such as

    >>> l=( ('John', 'Hunter'), (1,23), [[[[42,(5,23)]]]])

    so that

    >>> for i in flatten(l): print i,
    John Hunter 1 23 42 5 23

    By: Composite of Holger Krekel and Luther Blissett
    From: http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/121294
    and Recipe 1.12 in cookbook
    """
    for item in seq:
        if scalarp(item): yield item
        else:
            for subitem in flatten(item, scalarp):
               yield subitem



class Sorter:
   """

   Sort by attribute or item

   Example usage:
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
      if inplace: data[:] = result
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
      aux = [(getattr(data[i],attributename),i) for i in range(len(data))]
      return self._helper(data, aux, inplace)

   # a couple of handy synonyms
   sort = byItem
   __call__ = byItem





class Xlator(dict):
    """
    All-in-one multiple-string-substitution class

    Example usage:

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
        return re.compile("|".join(map(re.escape, self.keys())))

    def __call__(self, match):
        """ Handler invoked for each regex match """
        return self[match.group(0)]

    def xlat(self, text):
        """ Translate text, returns the modified text. """
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
            if not fc: fc = c   # Remember first letter
            d = soundex_digits[ord(c)-ord('A')]
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

    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return self
    def __str__(self): return "Null()"
    def __repr__(self): return "Null()"
    def __nonzero__(self): return 0

    def __getattr__(self, name): return self
    def __setattr__(self, name, value): return self
    def __delattr__(self, name): return self




def mkdirs(newdir, mode=0777):
   try: os.makedirs(newdir, mode)
   except OSError, err:
      # Reraise the error unless it's about an already existing directory
      if err.errno != errno.EEXIST or not os.path.isdir(newdir):
         raise


class GetRealpathAndStat:
    def __init__(self):
        self._cache = {}

    def __call__(self, path):
        result = self._cache.get(path)
        if result is None:
            realpath = os.path.realpath(path)
            stat = os.stat(realpath)
            stat_key = (stat.st_ino, stat.st_dev)
            result = realpath, stat_key
            self._cache[path] = result
        return result
get_realpath_and_stat = GetRealpathAndStat()

def dict_delall(d, keys):
    'delete all of the keys from the dict d'
    for key in keys:
        try: del d[key]
        except KeyError: pass


class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self,size_max):
        self.max = size_max
        self.data = []

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.cur] = x
            self.cur = (self.cur+1) % self.max
        def get(self):
            """ return list of elements in correct order """
            return self.data[self.cur:]+self.data[:self.cur]

    def append(self,x):
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


# use enumerate builtin if available, else use python version
try:
    import __builtin__
    enumerate = __builtin__.enumerate
except:
    def enumerate(seq):
        """Python equivalent to the enumerate builtin function
        enumerate() is new in Python 2.3
        """
        for i in range(len(seq)):
            yield i, seq[i]


# use reversed builtin if available, else use python version
try:
    import __builtin__
    reversed = __builtin__.reversed
except:
    def reversed(seq):
        """Python equivalent to the enumerate builtin function
        enumerate() is new in Python 2.3
        """
        for i in range(len(seq)-1,-1,-1):
            yield seq[i]


# use itertools.izip if available, else use python version
try:
    import itertools
    izip = itertools.izip
except:
    def izip(*iterables):
        """Python equivalent to itertools.izip
        itertools module - new in Python 2.3
        """
        iterables = map(iter, iterables)
        while iterables:
            result = [i.next() for i in iterables]
            yield tuple(result)


def get_split_ind(seq, N):
   """seq is a list of words.  Return the index into seq such that
   len(' '.join(seq[:ind])<=N
   """

   sLen = 0
   # todo: use Alex's xrange pattern from the cbook for efficiency
   for (word, ind) in zip(seq, range(len(seq))):
      sLen += len(word) + 1  # +1 to account for the len(' ')
      if sLen>=N: return ind
   return len(seq)


def wrap(prefix, text, cols):
    'wrap text with prefix at length cols'
    pad = ' '*len(prefix.expandtabs())
    available = cols - len(pad)

    seq = text.split(' ')
    Nseq = len(seq)
    ind = 0
    lines = []
    while ind<Nseq:
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
    Remove excess indentation from docstrings.

    Discards any leading blank lines, then removes up to
    n whitespace characters from each line, where n is
    the number of leading whitespace characters in the
    first line. It differs from textwrap.dedent in its
    deletion of leading blank lines and its use of the
    first non-blank line to determine the indentation.

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
    import os.path, fnmatch
    # Expand patterns from semicolon-separated string to list
    pattern_list = patterns.split(';')
    # Collect input and output arguments into one bunch
    class Bunch:
        def __init__(self, **kwds): self.__dict__.update(kwds)
    arg = Bunch(recurse=recurse, pattern_list=pattern_list,
        return_folders=return_folders, results=[])

    def visit(arg, dirname, files):
        # Append to arg.results all relevant files (and perhaps folders)
        for name in files:
            fullname = os.path.normpath(os.path.join(dirname, name))
            if arg.return_folders or os.path.isfile(fullname):
                for pattern in arg.pattern_list:
                    if fnmatch.fnmatch(name, pattern):
                        arg.results.append(fullname)
                        break
        # Block recursion if recursion was disallowed
        if not arg.recurse: files[:]=[]

    os.path.walk(root, visit, arg)

    return arg.results

def get_recursive_filelist(args):
    """
    Recurs all the files and dirs in args ignoring symbolic links and
    return the files as a list of strings
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
   "Break up the seq into num tuples"
   start = 0
   while 1:
      item = seq[start:start+num]
      if not len(item): break
      yield item
      start += num

def exception_to_str(s = None):

   sh = StringIO.StringIO()
   if s is not None: print >>sh, s
   traceback.print_exc(file=sh)
   return sh.getvalue()


def allequal(seq):
    """
    return true if all elements of seq compare equal.  If seq is 0 or
    1 length, return True
    """
    if len(seq)<2: return True
    val = seq[0]
    for i in xrange(1, len(seq)):
        thisval = seq[i]
        if thisval != val: return False
    return True

def alltrue(seq):
    #return true if all elements of seq are true.  If seq is empty return false
    if not len(seq): return False
    for val in seq:
        if not val: return False
    return True

def onetrue(seq):
    #return true if one element of seq is true.  If seq is empty return false
    if not len(seq): return False
    for val in seq:
        if val: return True
    return False

def allpairs(x):
    """
    return all possible pairs in sequence x

    Condensed by Alex Martelli from this thread on c.l.python
    http://groups.google.com/groups?q=all+pairs+group:*python*&hl=en&lr=&ie=UTF-8&selm=mailman.4028.1096403649.5135.python-list%40python.org&rnum=1
    """
    return [ (s, f) for i, f in enumerate(x) for s in x[i+1:] ]




# python 2.2 dicts don't have pop--but we don't support 2.2 any more
def popd(d, *args):
    """
    Should behave like python2.3 pop method; d is a dict

    # returns value for key and deletes item; raises a KeyError if key
    # is not in dict
    val = popd(d, key)

    # returns value for key if key exists, else default.  Delete key,
    # val item if it exists.  Will not raise a KeyError
    val = popd(d, key, default)
    """
    if len(args)==1:
        key = args[0]
        val = d[key]
        del d[key]
    elif len(args)==2:
        key, default = args
        val = d.get(key, default)
        try: del d[key]
        except KeyError: pass
    return val


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
        if len(self)>=self.maxsize:
            del self[self._killkeys[0]]
            del self._killkeys[0]
        dict.__setitem__(self, k, v)
        self._killkeys.append(k)



class Stack:
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
        if not len(self._elements): return self._default
        else: return self._elements[self._pos]

    def forward(self):
        'move the position forward and return the current element'
        N = len(self._elements)
        if self._pos<N-1: self._pos += 1
        return self()

    def back(self):
        'move the position back and return the current element'
        if self._pos>0: self._pos -= 1
        return self()

    def push(self, o):
        """
        push object onto stack at current position - all elements
        occurring later than the current position are discarded
        """
        self._elements = self._elements[:self._pos+1]
        self._elements.append(o)
        self._pos = len(self._elements)-1
        return self()

    def home(self):
        'push the first element onto the top of the stack'
        if not len(self._elements): return
        self.push(self._elements[0])
        return self()

    def empty(self):
        return len(self._elements)==0

    def clear(self):
        'empty the stack'
        self._pos = -1
        self._elements = []

    def bubble(self, o):
        """
        raise o to the top of the stack and return o.  o must be in
        the stack
        """

        if o not in self._elements:
            raise ValueError('Unknown element o')
        old = self._elements[:]
        self.clear()
        bubbles = []
        for thiso in old:
            if thiso==o: bubbles.append(thiso)
            else: self.push(thiso)
        for thiso in bubbles:
            self.push(o)
        return o

    def remove(self, o):
        'remove element o from the stack'
        if o not in self._elements:
            raise ValueError('Unknown element o')
        old = self._elements[:]
        self.clear()
        for thiso in old:
            if thiso==o: continue
            else: self.push(thiso)

def popall(seq):
    'empty a list'
    for i in xrange(len(seq)): seq.pop()

def finddir(o, match, case=False):
    """
    return all attributes of o which match string in match.  if case
    is True require an exact case match.
    """
    if case:
        names = [(name,name) for name in dir(o) if is_string_like(name)]
    else:
        names = [(name.lower(), name) for name in dir(o) if is_string_like(name)]
        match = match.lower()
    return [orig for name, orig in names if name.find(match)>=0]

def reverse_dict(d):
    'reverse the dictionary -- may lose data if values are not uniq!'
    return dict([(v,k) for k,v in d.items()])


def report_memory(i=0):  # argument may go away
    'return the memory consumed by process'
    pid = os.getpid()
    if sys.platform=='sunos5':
        a2 = os.popen('ps -p %d -o osz' % pid).readlines()
        mem = int(a2[-1].strip())
    elif sys.platform.startswith('linux'):
        a2 = os.popen('ps -p %d -o rss,sz' % pid).readlines()
        mem = int(a2[1].split()[1])
    elif sys.platform.startswith('darwin'):
        a2 = os.popen('ps -p %d -o rss,vsz' % pid).readlines()
        mem = int(a2[1].split()[0])

    return mem

_safezip_msg = 'In safezip, len(args[0])=%d but len(args[%d])=%d'
def safezip(*args):
    'make sure args are equal len before zipping'
    Nx = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != Nx:
            raise ValueError(_safezip_msg % (Nx, i+1, len(arg)))
    return zip(*args)


class MemoryMonitor:
    def __init__(self, nmax=20000):
        self._nmax = nmax
        self._mem = npy.zeros((self._nmax,), npy.int32)
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
        dn = int(n/segments)
        ii = range(0, n, dn)
        ii[-1] = n-1
        print
        print 'memory report: i, mem, dmem, dmem/nloops'
        print 0, self._mem[0]
        for i in range(1, len(ii)):
            di = ii[i] - ii[i-1]
            if di == 0:
                continue
            dm = self._mem[ii[i]] - self._mem[ii[i-1]]
            print '%5d %5d %3d %8.3f' % (ii[i], self._mem[ii[i]],
                                            dm, dm / float(di))
        if self._overflow:
            print "Warning: array size was too small for the number of calls."

    def xy(self, i0=0, isub=1):
        x = npy.arange(i0, self._n, isub)
        return x, self._mem[i0:self._n:isub]

    def plot(self, i0=0, isub=1, fig=None):
        if fig is None:
            from pylab import figure, show
            fig = figure()

        ax = fig.add_subplot(111)
        ax.plot(*self.xy(i0, isub))
        fig.canvas.draw()


def print_cycles(objects, outstream=sys.stdout, show_progress=False):
    """
    objects:       A list of objects to find cycles in.  It is often useful
                   to pass in gc.garbage to find the cycles that are
                   preventing some objects from being garbage collected.
    outstream:     The stream for output.
    show_progress: If True, print the number of objects reached as they are
                   found.
    """
    import gc
    from types import FrameType

    def print_path(path):
        for i, step in enumerate(path):
            # next "wraps around"
            next = path[(i + 1) % len(path)]

            outstream.write("   %s -- " % str(type(step)))
            if isinstance(step, dict):
                for key, val in step.items():
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
        recurse(obj, obj, { }, [])





if __name__=='__main__':
    assert( allequal([1,1,1]) )
    assert(not  allequal([1,1,0]) )
    assert( allequal([]) )
    assert( allequal(('a', 'a')))
    assert( not allequal(('a', 'b')))
