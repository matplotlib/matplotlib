"""
A collection of utility functions and classes.  Many (but not all)
from the Python Cookbook -- hence the name cbook
"""
from __future__ import generators
import re, os, errno, sys, StringIO, traceback


major, minor1, minor2, s, tmp = sys.version_info

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
    if hasattr(obj, 'shape'): return 0 # this is a workaround
                                       # for a bug in numeric<23.1
    try: obj + ''
    except (TypeError, ValueError): return 0
    return 1


def is_file_like(obj):
    if hasattr(obj, 'shape'): return 0 # this is a workaround
                                       # for a bug in numeric<23.1
    try: obj + ''
    except (TypeError, ValueError): return 0
    return 1

def is_scalar(obj):
    return is_string_like(obj) or not iterable(obj)

def is_numlike(obj):
    try: obj+1
    except TypeError: return False
    else: return True

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

def dedent(s):
    """
    Remove excess indentation from docstrings.

    Discards any leading blank lines, then removes up to
    n whitespace characters from each line, where n is
    the number of leading whitespace characters in the
    first line. It differs from textwrap.dedent in its
    deletion of leading blank lines and its use of the
    first non-blank line to determine the indentation.
    """
    lines = s.splitlines(True)
    ii = 0
    while lines[ii].strip() == '':
        ii += 1
    lines = lines[ii:]
    nshift = len(lines[0]) - len(lines[0].lstrip())
    for i, line in enumerate(lines):
        nwhite = len(line) - len(line.lstrip())
        lines[i] = line[min(nshift, nwhite):]
    return ''.join(lines)



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




# python 2.2 dicts don't have pop
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


if __name__=='__main__':
    assert( allequal([1,1,1]) )
    assert(not  allequal([1,1,0]) )
    assert( allequal([]) )
    assert( allequal(('a', 'a')))
    assert( not allequal(('a', 'b')))
