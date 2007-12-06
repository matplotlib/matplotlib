"""Configuration utilities for matplotlib
"""


# Stdlib imports
import os
import pytz
import sys
import tempfile
import warnings

# matplotlib imports
from verbose import verbose
from rcsetup import defaultParams

def is_string_like(obj):
    try: obj + ''
    except (TypeError, ValueError): return 0
    return 1

def is_writable_dir(p):
    """
    p is a string pointing to a putative writable dir -- return True p
    is such a string, else False
    """
    try: p + ''  # test is string like
    except TypeError: return False
    try:
        t = tempfile.TemporaryFile(dir=p)
        t.write('1')
        t.close()
    except OSError: return False
    else: return True

def _get_home():
    """Find user's home directory if possible.
    Otherwise raise error.

    :see:  http://mail.python.org/pipermail/python-list/2005-February/263921.html
    """
    path=''
    try:
        path=os.path.expanduser("~")
    except:
        pass
    if not os.path.isdir(path):
        for evar in ('HOME', 'USERPROFILE', 'TMP'):
            try:
                path = os.environ[evar]
                if os.path.isdir(path):
                    break
            except: pass
    if path:
        return path
    else:
        raise RuntimeError('please define environment variable $HOME')

get_home = verbose.wrap('$HOME=%s', _get_home, always=False)

def _get_configdir():
    """
    Return the string representing the configuration dir.

    default is HOME/.matplotlib.  you can override this with the
    MPLCONFIGDIR environment variable
    """

    configdir = os.environ.get('MPLCONFIGDIR')
    if configdir is not None:
        if not is_writable_dir(configdir):
            raise RuntimeError('Could not write to MPLCONFIGDIR="%s"'%configdir)
        return configdir

    h = get_home()
    p = os.path.join(get_home(), '.matplotlib')

    if os.path.exists(p):
        if not is_writable_dir(p):
            raise RuntimeError("""\
'%s' is not a writable dir; you must set %s/.matplotlib to be a writable dir.
You can also set environment variable MPLCONFIGDIR to any writable directory
where you want matplotlib data stored """%h)
    else:
        if not is_writable_dir(h):
            raise RuntimeError("Failed to create %s/.matplotlib; consider setting MPLCONFIGDIR to a writable directory for matplotlib configuration data"%h)

        os.mkdir(p)

    return p

get_configdir = verbose.wrap('CONFIGDIR=%s', _get_configdir, always=False)

def _get_data_path():
    'get the path to matplotlib data'

    if os.environ.has_key('MATPLOTLIBDATA'):
        path = os.environ['MATPLOTLIBDATA']
        if not os.path.isdir(path):
            raise RuntimeError('Path in environment MATPLOTLIBDATA not a directory')
        return path

    path = os.sep.join([os.path.dirname(__file__), 'mpl-data'])
    if os.path.isdir(path): return path

    # setuptools' namespace_packages may highjack this init file
    # so need to try something known to be in matplotlib, not basemap
    import matplotlib.afm
    path = os.sep.join([os.path.dirname(matplotlib.afm.__file__), 'mpl-data'])
    if os.path.isdir(path): return path

    # py2exe zips pure python, so still need special check
    if getattr(sys,'frozen',None):
        path = os.path.join(os.path.split(sys.path[0])[0], 'mpl-data')
        if os.path.isdir(path): return path
        else:
            # Try again assuming we need to step up one more directory
            path = os.path.join(os.path.split(os.path.split(sys.path[0])[0])[0],
                                'mpl-data')
        if os.path.isdir(path): return path
        else:
            # Try again assuming sys.path[0] is a dir not a exe
            path = os.path.join(sys.path[0], 'mpl-data')
            if os.path.isdir(path): return path

    raise RuntimeError('Could not find the matplotlib data files')

def _get_data_path_cached():
    if defaultParams['datapath'][0] is None:
        defaultParams['datapath'][0] = _get_data_path()
    return defaultParams['datapath'][0]

get_data_path = verbose.wrap('matplotlib data path %s', _get_data_path_cached, always=False)

def get_py2exe_datafiles():
    datapath = get_data_path()
    head, tail = os.path.split(datapath)
    d = {}
    for root, dirs, files in os.walk(datapath):
        # Need to explicitly remove cocoa_agg files or py2exe complains
        # NOTE I dont know why, but do as previous version
        if 'Matplotlib.nib' in files:
            files.remove('Matplotlib.nib')
        files = [os.path.join(root, filename) for filename in files]
        root = root.replace(tail, 'mpl-data')
        root = root[root.index('mpl-data'):]
        d[root] = files
    return d.items()

# changed name from matplotlib_fname:
def get_config_file(tconfig=False):
    """
    Return the path to the rc file

    Search order:

     * current working dir
     * environ var MATPLOTLIBRC
     * HOME/.matplotlib/matplotlibrc
     * MATPLOTLIBDATA/matplotlibrc

    """
    if tconfig:
        filename = 'matplotlib.conf'
    else:
        filename = 'matplotlibrc'

    fname = os.path.join( os.getcwd(), filename)
    if os.path.exists(fname): return fname

    if os.environ.has_key('MATPLOTLIBRC'):
        path =  os.environ['MATPLOTLIBRC']
        if os.path.exists(path):
            fname = os.path.join(path, filename)
            if os.path.exists(fname):
                return fname

    fname = os.path.join(get_configdir(), filename)
    if os.path.exists(fname): return fname


    path =  get_data_path() # guaranteed to exist or raise
    fname = os.path.join(path, filename)
    if not os.path.exists(fname):
        warnings.warn('Could not find %s; using defaults'%filename)
    return fname
