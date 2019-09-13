import builtins
import os
import subprocess
import sys
import textwrap

import matplotlib


def test_simple():
    assert 1 + 1 == 2


def test_override_builtins():
    import pylab

    ok_to_override = {
        '__name__',
        '__doc__',
        '__package__',
        '__loader__',
        '__spec__',
        'any',
        'all',
        'sum',
        'divmod'
    }
    overridden = False
    for key in dir(pylab):
        if key in dir(builtins):
            if (getattr(pylab, key) != getattr(builtins, key) and
                    key not in ok_to_override):
                print("'%s' was overridden in globals()." % key)
                overridden = True

    assert not overridden


def test_lazy_imports():
    source = textwrap.dedent("""
    import sys

    import matplotlib.figure
    import matplotlib.backend_bases
    import matplotlib.pyplot

    assert 'matplotlib._png' not in sys.modules
    assert 'matplotlib._tri' not in sys.modules
    assert 'matplotlib._qhull' not in sys.modules
    assert 'matplotlib._contour' not in sys.modules
    assert 'urllib.request' not in sys.modules
    """)

    subprocess.check_call(
        [sys.executable, '-c', source],
        env={**os.environ, "MPLBACKEND": "", "MATPLOTLIBRC": os.devnull})
