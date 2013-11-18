from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import shutil
import tempfile
from contextlib import contextmanager

import matplotlib as mpl
from matplotlib import style
from matplotlib.style.core import USER_LIBRARY_PATHS, STYLE_EXTENSION

import six


PARAM = 'image.cmap'
VALUE = 'pink'
DUMMY_SETTINGS = {PARAM: VALUE}


@contextmanager
def temp_style(style_name, settings=None):
    """Context manager to create a style sheet in a temporary directory."""
    settings = DUMMY_SETTINGS
    temp_file = '%s.%s' % (style_name, STYLE_EXTENSION)

    # Write style settings to file in the temp directory.
    tempdir = tempfile.mkdtemp()
    with open(os.path.join(tempdir, temp_file), 'w') as f:
        for k, v in six.iteritems(settings):
            f.write('%s: %s' % (k, v))

    # Add temp directory to style path and reload so we can access this style.
    USER_LIBRARY_PATHS.append(tempdir)
    style.reload_library()

    try:
        yield
    finally:
        shutil.rmtree(tempdir)
        style.reload_library()


def test_available():
    with temp_style('_test_', DUMMY_SETTINGS):
        assert '_test_' in style.available


def test_use():
    mpl.rcParams[PARAM] = 'gray'
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('test'):
            assert mpl.rcParams[PARAM] == VALUE


def test_use_url():
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('https://gist.github.com/adrn/6590261/raw'):
            assert mpl.rcParams['axes.facecolor'] == "#adeade"


def test_context():
    mpl.rcParams[PARAM] = 'gray'
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('test'):
            assert mpl.rcParams[PARAM] == VALUE
    # Check that this value is reset after the exiting the context.
    assert mpl.rcParams[PARAM] == 'gray'


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
