#!/usr/bin/env python
#
# This allows running the matplotlib tests from the command line: e.g.
#
#   $ python tests.py -v -d
#
# The arguments are identical to the arguments accepted by nosetests.
#
# See https://nose.readthedocs.org/ for a detailed description of
# these options.

import os
import sys
import time

import matplotlib
matplotlib.use('agg')

import nose
from matplotlib.testing.noseclasses import KnownFailure
from matplotlib import default_test_modules

from matplotlib import font_manager
# Make sure the font caches are created before starting any possibly
# parallel tests
if font_manager._fmcache is not None:
    while not os.path.exists(font_manager._fmcache):
        time.sleep(0.5)

plugins = [KnownFailure]

# Nose doesn't automatically instantiate all of the plugins in the
# child processes, so we have to provide the multiprocess plugin with
# a list.
from nose.plugins import multiprocess
multiprocess._instantiate_plugins = plugins


def run():
    try:
        import faulthandler
    except ImportError:
        pass
    else:
        faulthandler.enable()

    nose.main(addplugins=[x() for x in plugins],
              defaultTest=default_test_modules)

if __name__ == '__main__':
    if '--no-pep8' in sys.argv:
        default_test_modules.remove('matplotlib.tests.test_coding_standards')
        sys.argv.remove('--no-pep8')
    elif '--pep8' in sys.argv:
        default_test_modules = ['matplotlib.tests.test_coding_standards']
        sys.argv.remove('--pep8')

    run()
