#!/usr/bin/env python
#
# This allows running the matplotlib tests from the command line: e.g.
# python tests.py -v -d
# See http://somethingaboutorange.com/mrl/projects/nose/1.0.0/usage.html
# for options.

import os
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

def run():
    nose.main(addplugins=[KnownFailure()],
              defaultTest=default_test_modules)

if __name__ == '__main__':
    run()
