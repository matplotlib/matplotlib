#!/usr/bin/env python
#
# This allows running the matplotlib tests from the command line: e.g.
# python tests.py -v -d
# See http://somethingaboutorange.com/mrl/projects/nose/1.0.0/usage.html
# for options.

import matplotlib
matplotlib.use('agg')

import nose
from matplotlib.testing.noseclasses import KnownFailure
from matplotlib import default_test_modules

def run():
    nose.main(addplugins=[KnownFailure()],
              defaultTest=default_test_modules)

if __name__ == '__main__':
    run()
