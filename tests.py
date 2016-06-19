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
from nose.plugins import attrib
from matplotlib.testing.noseclasses import KnownFailure
from matplotlib import default_test_modules

plugins = [KnownFailure, attrib.Plugin]

# Nose doesn't automatically instantiate all of the plugins in the
# child processes, so we have to provide the multiprocess plugin with
# a list.
from nose.plugins import multiprocess
multiprocess._instantiate_plugins = plugins


def run(extra_args):
    try:
        import faulthandler
    except ImportError:
        pass
    else:
        faulthandler.enable()

    if not os.path.isdir(
            os.path.join(os.path.dirname(matplotlib.__file__), 'tests')):
        raise ImportError("matplotlib test data is not installed")

    nose.main(addplugins=[x() for x in plugins],
              defaultTest=default_test_modules,
              argv=sys.argv + extra_args)


if __name__ == '__main__':
    extra_args = []

    if '--no-pep8' in sys.argv:
        default_test_modules.remove('matplotlib.tests.test_coding_standards')
        sys.argv.remove('--no-pep8')
    elif '--pep8' in sys.argv:
        default_test_modules = ['matplotlib.tests.test_coding_standards']
        sys.argv.remove('--pep8')
    if '--no-network' in sys.argv:
        from matplotlib.testing import disable_internet
        disable_internet.turn_off_internet()
        extra_args.extend(['-a', '!network'])
        sys.argv.remove('--no-network')

    run(extra_args)
