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
from matplotlib import default_test_modules


def run(extra_args):
    from nose.plugins import multiprocess

    env = matplotlib._get_nose_env()

    matplotlib._init_tests()

    # Nose doesn't automatically instantiate all of the plugins in the
    # child processes, so we have to provide the multiprocess plugin with
    # a list.
    plugins = matplotlib._get_extra_test_plugins()
    multiprocess._instantiate_plugins = plugins

    nose.main(addplugins=[x() for x in plugins],
              defaultTest=default_test_modules,
              argv=sys.argv + extra_args,
              env=env)

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
        extra_args.extend(['--eval-attr="not network"'])
        sys.argv.remove('--no-network')

    run(extra_args)
