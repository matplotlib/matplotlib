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

# import nose
import pytest
from matplotlib import default_test_modules


def run(extra_args):
    matplotlib._init_tests()

    argv = sys.argv + extra_args
    # pytest.main(['--pyargs', '--cov=matplotlib'] + default_test_modules)
    print(argv + ['--pyargs'] + default_test_modules)
    return pytest.main(argv + ['--pyargs'] + default_test_modules)

if __name__ == '__main__':
    # extra_args = ['--cov=matplotlib']
    extra_args = []

    if '--pep8' in sys.argv:
        extra_args.extend(['--pep8'])
        extra_args.extend(['-m pep8'])
        sys.argv.remove('--pep8')
    if '--no-network' in sys.argv:
        from matplotlib.testing import disable_internet
        disable_internet.turn_off_internet()
        extra_args.extend(['--eval-attr="not network"'])
        sys.argv.remove('--no-network')

    returnvar = run(extra_args)
    sys.exit(returnvar)
