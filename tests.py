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

import sys


if __name__ == '__main__':
    from matplotlib import default_test_modules, test

    extra_args = []

    if '--no-pep8' in sys.argv:
        default_test_modules.remove('matplotlib.tests.test_coding_standards')
        sys.argv.remove('--no-pep8')
    elif '--pep8' in sys.argv:
        default_test_modules[:] = ['matplotlib.tests.test_coding_standards']
        sys.argv.remove('--pep8')
    if '--no-network' in sys.argv:
        from matplotlib.testing import disable_internet
        disable_internet.turn_off_internet()
        extra_args.extend(['-a', '!network'])
        sys.argv.remove('--no-network')
    if '-j' in sys.argv:
        nproc = sys.argv[sys.argv.index('-j') + 1]
        extra_args.extend([
            '--processes={}'.format(int(nproc)),
            '--process-timeout=300'
        ])
        sys.argv.remove('-j')
        sys.argv.remove(nproc)

    print('Python byte-compilation optimization level: %d' % sys.flags.optimize)

    success = test(argv=sys.argv + extra_args, switch_backend_warn=False)
    sys.exit(not success)
