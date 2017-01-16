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
import argparse


if __name__ == '__main__':
    from matplotlib import default_test_modules, test

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--no-pep8', action='store_true',
                        help='Run all tests except PEP8 testing')
    parser.add_argument('--pep8', action='store_true',
                        help='Run only PEP8 testing')
    parser.add_argument('--no-network', action='store_true',
                        help='Run tests without network connection')
    parser.add_argument('-j', type=int,
                        help='Shortcut for specifying number of test processes')
    parser.add_argument('--recursionlimit', type=int, default=0,
                        help='Specify recursionlimit for test run')
    args, extra_args = parser.parse_known_args()

    if args.no_pep8:
        default_test_modules.remove('matplotlib.tests.test_coding_standards')
    elif args.pep8:
        default_test_modules[:] = ['matplotlib.tests.test_coding_standards']
    if args.no_network:
        from matplotlib.testing import disable_internet
        disable_internet.turn_off_internet()
        extra_args.extend(['-a', '!network'])
    if args.j:
        extra_args.extend([
            '--processes={}'.format(args.j),
            '--process-timeout=300'
        ])

    print('Python byte-compilation optimization level: %d' % sys.flags.optimize)

    success = test(argv=sys.argv[0:1] + extra_args, switch_backend_warn=False,
                   recursionlimit=args.recursionlimit)
    sys.exit(not success)
