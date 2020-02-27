#!/usr/bin/env python
#
# This allows running the matplotlib tests from the command line: e.g.
#
#   $ python tests.py -v -d
#
# The arguments are identical to the arguments accepted by pytest.
#
# See http://doc.pytest.org/ for a detailed description of these options.

import sys
import argparse


if __name__ == '__main__':
    from matplotlib import test

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--recursionlimit', type=int, default=0,
                        help='Specify recursionlimit for test run')
    args, extra_args = parser.parse_known_args()

    print('Python byte-compilation optimization level:', sys.flags.optimize)

    retcode = test(argv=extra_args, recursionlimit=args.recursionlimit)
    sys.exit(retcode)
