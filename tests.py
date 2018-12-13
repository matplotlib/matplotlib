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

    import dateutil.parser
    try:
        import setuptools
    except ImportError:
        pass

    # The warnings need to be before any of matplotlib imports, but after
    # dateutil.parser and setuptools (if present) which has syntax error with
    # the warnings enabled.  Filtering by module does not work as this will be
    # raised by Python itself so `module=matplotlib.*` is out of question.

    import warnings

    # Python 3.6 deprecate invalid character-pairs \A, \* ... in non
    # raw-strings and other things. Let's not re-introduce them
    warnings.filterwarnings('error', '.*invalid escape sequence.*',
        category=DeprecationWarning)
    warnings.filterwarnings(
        'default',
        r'.*inspect.getargspec\(\) is deprecated.*',
        category=DeprecationWarning)

    from matplotlib import test

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--no-network', action='store_true',
                        help='Run tests without network connection')
    parser.add_argument('--recursionlimit', type=int, default=0,
                        help='Specify recursionlimit for test run')
    args, extra_args = parser.parse_known_args()

    if args.no_network:
        from matplotlib.testing import disable_internet
        disable_internet.turn_off_internet()
        extra_args.extend(['-m', 'not network'])

    print('Python byte-compilation optimization level:', sys.flags.optimize)

    retcode = test(argv=extra_args, switch_backend_warn=False,
                   recursionlimit=args.recursionlimit)
    sys.exit(retcode)
