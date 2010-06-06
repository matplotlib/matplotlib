# -*- coding: ascii -*-

import unittest, os, os.path, sys
from doctest import DocTestSuite

# We test the documentation this way instead of using DocFileSuite so
# we can run the tests under Python 2.3
def test_README():
    pass

this_dir = os.path.dirname(__file__)
locs = [
    os.path.join(this_dir, os.pardir, 'README.txt'),
    os.path.join(this_dir, os.pardir, os.pardir, 'README.txt'),
    ]
for loc in locs:
    if os.path.exists(loc):
        test_README.__doc__ = open(loc).read()
        break
if test_README.__doc__ is None:
    raise RuntimeError('README.txt not found')


def test_suite():
    "For the Z3 test runner"
    return DocTestSuite()


if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(
        this_dir, os.pardir, os.pardir
        )))
    unittest.main(defaultTest='test_suite')


