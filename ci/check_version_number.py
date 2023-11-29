#!/usr/bin/env python3

"""
Check that the version number of the install Matplotlib does not start with 0

To run:
    $ python3 -m build .
    $ pip install dist/matplotlib*.tar.gz for sdist
    $ pip install dist/matplotlib*.whl for wheel
    $ ./ci/check_version_number.py
"""
import sys

import matplotlib


print(f"Version {matplotlib.__version__} installed")
if matplotlib.__version__[0] == "0":
    sys.exit("Version incorrectly starts with 0")
