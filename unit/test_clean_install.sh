#!/bin/bash

# Tests that matplotlib can install into a completely clean virtual
# environment.

set -e
cd ..
rm -rf build
rm -rf numpy*
rm -rf python.tmp
python unit/virtualenv.py python.tmp
python.tmp/bin/python setup.py install
python.tmp/bin/python -c "import matplotlib"
rm -rf python.tmp

set -e
cd ..
rm -rf build
rm -rf numpy*
rm -rf python.tmp
python3 unit/virtualenv.py python.tmp
python.tmp/bin/python3 setup.py install
python.tmp/bin/python3 -c "import matplotlib"
rm -rf python.tmp
