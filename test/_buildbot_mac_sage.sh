#!/bin/bash
set -e
export PYTHON=/Users/jdh2358/dev/bin/python
export PREFIX=/Users/jdh2358/devbb 
export PYTHONPATH=${PREFIX}/lib/python2.6/site-packages:/Users/jdh2358/dev/lib/python2.6/site-packages

make -f make.osx mpl_install
echo ${PYTHONPATH}

cd test && python run-mpl-test.py --all --keep-failed