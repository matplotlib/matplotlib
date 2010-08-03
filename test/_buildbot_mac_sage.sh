#!/bin/bash
set -e
rm -rf build

export MPLCONFIGDIR=${HOME}/.matplotlib_buildbot
export PATH=${HOME}/dev/bin:$PATH
export PYTHON=/usr/bin/python2.6
export PREFIX=${HOME}/devbb
export PYTHONPATH=${PREFIX}/lib/python2.6/site-packages:${HOME}/dev/lib/python2.6/site-packages
export LD_LIBRARY_PATH=${PREFIX}/lib
export MPLSETUPCFG=test/setup.sageosx.cfg
rm -rf ${MPLCONFIGDIR}/*
rm -rf ${PREFIX}/lib/python2.6/site-packages/matplotlib*
echo 'backend : Agg' > $MPLCONFIGDIR/matplotlibrc



make -f make.osx mpl_install
echo ${PYTHONPATH}

cd test
rm -f failed-diff-*.png
python -c "import sys, matplotlib; success = matplotlib.test(verbosity=2); sys.exit(not success)"
