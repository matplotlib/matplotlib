#!/bin/bash
set -e
rm -rf ${HOME}/.matplotlib/*
rm -rf build

export PYTHON=${HOME}/dev/bin/python
export PREFIX=${HOME}/devbb 
export PYTHONPATH=${PREFIX}/lib/python2.6/site-packages:${HOME}/dev/lib/python2.6/site-packages

make -f make.osx mpl_install
echo ${PYTHONPATH}

cd test && python run-mpl-test.py --all --keep-failed