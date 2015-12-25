#!/bin/bash

if [ `uname` == Linux ]; then
    pushd $PREFIX/lib
    ln -s libtcl8.5.so libtcl.so
    ln -s libtk8.5.so libtk.so
    popd
fi

if [ `uname` == Darwin ]; then
    # run tests with Agg...
    # prevents a problem with Macosx mpl not installed as framework
    export MPLBACKEND=Agg
    # This seems to be not anymore needed...
    #sed s:'#ifdef WITH_NEXT_FRAMEWORK':'#if 1':g -i src/_macosx.m
fi

cp setup.cfg.template setup.cfg || exit 1

# on mac there is an error if done inplace:
#   sed: -i: No such file or directory
# travis macosx sed has not even --help...
mv setupext.py setupext.py_orig
cat setupext.py_orig | sed s:/usr/local:$PREFIX:g > setupext.py

$PYTHON setup.py install

rm -rf $SP_DIR/PySide
rm -rf $SP_DIR/__pycache__
rm -rf $PREFIX/bin/nose*

