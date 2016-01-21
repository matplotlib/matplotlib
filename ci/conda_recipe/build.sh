#!/bin/bash

if [ `uname` == Linux ]; then
    pushd $PREFIX/lib
    ln -s libtcl8.5.so libtcl.so
    ln -s libtk8.5.so libtk.so
    popd
fi

if [ `uname` == Darwin ]; then
    sed s:'#ifdef WITH_NEXT_FRAMEWORK':'#if 1':g -i src/_macosx.m
fi

cp setup.cfg.template setup.cfg || exit 1

sed s:/usr/local:$PREFIX:g -i setupext.py

$PYTHON setup.py install

rm -rf $SP_DIR/PySide
rm -rf $SP_DIR/__pycache__
rm -rf $PREFIX/bin/nose*

