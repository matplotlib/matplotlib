#!/bin/bash
set -e

TARGET=`pwd`/PYmpl
TARGET_py=$TARGET/bin/python

$TARGET_py -c "import shutil,matplotlib; x=matplotlib.get_configdir(); shutil.rmtree(x)"

TARGET_easy_install=$TARGET/bin/easy_install

$TARGET_easy_install sphinx

cd doc

$TARGET_py make.py all
