#!/bin/bash
set -e

TARGET=`pwd`/PYmpl
source $TARGET/bin/activate

python -c "import shutil,matplotlib; x=matplotlib.get_configdir(); shutil.rmtree(x)"

easy_install sphinx

cd doc

python make.py all
