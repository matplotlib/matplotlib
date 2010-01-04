#!/bin/bash
set -e

TARGET=`pwd`/PYmpl
echo "activating virtual environment"
source $TARGET/bin/activate

echo "removing MPL config dir"
python -c "import shutil,matplotlib; x=matplotlib.get_configdir(); shutil.rmtree(x)"

echo "calling 'easy_install sphinx'"
easy_install sphinx

echo "calling 'cd doc'"
cd doc

echo "calling 'python make.py clean'"
python make.py clean

echo "calling 'python make.py all'"
python make.py all

# SourceForce needs the below
echo "configuring for upload to SourceForge"

echo "Options +Indexes" > build/html/.htaccess

chmod -R a+r build
find build -type d | xargs chmod a+rx

echo "listing built files"
find build
