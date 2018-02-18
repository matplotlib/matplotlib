#! /bin/bash

set -ex

# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# The number of processes is hardcoded, because using too many causes the
# Travis VM to run out of memory (since so many copies of inkscape and
# ghostscript are running at the same time).

if [[ $DELETE_FONT_CACHE == 1 ]]; then
  rm -rf ~/.cache/matplotlib
fi

echo The following args are passed to pytest $PYTEST_ARGS $RUN_PEP8

python -mpytest $PYTEST_ARGS $RUN_PEP8
