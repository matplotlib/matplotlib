#! /bin/bash

set -v

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
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
  pytest $PYTEST_ARGS $RUN_PEP8
  exit $?
else
  python $PYTHON_ARGS -m pytest $PYTEST_ARGS $RUN_PEP8
  exit_val=$?
  echo "pytest exited with result $exit_val"
  exit $exit_val
fi
