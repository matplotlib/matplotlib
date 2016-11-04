#!/bin/bash
# The number of processes is hardcoded, because using too many causes the
# Travis VM to run out of memory (since so many copies of inkscape and
# ghostscript are running at the same time).

set -e

run_tests() {
    if [[ $DELETE_FONT_CACHE == 1 ]]; then
       rm -rf ~/.cache/matplotlib
    fi
    export MPL_REPO_DIR=$PWD  # needed for pep8-conformance test of the examples
    if [[ $USE_PYTEST == false ]]; then
        echo The following args are passed to nose $NOSE_ARGS
    if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
        python tests.py $NOSE_ARGS $TEST_ARGS
    else
        gdb -return-child-result -batch -ex r -ex bt --args python $PYTHON_ARGS tests.py $NOSE_ARGS $TEST_ARGS
    fi
    else
        echo The following args are passed to pytest $PYTEST_ARGS
        py.test $PYTEST_ARGS $TEST_ARGS
    fi
    }

if [[ $BUILD_DOCS == true ]]; then
    cd doc
    python make.py html -n 2
    # We don't build the LaTeX docs here, so linkchecker will complain
    touch build/html/Matplotlib.pdf
    # Linkchecker only works with python 2.7 for the time being
    deactivate
    source ~/virtualenv/python2.7/bin/activate
    pip install pip --upgrade
    # linkchecker is currently broken with requests 2.10.0 so force an earlier version
    pip install $PRE requests==2.9.2 linkchecker
    linkchecker build/html/index.html
fi


if [[ $RUN_FLAKE8 == true ]]; then
    source ci/travis/flake8_diff.sh
fi

if [[ $SKIP_TEST != true ]]; then
    echo Testing import of tkagg backend
    MPLBACKEND="tkagg" python -c 'import matplotlib.pyplot as plt; print(plt.get_backend())'
    run_tests
fi

rm -rf $HOME/.cache/matplotlib/tex.cache
rm -rf $HOME/.cache/matplotlib/test_cache
