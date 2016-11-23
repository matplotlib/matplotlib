#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# Install into our own pristine virtualenv
if [[ $TRAVIS_OS_NAME != 'osx' ]]; then
  pip install --upgrade virtualenv
  python -m virtualenv venv
  source venv/bin/activate
  export PATH=/usr/lib/ccache:$PATH
else
  rvm get head
  brew update
  brew tap homebrew/gui
  brew install python libpng  ffmpeg imagemagick mplayer ccache
  # We could install ghostscript and inkscape here to test svg and pdf
  # but this makes the test time really long.
  # brew install ghostscript inkscape
  export PATH=/usr/local/opt/ccache/libexec:$PATH
fi

# Setup environment
ccache -s

# Upgrade pip and setuptools and wheel to get as clean an install as possible
pip install --upgrade pip
pip install --upgrade wheel
pip install --upgrade setuptools

# Install dependencies from pypi
pip install $PRE python-dateutil $NUMPY pyparsing!=2.1.6 $PANDAS cycler coveralls coverage $MOCK

# Install nose from a build which has partial
# support for python36 and suport for coverage output suppressing
pip install git+https://github.com/jenshnielsen/nose.git@matplotlibnose

# pytest-cov>=2.3.1 due to https://github.com/pytest-dev/pytest-cov/issues/124
pip install $PRE pytest 'pytest-cov>=2.3.1' pytest-timeout pytest-xdist pytest-faulthandler

# We manually install humor sans using the package from Ubuntu 14.10.
# Unfortunately humor sans is not available in the Ubuntu version used by
# Travis but we can manually install the deb from a later version since is it
# basically just a .ttf file The current Travis Ubuntu image is to old to
# search .local/share/fonts so we store fonts in .fonts

if [[ $BUILD_DOCS == true ]]; then
  pip install $PRE -r doc-requirements.txt
  wget https://github.com/google/fonts/blob/master/ofl/felipa/Felipa-Regular.ttf?raw=true -O Felipa-Regular.ttf
  wget http://mirrors.kernel.org/ubuntu/pool/universe/f/fonts-humor-sans/fonts-humor-sans_1.0-1_all.deb
  mkdir -p tmp
  mkdir -p ~/.fonts
  dpkg -x fonts-humor-sans_1.0-1_all.deb tmp
  cp tmp/usr/share/fonts/truetype/humor-sans/Humor-Sans.ttf ~/.fonts
  cp Felipa-Regular.ttf ~/.fonts
  fc-cache -f -v
else
    # Use the special local version of freetype for testing
  cp ci/travis/setup.cfg .
fi;

if [[ $RUN_FLAKE8 == true ]]; then
  pip install flake8
else
  # Install matplotlib
  pip install -e .
fi;
