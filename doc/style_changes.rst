Matplotlib v2.0
===============

Changing the default colors and style of matplotlib has proven to be a
much larger task than initially thought, but we are finally
approaching the release.

For the full details of what has been changed see a `draft of the
release notes
<http://matplotlib.org/2.0.0rc1/users/dflt_style_changes.html>`_.

The current pre-release is
`v2.0.0rc1 <https://github.com/matplotlib/matplotlib/releases/tag/v2.0.0rc1>`_

You can install pre-releases via ::

  pip install --pre matplotlib

which has source + wheels for Mac, Win, and manylinux or

using ::

  conda install -c conda-forge/label/rc -c conda-forge matplotlib

which has binaries for Mac, Win, and linux.  You can also install from
`source
<http://matplotlib.org/users/installing.html#installing-from-source>` from
git ::

  git clone https://github.com/matplotlib/matplotlib.git
  cd matplotlib
  git checkout v2.0.0rc1

or tarball ::

  wget https://github.com/matplotlib/matplotlib/archive/v2.0.0rc1.tar.gz -O matplotlib-v2.0.0rc1.tar.gz
  tar -xzvf matplotlib-v2.0.0rc1.tar.gz
  cd matplotlib-v2.0.0rc1

via ::

  pip install -v .
