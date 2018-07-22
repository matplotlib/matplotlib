Running ``python setup.py bdist_wheel`` (*not* ``pip wheel .``) generates a
wheel for a ``matplotlib.tests`` distribution (in the distutils sense, i.e.
a PyPI package) that can be installed alongside a main, test-less Matplotlib
distribution.

Note that

- ``pip wheel`` doesn't work as that starts by copying the *current* directory
  to a temporary one (for isolation purposes), before we get to ``chdir`` back
  to the root directory.
- ``python setup.py sdist`` doesn't work as that would pick up the
  ``MANIFEST.in`` file in the root directory.
