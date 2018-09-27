.. _testing:

============================
Developer's tips for testing
============================

Matplotlib's testing infrastructure depends on pytest_. The tests are in
:file:`lib/matplotlib/tests`, and customizations to the pytest testing
infrastructure are in :mod:`matplotlib.testing`.

.. _pytest: http://doc.pytest.org/en/latest/
.. _Ghostscript: https://www.ghostscript.com/
.. _Inkscape: https://inkscape.org
.. _pytest-cov: https://pytest-cov.readthedocs.io/en/latest/
.. _pytest-pep8: https://pypi.python.org/pypi/pytest-pep8
.. _pytest-xdist: https://pypi.python.org/pypi/pytest-xdist
.. _pytest-timeout: https://pypi.python.org/pypi/pytest-timeout

Requirements
------------

Install the latest version of Matplotlib as documented in
:ref:`installing_for_devs` In particular, follow the instructions to use a
local FreeType build

The following software is required to run the tests:

- pytest_ (>=3.4)
- Ghostscript_ (>= 9.0, to render PDF files)
- Inkscape_ (to render SVG files)

Optionally you can install:

- pytest-cov_ (>=2.3.1) to collect coverage information
- pytest-pep8_ to test coding standards
- pytest-timeout_ to limit runtime in case of stuck tests
- pytest-xdist_ to run tests in parallel


Running the tests
-----------------

Running the tests is simple. Make sure you have pytest installed and run::

   pytest

or::

   pytest .


in the root directory of the distribution. The script takes a set of
commands, such as:

========================  ===========
``--pep8``                Perform pep8 checks (requires pytest-pep8_)
``-m "not network"``      Disable tests that require network access
========================  ===========

Additional arguments are passed on to pytest. See the pytest documentation for
`supported arguments`_. Some of the more important ones are given here:

=============================  ===========
``--verbose``                  Be more verbose
``--n NUM``                    Run tests in parallel over NUM
                               processes (requires pytest-xdist_)
``--timeout=SECONDS``          Set timeout for results from each test
                               process (requires pytest-timeout_)
``--capture=no`` or ``-s``     Do not capture stdout
=============================  ===========

To run a single test from the command line, you can provide a file path,
optionally followed by the function separated by two colons, e.g., (tests do
not need to be installed, but Matplotlib should be)::

  pytest lib/matplotlib/tests/test_simplification.py::test_clipping

or, if tests are installed, a dot-separated path to the module, optionally
followed by the function separated by two colons, such as::

  pytest --pyargs matplotlib.tests.test_simplification::test_clipping

If you want to run the full test suite, but want to save wall time try
running the tests in parallel::

  pytest --verbose -n 5

Depending on your version of Python and pytest-xdist, you may need to set
``PYTHONHASHSEED`` to a fixed value when running in parallel::

  PYTHONHASHSEED=0 pytest --verbose -n 5

An alternative implementation that does not look at command line arguments
and works from within Python is to run the tests from the Matplotlib library
function :func:`matplotlib.test`::

  import matplotlib
  matplotlib.test()


.. _supported arguments: http://doc.pytest.org/en/latest/usage.html


Writing a simple test
---------------------

Many elements of Matplotlib can be tested using standard tests. For
example, here is a test from :mod:`matplotlib.tests.test_basic`::

  def test_simple():
      """
      very simple example test
      """
      assert 1 + 1 == 2

Pytest determines which functions are tests by searching for files whose names
begin with ``"test_"`` and then within those files for functions beginning with
``"test"`` or classes beginning with ``"Test"``.

Some tests have internal side effects that need to be cleaned up after their
execution (such as created figures or modified rc params). The pytest fixture
:func:`~matplotlib.testing.conftest.mpl_test_settings` will automatically clean
these up; there is no need to do anything further.

Random data in tests
--------------------

Random data can is a very convenient way to generate data for examples,
however the randomness is problematic for testing (as the tests
must be deterministic!).  To work around this set the seed in each test.
For numpy use::

  import numpy as np
  np.random.seed(19680801)

and Python's random number generator::

  import random
  random.seed(19680801)

The seed is John Hunter's birthday.

Writing an image comparison test
--------------------------------

Writing an image based test is only slightly more difficult than a
simple test. The main consideration is that you must specify the
"baseline", or expected, images in the
:func:`~matplotlib.testing.decorators.image_comparison` decorator. For
example, this test generates a single image and automatically tests it::

  import numpy as np
  import matplotlib
  from matplotlib.testing.decorators import image_comparison
  import matplotlib.pyplot as plt

  @image_comparison(baseline_images=['spines_axes_positions'],
                    extensions=['png'])
  def test_spines_axes_positions():
      # SF bug 2852168
      fig = plt.figure()
      x = np.linspace(0,2*np.pi,100)
      y = 2*np.sin(x)
      ax = fig.add_subplot(1,1,1)
      ax.set_title('centered spines')
      ax.plot(x,y)
      ax.spines['right'].set_position(('axes',0.1))
      ax.yaxis.set_ticks_position('right')
      ax.spines['top'].set_position(('axes',0.25))
      ax.xaxis.set_ticks_position('top')
      ax.spines['left'].set_color('none')
      ax.spines['bottom'].set_color('none')

The first time this test is run, there will be no baseline image to
compare against, so the test will fail.  Copy the output images (in
this case `result_images/test_category/spines_axes_positions.png`) to
the correct subdirectory of `baseline_images` tree in the source
directory (in this case
`lib/matplotlib/tests/baseline_images/test_category`).  Put this new
file under source code revision control (with `git add`).  When
rerunning the tests, they should now pass.

The :func:`~matplotlib.testing.decorators.image_comparison` decorator
defaults to generating ``png``, ``pdf`` and ``svg`` output, but in
interest of keeping the size of the library from ballooning we should only
include the ``svg`` or ``pdf`` outputs if the test is explicitly exercising
a feature dependent on that backend.

There are two optional keyword arguments to the `image_comparison`
decorator:

- `extensions`: If you only wish to test additional image formats (rather than
  just `png`), pass any additional file types in the list of the extensions to
  test.  When copying the new baseline files be sure to only copy the output
  files, not their conversions to ``png``.  For example only copy the files
  ending in ``pdf``, not in ``_pdf.png``.

- `tol`: This is the image matching tolerance, the default `1e-3`. If some
  variation is expected in the image between runs, this value may be adjusted.

Known failing tests
-------------------

If you're writing a test, you may mark it as a known failing test with the
:func:`pytest.mark.xfail` decorator. This allows the test to be added to the
test suite and run on the buildbots without causing undue alarm. For example,
although the following test will fail, it is an expected failure::

  import pytest

  @pytest.mark.xfail
  def test_simple_fail():
      '''very simple example test that should fail'''
      assert 1 + 1 == 3

Note that the first argument to the :func:`~pytest.mark.xfail` decorator is a
fail condition, which can be a value such as True, False, or may be a
dynamically evaluated expression. If a condition is supplied, then a reason
must also be supplied with the ``reason='message'`` keyword argument.

Creating a new module in matplotlib.tests
-----------------------------------------

We try to keep the tests categorized by the primary module they are
testing.  For example, the tests related to the ``mathtext.py`` module
are in ``test_mathtext.py``.

Using Travis CI
---------------

`Travis CI <https://travis-ci.org/>`_ is a hosted CI system "in the
cloud".

Travis is configured to receive notifications of new commits to GitHub
repos (via GitHub "service hooks") and to run builds or tests when it
sees these new commits. It looks for a YAML file called
``.travis.yml`` in the root of the repository to see how to test the
project.

Travis CI is already enabled for the `main matplotlib GitHub
repository <https://github.com/matplotlib/matplotlib/>`_ -- for
example, see `its Travis page
<https://travis-ci.org/matplotlib/matplotlib>`_.

If you want to enable Travis CI for your personal Matplotlib GitHub
repo, simply enable the repo to use Travis CI in either the Travis CI
UI or the GitHub UI (Admin | Service Hooks). For details, see `the
Travis CI Getting Started page
<https://docs.travis-ci.com/user/getting-started/>`_.  This
generally isn't necessary, since any pull request submitted against
the main Matplotlib repository will be tested.

Once this is configured, you can see the Travis CI results at
https://travis-ci.org/your_GitHub_user_name/matplotlib -- here's `an
example <https://travis-ci.org/msabramo/matplotlib>`_.


Using tox
---------

`Tox <https://tox.readthedocs.io/en/latest/>`_ is a tool for running
tests against
multiple Python environments, including multiple versions of Python
(e.g., 3.5, 3.6) and even different Python implementations
altogether (e.g., CPython, PyPy, Jython, etc.)

Testing all versions of Python (3.5, 3.6, ...) requires
having multiple versions of Python installed on your system and on the
PATH. Depending on your operating system, you may want to use your
package manager (such as apt-get, yum or MacPorts) to do this.

tox makes it easy to determine if your working copy introduced any
regressions before submitting a pull request. Here's how to use it:

.. code-block:: bash

    $ pip install tox
    $ tox

You can also run tox on a subset of environments:

.. code-block:: bash

    $ tox -e py36,py37

Tox processes everything serially so it can take a long time to test
several environments. To speed it up, you might try using a new,
parallelized version of tox called ``detox``. Give this a try:

.. code-block:: bash

    $ pip install -U -i http://pypi.testrun.org detox
    $ detox

Tox is configured using a file called ``tox.ini``. You may need to
edit this file if you want to add new environments to test (e.g.,
``py33``) or if you want to tweak the dependencies or the way the
tests are run. For more info on the ``tox.ini`` file, see the `Tox
Configuration Specification
<https://tox.readthedocs.io/en/latest/config.html>`_.
