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
.. _pytest-flake8: https://pypi.org/project/pytest-flake8/
.. _pytest-xdist: https://pypi.org/project/pytest-xdist/
.. _pytest-timeout: https://pypi.org/project/pytest-timeout/
.. _flake8: https://pypi.org/project/flake8/

Requirements
------------

Install the latest version of Matplotlib as documented in
:ref:`installing_for_devs`.

The following software is required to run the tests:

- pytest_ (>=3.6)
- Ghostscript_ (>= 9.0, to render PDF files)
- Inkscape_ (<1.0, to render SVG files)

Optionally you can install:

- pytest-cov_ (>=2.3.1) to collect coverage information
- pytest-flake8_ to test coding standards using flake8_
- pytest-timeout_ to limit runtime in case of stuck tests
- pytest-xdist_ to run tests in parallel


Running the tests
-----------------

Running the tests is simple. Make sure you have pytest installed and run::

   pytest

in the root directory of the repository.

pytest can be configured via a lot of `command-line parameters`_. Some
particularly useful ones are:

=============================  ===========
``-v`` or ``--verbose``        Be more verbose
``-n NUM``                     Run tests in parallel over NUM
                               processes (requires pytest-xdist_)
``--timeout=SECONDS``          Set timeout for results from each test
                               process (requires pytest-timeout_)
``--capture=no`` or ``-s``     Do not capture stdout
``--flake8``                   Check coding standards using flake8_
                               (requires pytest-flake8_)
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

An alternative implementation that does not look at command line arguments
and works from within Python is to run the tests from the Matplotlib library
function :func:`matplotlib.test`::

  import matplotlib
  matplotlib.test()


.. _command-line parameters: http://doc.pytest.org/en/latest/usage.html


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
execution (such as created figures or modified `.rcParams`). The pytest fixture
:func:`~matplotlib.testing.conftest.mpl_test_settings` will automatically clean
these up; there is no need to do anything further.

Random data in tests
--------------------

Random data is a very convenient way to generate data for examples,
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

Writing an image-based test is only slightly more difficult than a simple
test. The main consideration is that you must specify the "baseline", or
expected, images in the `~matplotlib.testing.decorators.image_comparison`
decorator. For example, this test generates a single image and automatically
tests it::

   from matplotlib.testing.decorators import image_comparison
   import matplotlib.pyplot as plt

   @image_comparison(baseline_images=['line_dashes'], remove_text=True,
                     extensions=['png'])
   def test_line_dashes():
       fig, ax = plt.subplots()
       ax.plot(range(10), linestyle=(0, (3, 3)), lw=5)

The first time this test is run, there will be no baseline image to compare
against, so the test will fail.  Copy the output images (in this case
:file:`result_images/test_lines/test_line_dashes.png`) to the correct
subdirectory of :file:`baseline_images` tree in the source directory (in this
case :file:`lib/matplotlib/tests/baseline_images/test_lines`).  Put this new
file under source code revision control (with ``git add``).  When rerunning
the tests, they should now pass.

Baseline images take a lot of space in the Matplotlib repository.
An alternative approach for image comparison tests is to use the
`~matplotlib.testing.decorators.check_figures_equal` decorator, which should be
used to decorate a function taking two `.Figure` parameters and draws the same
images on the figures using two different methods (the tested method and the
baseline method).  The decorator will arrange for setting up the figures and
then collect the drawn results and compare them.

See the documentation of `~matplotlib.testing.decorators.image_comparison` and
`~matplotlib.testing.decorators.check_figures_equal` for additional information
about their use.

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

`Travis CI <https://travis-ci.com/>`_ is a hosted CI system "in the
cloud".

Travis is configured to receive notifications of new commits to GitHub
repos (via GitHub "service hooks") and to run builds or tests when it
sees these new commits. It looks for a YAML file called
``.travis.yml`` in the root of the repository to see how to test the
project.

Travis CI is already enabled for the `main Matplotlib GitHub
repository <https://github.com/matplotlib/matplotlib/>`_ -- for
example, see `its Travis page
<https://travis-ci.com/matplotlib/matplotlib>`_.

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

`Tox <https://tox.readthedocs.io/en/latest/>`_ is a tool for running tests
against multiple Python environments, including multiple versions of Python
(e.g., 3.6, 3.7) and even different Python implementations altogether
(e.g., CPython, PyPy, Jython, etc.), as long as all these versions are
available on your system's $PATH (consider using your system package manager,
e.g. apt-get, yum, or Homebrew, to install them).

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

Building old versions of Matplotlib
-----------------------------------

When running a ``git bisect`` to see which commit introduced a certain bug,
you may (rarely) need to build very old versions of Matplotlib.  The following
constraints need to be taken into account:

- Matplotlib 1.3 (or earlier) requires numpy 1.8 (or earlier).
