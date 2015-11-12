.. _testing:

Testing
=======

Matplotlib has a testing infrastructure based on nose_, making it easy
to write new tests. The tests are in :mod:`matplotlib.tests`, and
customizations to the nose testing infrastructure are in
:mod:`matplotlib.testing`. (There is other old testing cruft around,
please ignore it while we consolidate our testing to these locations.)

.. _nose: http://nose.readthedocs.org/en/latest/

Requirements
------------

The following software is required to run the tests:

  - nose_, version 1.0 or later

  - `mock <http://www.voidspace.org.uk/python/mock/>`_, when running python
    versions < 3.3

  - `Ghostscript <http://pages.cs.wisc.edu/~ghost/>`_ (to render PDF
    files)

  - `Inkscape <http://inkscape.org>`_ (to render SVG files)

Optionally you can install:

  - `coverage <http://nedbatchelder.com/code/coverage/>`_ to collect coverage
    information

  - `pep8 <http://pep8.readthedocs.org/en/latest>`_ to test coding standards

Building matplotlib for image comparison tests
----------------------------------------------

matplotlib's test suite makes heavy use of image comparison tests,
meaning the result of a plot is compared against a known good result.
Unfortunately, different versions of FreeType produce differently
formed characters, causing these image comparisons to fail.  To make
them reproducible, matplotlib can be built with a special local copy
of FreeType.  This is recommended for all matplotlib developers.

Add the following content to a ``setup.cfg`` file at the root of the
matplotlib source directory::

  [test]
  local_freetype = True

Running the tests
-----------------

Running the tests is simple. Make sure you have nose installed and run::

   python tests.py

in the root directory of the distribution. The script takes a set of
commands, such as:

========================  ===========
``--pep8``                pep8 checks
``--no-pep8``             Do not perform pep8 checks
``--no-network``          Disable tests that require network access
========================  ===========

Additional arguments are passed on to nosetests. See the nose 
documentation for supported arguments. Some of the more important ones are given
here:

=============================  ===========
``--verbose``                  Be more verbose
``--processes=NUM``            Run tests in parallel over NUM processes
``--process-timeout=SECONDS``  Set timeout for results from test runner process
``--nocapture``                Do not capture stdout
=============================  ===========

To run a single test from the command line, you can provide a
dot-separated path to the module followed by the function separated by
a colon, e.g., (this is assuming the test is installed)::

  python tests.py matplotlib.tests.test_simplification:test_clipping

If you want to run the full test suite, but want to save wall time try
running the tests in parallel::

  python tests.py --nocapture --nose-verbose --processes=5 --process-timeout=300


An alternative implementation that does not look at command line
arguments works from within Python is to run the tests from the
matplotlib library function :func:`matplotlib.test`::

  import matplotlib
  matplotlib.test()

.. hint::

   To run the tests you need to install nose and mock if using python 2.7::

      pip install nose
      pip install mock


.. _`nosetest arguments`: http://nose.readthedocs.org/en/latest/usage.html


Writing a simple test
---------------------

Many elements of Matplotlib can be tested using standard tests. For
example, here is a test from :mod:`matplotlib.tests.test_basic`::

  from nose.tools import assert_equal

  def test_simple():
      """
      very simple example test
      """
      assert_equal(1+1,2)

Nose determines which functions are tests by searching for functions
beginning with "test" in their name.

If the test has side effects that need to be cleaned up, such as
creating figures using the pyplot interface, use the ``@cleanup``
decorator::

  from matplotlib.testing.decorators import cleanup

  @cleanup
  def test_create_figure():
      """
      very simple example test that creates a figure using pyplot.
      """
      fig = figure()
      ...


Writing an image comparison test
--------------------------------

Writing an image based test is only slightly more difficult than a
simple test. The main consideration is that you must specify the
"baseline", or expected, images in the
:func:`~matplotlib.testing.decorators.image_comparison` decorator. For
example, this test generates a single image and automatically tests
it::

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

   - `extensions`: If you only wish to test additional image formats
     (rather than just `png`), pass any additional file types in the
     list of the extensions to test.  When copying the new
     baseline files be sure to only copy the output files, not their
     conversions to ``png``.  For example only copy the files
     ending in ``pdf``, not in ``_pdf.png``.

   - `tol`: This is the image matching tolerance, the default `1e-3`.
     If some variation is expected in the image between runs, this
     value may be adjusted.

Known failing tests
-------------------

If you're writing a test, you may mark it as a known failing test with
the :func:`~matplotlib.testing.decorators.knownfailureif`
decorator. This allows the test to be added to the test suite and run
on the buildbots without causing undue alarm. For example, although
the following test will fail, it is an expected failure::

  from nose.tools import assert_equal
  from matplotlib.testing.decorators import knownfailureif

  @knownfailureif(True)
  def test_simple_fail():
      '''very simple example test that should fail'''
      assert_equal(1+1,3)

Note that the first argument to the
:func:`~matplotlib.testing.decorators.knownfailureif` decorator is a
fail condition, which can be a value such as True, False, or
'indeterminate', or may be a dynamically evaluated expression.

Creating a new module in matplotlib.tests
-----------------------------------------

We try to keep the tests categorized by the primary module they are
testing.  For example, the tests related to the ``mathtext.py`` module
are in ``test_mathtext.py``.

Let's say you've added a new module named ``whizbang.py`` and you want
to add tests for it in ``matplotlib.tests.test_whizbang``.  To add
this module to the list of default tests, append its name to
``default_test_modules`` in :file:`lib/matplotlib/__init__.py`.

Using Travis CI
---------------

`Travis CI <http://travis-ci.org/>`_ is a hosted CI system "in the
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

If you want to enable Travis CI for your personal matplotlib GitHub
repo, simply enable the repo to use Travis CI in either the Travis CI
UI or the GitHub UI (Admin | Service Hooks). For details, see `the
Travis CI Getting Started page
<http://about.travis-ci.org/docs/user/getting-started/>`_.  This
generally isn't necessary, since any pull request submitted against
the main matplotlib repository will be tested.

Once this is configured, you can see the Travis CI results at
http://travis-ci.org/your_GitHub_user_name/matplotlib -- here's `an
example <https://travis-ci.org/msabramo/matplotlib>`_.


Using tox
---------

`Tox <http://tox.testrun.org/>`_ is a tool for running tests against
multiple Python environments, including multiple versions of Python
(e.g., 2.7, 3.4, 3.5) and even different Python implementations
altogether (e.g., CPython, PyPy, Jython, etc.)

Testing all versions of Python (2.6, 2.7, 3.*) requires
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

    $ tox -e py26,py27

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
<http://tox.testrun.org/latest/config.html>`_.
