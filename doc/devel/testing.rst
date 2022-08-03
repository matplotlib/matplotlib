.. _testing:

=======
Testing
=======

Matplotlib uses the pytest_ framework.

The tests are in :file:`lib/matplotlib/tests`, and customizations to the pytest
testing infrastructure are in :mod:`matplotlib.testing`.

.. _pytest: http://doc.pytest.org/en/latest/
.. _pytest-xdist: https://pypi.org/project/pytest-xdist/


.. _testing_requirements:

Requirements
------------

To run the tests you will need to
:ref:`set up Matplotlib for development <installing_for_devs>`. Note in
particular the :ref:`additional dependencies <test-dependencies>` for testing.

.. note::

   We will assume that you want to run the tests in a development setup.

   While you can run the tests against a regular installed version of
   Matplotlib, this is a far less common use case. You still need the
   :ref:`additional dependencies <test-dependencies>` for testing.
   You have to additionally get the reference images from the repository,
   because they are not distributed with pre-built Matplotlib packages.

Running the tests
-----------------

In the root directory of your development repository run::

   python -m pytest


pytest can be configured via a lot of `command-line parameters`_. Some
particularly useful ones are:

=============================  ===========
``-v`` or ``--verbose``        Be more verbose
``-n NUM``                     Run tests in parallel over NUM
                               processes (requires pytest-xdist_)
``--capture=no`` or ``-s``     Do not capture stdout
=============================  ===========

To run a single test from the command line, you can provide a file path,
optionally followed by the function separated by two colons, e.g., (tests do
not need to be installed, but Matplotlib should be)::

  pytest lib/matplotlib/tests/test_simplification.py::test_clipping


.. _command-line parameters: http://doc.pytest.org/en/latest/usage.html


Writing a simple test
---------------------

Many elements of Matplotlib can be tested using standard tests. For
example, here is a test from :file:`matplotlib/tests/test_basic.py`::

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
``matplotlib.testing.conftest.mpl_test_settings`` will automatically clean
these up; there is no need to do anything further.

Random data in tests
--------------------

Random data is a very convenient way to generate data for examples,
however the randomness is problematic for testing (as the tests
must be deterministic!).  To work around this set the seed in each test.
For numpy's default random number generator use::

  import numpy as np
  rng = np.random.default_rng(19680801)

and then use ``rng`` when generating the random numbers.

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

Creating a new module in matplotlib.tests
-----------------------------------------

We try to keep the tests categorized by the primary module they are
testing.  For example, the tests related to the ``mathtext.py`` module
are in ``test_mathtext.py``.

Using GitHub Actions for CI
---------------------------

`GitHub Actions <https://docs.github.com/en/actions>`_ is a hosted CI system
"in the cloud".

GitHub Actions is configured to receive notifications of new commits to GitHub
repos and to run builds or tests when it sees these new commits. It looks for a
YAML files in ``.github/workflows`` to see how to test the project.

GitHub Actions is already enabled for the `main Matplotlib GitHub repository
<https://github.com/matplotlib/matplotlib/>`_ -- for example, see `the Tests
workflows
<https://github.com/matplotlib/matplotlib/actions?query=workflow%3ATests>`_.

GitHub Actions should be automatically enabled for your personal Matplotlib
fork once the YAML workflow files are in it. It generally isn't necessary to
look at these workflows, since any pull request submitted against the main
Matplotlib repository will be tested. The Tests workflow is skipped in forked
repositories but you can trigger a run manually from the `GitHub web interface
<https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow>`_.

You can see the GitHub Actions results at
https://github.com/your_GitHub_user_name/matplotlib/actions -- here's `an
example <https://github.com/QuLogic/matplotlib/actions>`_.


Using tox
---------

`Tox <https://tox.readthedocs.io/en/latest/>`_ is a tool for running tests
against multiple Python environments, including multiple versions of Python
(e.g., 3.7, 3.8) and even different Python implementations altogether
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

    $ tox -e py38,py39

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

Testing released versions of Matplotlib
---------------------------------------
Running the tests on an installation of a released version (e.g. PyPI package
or conda package) also requires additional setup.

.. note::

   For an end-user, there is usually no need to run the tests on released
   versions of Matplotlib. Official releases are tested before publishing.

Install additional dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Install the :ref:`additional dependencies for testing <test-dependencies>`.

Obtain the reference images
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Many tests compare the plot result against reference images. The reference
images are not part of the regular packaged versions (pip wheels or conda
packages). If you want to run tests with reference images, you need to obtain
the reference images matching the version of Matplotlib you want to test.

To do so, either download the matching source distribution
``matplotlib-X.Y.Z.tar.gz`` from `PyPI <https://pypi.org/project/matplotlib/>`_
or alternatively, clone the git repository and ``git checkout vX.Y.Z``. Copy
the folder :file:`lib/matplotlib/tests/baseline_images` to the folder
:file:`matplotlib/tests` of your the matplotlib installation to test.
The correct target folder can be found using::

    python -c "import matplotlib.tests; print(matplotlib.tests.__file__.rsplit('/', 1)[0])"

An analogous copying of :file:`lib/mpl_toolkits/tests/baseline_images`
is necessary for testing ``mpl_toolkits``.

Run the tests
^^^^^^^^^^^^^
To run the all the tests on your installed version of Matplotlib::

    python -m pytest --pyargs matplotlib.tests

The test discovery scope can be narrowed to single test modules or even single
functions::

    python -m pytest --pyargs matplotlib.tests.test_simplification.py::test_clipping
