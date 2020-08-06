Baseline image generation
-------------------------

Earlier the baseline images were present in the ``baseline_images`` folder. Now they are present in the
:file:`sub-wheels/matpotlib-baseline-images/lib/matplotlib_baseline_images`.

In order to run local image tests, the developer should checkout the ``master`` branch and run the tests once with the
``generate_images`` flag in order to get a set of baseline images that are consistent with the code in ``master``. The
command will be ::

    python3 -mpytest --generate_images

Then subsequent tests from their branches will test against these baseline images.

Alternately, the developer could download the ``matplotlib-baseline-images`` package and compare against these images.
However, the developer will have to be certain to have development environment that has the same free type and other
dependencies as the testing environment on the the machine that made ``matplotlib-baseline-images``. Developer can
install the ``matplotlib-baseline-images`` package by running the command ::

    python3 -mpip install  -ve  sub-wheels/matplotlib-baseline-images

or can install the package from PyP by the command ::

    python3 -mpip install matplotlib-baseline-images

