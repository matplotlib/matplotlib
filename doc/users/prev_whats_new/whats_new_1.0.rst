.. _whats-new-1-0:

What's new in Matplotlib 1.0 (Jul 06, 2010)
===========================================

.. contents:: Table of Contents
   :depth: 2

.. _whats-new-html5:

HTML5/Canvas backend
--------------------

Simon Ratcliffe and Ludwig Schwardt have released an `HTML5/Canvas
<https://code.google.com/archive/p/mplh5canvas>`__ backend for matplotlib.  The
backend is almost feature complete, and they have done a lot of work
comparing their html5 rendered images with our core renderer Agg.  The
backend features client/server interactive navigation of matplotlib
figures in an html5 compliant browser.

Sophisticated subplot grid layout
---------------------------------

Jae-Joon Lee has written :mod:`~matplotlib.gridspec`, a new module for
doing complex subplot layouts, featuring row and column spans and
more.  See :doc:`/tutorials/intermediate/gridspec` for a tutorial overview.

.. figure:: ../../gallery/userdemo/images/sphx_glr_demo_gridspec01_001.png
   :target: ../../gallery/userdemo/demo_gridspec01.html
   :align: center
   :scale: 50

Easy pythonic subplots
-----------------------

Fernando Perez got tired of all the boilerplate code needed to create a
figure and multiple subplots when using the matplotlib API, and wrote
a :func:`~matplotlib.pyplot.subplots` helper function.  Basic usage
allows you to create the figure and an array of subplots with numpy
indexing (starts with 0).  e.g.::

  fig, axarr = plt.subplots(2, 2)
  axarr[0,0].plot([1,2,3])   # upper, left

See :doc:`/gallery/subplots_axes_and_figures/subplot` for several code examples.

Contour fixes and and triplot
-----------------------------

Ian Thomas has fixed a long-standing bug that has vexed our most
talented developers for years.  :func:`~matplotlib.pyplot.contourf`
now handles interior masked regions, and the boundaries of line and
filled contours coincide.

Additionally, he has contributed a new module :mod:`~matplotlib.tri` and
helper function :func:`~matplotlib.pyplot.triplot` for creating and
plotting unstructured triangular grids.

.. figure:: ../../gallery/images_contours_and_fields/images/sphx_glr_triplot_demo_001.png
   :target: ../../gallery/images_contours_and_fields/triplot_demo.html
   :align: center
   :scale: 50

multiple calls to show supported
--------------------------------

A long standing request is to support multiple calls to
:func:`~matplotlib.pyplot.show`.  This has been difficult because it
is hard to get consistent behavior across operating systems, user
interface toolkits and versions.  Eric Firing has done a lot of work
on rationalizing show across backends, with the desired behavior to
make show raise all newly created figures and block execution until
they are closed.  Repeated calls to show should raise newly created
figures since the last call.  Eric has done a lot of testing on the
user interface toolkits and versions and platforms he has access to,
but it is not possible to test them all, so please report problems to
the `mailing list
<https://mail.python.org/mailman/listinfo/matplotlib-users>`__
and `bug tracker
<https://github.com/matplotlib/matplotlib/issues>`__.


mplot3d graphs can be embedded in arbitrary axes
------------------------------------------------

You can now place an mplot3d graph into an arbitrary axes location,
supporting mixing of 2D and 3D graphs in the same figure, and/or
multiple 3D graphs in a single figure, using the "projection" keyword
argument to add_axes or add_subplot.  Thanks Ben Root.

.. plot::

    from mpl_toolkits.mplot3d.axes3d import get_test_data

    fig = plt.figure()

    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0, antialiased=False)
    ax.set_zlim3d(-1.01, 1.01)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    X, Y, Z = get_test_data(0.05)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    plt.show()

tick_params
-----------

Eric Firing wrote tick_params, a convenience method for changing the
appearance of ticks and tick labels. See pyplot function
:func:`~matplotlib.pyplot.tick_params` and associated Axes method
:meth:`~matplotlib.axes.Axes.tick_params`.

Lots of performance and feature enhancements
--------------------------------------------


* Faster magnification of large images, and the ability to zoom in to
  a single pixel

* Local installs of documentation work better

* Improved "widgets" -- mouse grabbing is supported

* More accurate snapping of lines to pixel boundaries

* More consistent handling of color, particularly the alpha channel,
  throughout the API

Much improved software carpentry
--------------------------------

The matplotlib trunk is probably in as good a shape as it has ever
been, thanks to improved `software carpentry
<https://software-carpentry.org/>`__.  We now have a `buildbot
<https://buildbot.net>`__ which runs a suite of `nose
<http://code.google.com/p/python-nose/>`__ regression tests on every
svn commit, auto-generating a set of images and comparing them against
a set of known-goods, sending emails to developers on failures with a
pixel-by-pixel image comparison.  Releases and release
bugfixes happen in branches, allowing active new feature development
to happen in the trunk while keeping the release branches stable.
Thanks to Andrew Straw, Michael Droettboom and other matplotlib
developers for the heavy lifting.

Bugfix marathon
---------------

Eric Firing went on a bug fixing and closing marathon, closing over 100 bugs on
the (now-closed) SourceForge bug tracker with help from Jae-Joon Lee, Michael
Droettboom, Christoph Gohlke and Michiel de Hoon.
