Below we describe several common approaches to plotting with Matplotlib.

.. contents::

The Pyplot API
--------------

The :mod:`matplotlib.pyplot` module contains functions that allow you to generate
many kinds of plots quickly. For examples that showcase the use
of the :mod:`matplotlib.pyplot` module, see the
:ref:`sphx_glr_tutorials_01_introductory_pyplot.py`
or the :ref:`pyplots_examples`. We also recommend that you look into
the object-oriented approach to plotting, described below.

.. currentmodule:: matplotlib.pyplot

.. autofunction:: plotting

The Object-Oriented API
-----------------------

Most of these functions also exist as methods in the
:class:`matplotlib.axes.Axes` class. You can use them with the
"Object Oriented" approach to Matplotlib.

While it is easy to quickly generate plots with the
:mod:`matplotlib.pyplot` module,
we recommend using the object-oriented approach for more control
and customization of your plots. See the methods in the
:meth:`matplotlib.axes.Axes` class for many of the same plotting functions.
For examples of the OO approach to Matplotlib, see the
:ref:`API Examples<api_examples>`.

Colors in Matplotlib
--------------------

There are many colormaps you can use to map data onto color values.
Below we list several ways in which color can be utilized in Matplotlib.

For a more in-depth look at colormaps, see the
:ref:`sphx_glr_tutorials_colors_colormaps.py` tutorial.

.. autofunction:: colormaps
