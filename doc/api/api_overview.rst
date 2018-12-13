API Overview
============

Below we describe several common approaches to plotting with Matplotlib.

.. contents::

The pyplot API
--------------

`matplotlib.pyplot` is a collection of command style functions that make
Matplotlib work like MATLAB. Each pyplot function makes some change to a
figure: e.g., creates a figure, creates a plotting area in a figure, plots
some lines in a plotting area, decorates the plot with labels, etc.

`.pyplot` is mainly intended for interactive plots and simple cases of
programmatic plot generation.

Further reading:

- The `matplotlib.pyplot` function reference
- :doc:`/tutorials/introductory/pyplot`
- :ref:`Pyplot examples <pyplots_examples>`

The object-oriented API
-----------------------

At its core, Matbplotlib is object-oriented. We recommend directly working
with the objects, if you need more control and customization of your plots.

In many cases you will create a `.Figure` and one or more
`~matplotlib.axes.Axes` using `.pyplot.subplots` and from then on only work
on these objects. However, it's also possible to create `.Figure`\ s
explicitly (e.g. when including them in GUI applications).

Further reading:

- `matplotlib.axes.Axes` and `matplotlib.figure.Figure` for an overview of
  plotting functions.
- Most of the :ref:`examples <examples-index>` use the object-oriented approach
  (except for the pyplot section)
- The list of :doc:`matplotlib modules </api/index>`.


The pylab API (disapproved)
---------------------------

.. warning::
   Since heavily importing into the global namespace may result in unexpected
   behavior, the use of pylab is strongly discouraged. Use `matplotlib.pyplot`
   instead.

`matplotlib.pylab` is a module that includes `matplotlib.pyplot`, `numpy`
and some additional functions within a single namespace. It's original puropse
was to mimic a MATLAB-like way of working by importing all functions into the
global namespace. This is considered bad style nowadays.
