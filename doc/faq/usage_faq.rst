.. _usage-faq:

***************
Usage
***************

.. contents::
   :backlinks: none


.. _general_concepts:

General Concepts
================

:mod:`matplotlib` has an extensive codebase that can be daunting to many
new users. However, most of matplotlib can be understood with a fairly
simple conceptual framework and knowledge of a few important points.

Plotting requires action on a range of levels, from the most general
(e.g., 'contour this 2-D array') to the most specific (e.g., 'color
this screen pixel red'). The purpose of a plotting package is to assist
you in visualizing your data as easily as possible, with all the necessary
control -- that is, by using relatively high-level commands most of
the time, and still have the ability to use the low-level commands when
needed.

Therefore, everything in matplotlib is organized in a hierarchy. At the top
of the hierarchy is the matplotlib "state-machine environment" which is
provided by the :mod:`matplotlib.pyplot` module. At this level, simple
functions are used to add plot elements (lines, images, text, etc.) to
the current axes in the current figure.

.. note::
   Pyplot's state-machine environment behaves similarly to MATLAB and
   should be most familiar to users with MATLAB experience.

The next level down in the hierarchy is the first level of the object-oriented
interface, in which pyplot is used only for a few functions such as figure
creation, and the user explicitly creates and keeps track of the figure
and axes objects. At this level, the user uses pyplot to create figures,
and through those figures, one or more axes objects can be created. These
axes objects are then used for most plotting actions.

For even more control -- which is essential for things like embedding
matplotlib plots in GUI applications -- the pyplot level may be dropped
completely, leaving a purely object-oriented approach.

.. _pylab:

Matplotlib, pylab, and pyplot: how are they related?
====================================================

Matplotlib is the whole package; :mod:`pylab` is a module in matplotlib
that gets installed alongside :mod:`matplotlib`; and :mod:`matplotlib.pyplot`
is a module in matplotlib.

Pyplot provides the state-machine interface to the underlying plotting
library in matplotlib. This means that figures and axes are implicitly
and automatically created to achieve the desired plot. For example,
calling ``plot`` from pyplot will automatically create the necessary
figure and axes to achieve the desired plot. Setting a title will
then automatically set that title to the current axes object::

    import matplotlib.pyplot as plt

    plt.plot(range(10), range(10))
    plt.title("Simple Plot")
    plt.show()

Pylab combines the pyplot functionality (for plotting) with the numpy
functionality (for mathematics and for working with arrays)
in a single namespace, making that namespace
(or environment) even more MATLAB-like.
For example, one can call the `sin` and `cos` functions just like
you could in MATLAB, as well as having all the features of pyplot.

The pyplot interface is generally preferred for non-interactive plotting
(i.e., scripting). The pylab interface is convenient for interactive
calculations and plotting, as it minimizes typing. Note that this is
what you get if you use the *ipython* shell with the *-pylab* option,
which imports everything from pylab and makes plotting fully interactive.

.. _coding_styles:

Coding Styles
==================

When viewing this documentation and examples, you will find different
coding styles and usage patterns. These styles are perfectly valid
and have their pros and cons. Just about all of the examples can be
converted into another style and achieve the same results.
The only caveat is to avoid mixing the coding styles for your own code.

.. note::
   Developers for matplotlib have to follow a specific style and guidelines.
   See :ref:`developers-guide-index`.

Of the different styles, there are two that are officially supported.
Therefore, these are the preferred ways to use matplotlib.

For the preferred pyplot style, the imports at the top of your
scripts will typically be::

    import matplotlib.pyplot as plt
    import numpy as np

Then one calls, for example, np.arange, np.zeros, np.pi, plt.figure,
plt.plot, plt.show, etc. So, a simple example in this style would be::

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(0, 10, 0.2)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()

Note that this example used pyplot's state-machine to
automatically and implicitly create a figure and an axes. For full
control of your plots and more advanced usage, use the pyplot interface
for creating figures, and then use the object methods for the rest::

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(0, 10, 0.2)
    y = np.sin(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    plt.show()

Next, the same example using a pure MATLAB-style::

    from pylab import *
    x = arange(0, 10, 0.2)
    y = sin(x)
    plot(x, y)
    show()


So, why all the extra typing as one moves away from the pure
MATLAB-style?  For very simple things like this example, the only
advantage is academic: the wordier styles are more explicit, more
clear as to where things come from and what is going on.  For more
complicated applications, this explicitness and clarity becomes
increasingly valuable, and the richer and more complete object-oriented
interface will likely make the program easier to write and maintain.


